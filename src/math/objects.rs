use num_traits::NumCast;
use std::borrow::Cow;
use std::fmt;
use std::ops;

use crate::expr_binop_from_enum;
use crate::lang::evaluator;
use crate::math::{Complex, Env, VarStack};
use crate::math::expressions::Expression;
use crate::math::matrices_and_vectors::{Matrix, Vector};
use crate::math::operations::*;
use crate::math::utils::{approx_eq, expect_int, format_trimmed, quo, Quo};
use crate::status::{ExtResult, Status};


/// Here, objects are things an identifier (e.g. "x") can represent, that is:
/// - A numerical constant (in the type we always use, f64)
/// - A constant vector/matrix
#[derive(Debug, Clone)]
pub enum Object {
    /// Returned when a parsed expression is the definition of a function
    /// => No specific f64 value can be assigned to it, but we can signal a successful definition.
    Success,
    /// May be returned by a derivative if the given expression is not differentiable.
    Undefined,
    Real(f64),
    Complex(Complex),
    Tuple(Vec<Object>),
    /// Vector/matrix operations are implemented for references to vectors/matrices anyway, so only
    /// using references to Vector/Matrix makes sense here.
    Vector(Vector),
    Matrix(Matrix),
    /// It might seem strange to have this be a variant of `Object`, but fundamentally, `Object` is the
    /// return type of `eval(expression)` and the evaluation of `d/dx f(x)` will be an expression again.
    /// One _could_ create another enum `EvalReturnType {Object, Expression}`, but this would be
    /// unnecessarily verbose.
    LiteralExpression(Expression)
}

#[derive(Debug, Clone, PartialEq)]
pub enum ObjType {
    /// `Success` or `Undefined`
    NonObject,
    /// Real or complex
    Scalar,
    Vector(usize),
    Matrix(usize, usize),
    /// Size of tuple doesn't matter since its doesn't support non-trivial operations anyway.
    Tuple,
    LiteralExpression
}
impl ObjType {
    /// Returns an `Object` representative of for the given `ObjType`.
    /// 
    /// When recursively parsing the type of e.g. a folded operation,
    /// this allows to inform of the type of the index variable via a `VarStack`.
    /// We do not construct a `TypeStack` instead because we only add very few frames
    /// to the stack meanwhile the already given initial `VarStack` could be relatively large.
    #[inline]
    pub fn representative(&self) -> Object {
        // I wrapped this in this way to underline in applications when the true value really doesn't matter
        self.zero()
    }

    /// Returns the corresponding `Object` filled with zeros.
    pub fn zero(&self) -> Object {
        match self {
            ObjType::NonObject => Object::Undefined,
            ObjType::Scalar => Object::Real(0.0),
            ObjType::Vector(n) => Object::Vector(Vector::zeros(*n)),
            ObjType::Matrix(m, n) => Object::Matrix(Matrix::zeros(*m, *n)),
            ObjType::Tuple => Object::Tuple(vec![]),
            ObjType::LiteralExpression => Object::LiteralExpression(Expression::None)
        }
    }

    /// Returns the corresponding multiplicative identity as `Object`.
    pub fn one(&self) -> Object {
        match self {
            ObjType::NonObject => Object::Undefined,
            ObjType::Scalar => Object::Real(1.0),
            ObjType::Vector(n) => Object::Vector(Vector { values: vec![1.0; *n] }), // Vector doesn't really have an identity
            ObjType::Matrix(m, _) => { // In order to get out an mxn matrix after multiplying this with an mxn matrix, this must be an mxm matrix
                Object::Matrix(Matrix::identity(*m))
            }
            ObjType::Tuple => Object::Tuple(vec![]),
            ObjType::LiteralExpression => Object::LiteralExpression(Expression::None)
        }
    }
}
impl fmt::Display for ObjType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ObjType::NonObject => write!(f, "NonObject"),
            ObjType::Scalar => write!(f, "Scalar"),
            ObjType::Vector(n) => write!(f, "Vector<{}>", n),
            ObjType::Matrix(m, n) => write!(f, "Matrix<{}x{}>", m, n),
            ObjType::Tuple => write!(f, "Tuple"),
            ObjType::LiteralExpression => write!(f, "Expression")
        }
    }
}

impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Object::Success => write!(f, "Object::Success"),
            Object::Undefined => write!(f, "Undefined"),
            Object::Real(x) => {
                if *x == f64::INFINITY {
                    write!(f, "∞")
                } else if *x == -f64::INFINITY {
                    write!(f, "-∞")
                } else {
                    write!(f, "{}", x)
                }
            }
            Object::Complex(x) => write!(f, "{}", x),
            Object::Tuple(x) => write!(f, "({})", x.iter().map(|o| format!("{:?}", o)).collect::<Vec<String>>().join(", ")),
            Object::Vector(x) => write!(f, "{:?}", x),
            Object::Matrix(x) => write!(f, "{:?}", x),
            Object::LiteralExpression(x) => write!(f, "{:?}", x),
        }
    }
}
impl Object {
    /// Formats an object to a string that may stretch over multiple lines.
    /// The lines will be returned as a vector of strings, not as a single string containing newline chars.
    pub fn to_multline(&self) -> Vec<String> {
        match self {
            Object::Success => vec!["Success".to_string()],
            Object::Undefined => vec!["Undefined".to_string()],
            Object::Real(x) => vec![x.to_string()],
            Object::Complex(x) => vec![format!("{x}")],
            Object::Tuple(x) => {
                let multlined_inner = x.iter().map(|o| o.to_multline()).collect::<Vec<Vec<String>>>();
                if multlined_inner.iter().all(|v| v.len() == 1) {
                    // If all inner objects have length one, simply join them into a single line with commas
                    vec![format!("({})", multlined_inner.into_iter().map(|v| v.into_iter().next().unwrap()).collect::<Vec<String>>().join(", "))]
                } else {
                    let mut res = vec!["(".to_string()];
                    let len = multlined_inner.len();
                    for (i, v) in multlined_inner.into_iter().enumerate() {
                        for l in v.into_iter() {
                            res.push(format!("    {l}"));
                        }
                        if i < len - 1 {
                            res.last_mut().unwrap().push(',');
                        }
                    }
                    res.push(")".to_string());
                    res
                }
            }
            Object::Vector(v) => vec![format!("[{}]", &v.values.iter().map(|x| format_trimmed(*x, 8)).collect::<Vec<String>>().join("; "))],
            Object::Matrix(x) => {
                // First, we go through all element to know how much space each column needs.
                let mut column_lengths = Vec::<usize>::with_capacity(x.n());
                let mut entries = Vec::<String>::with_capacity(x.n() * x.m()); // Notice this is the transposed version of the typical flattened vector
                for j in 0..x.n() {
                    column_lengths.push((0..x.m()).map(
                        |i| {
                            let s = format_trimmed(x.get(i, j), 8);
                            let len = s.chars().count();
                            entries.push(s);
                            len
                        }
                    ).max().unwrap_or(0))
                }
                // Cache locality isn't very important here since only so much can be displayed on a reasonable screen anyway
                let row_length = column_lengths.iter().sum::<usize>() + 2*x.n(); // Between two columns, add 2 spaces. Before the first columns and after the last one, only 1 space.
                let mut lines = vec![format!("╭{}╮", (0..row_length).map(|_| ' ').collect::<String>())];
                for i in 0..x.m() {
                    lines.push(format!("│ {}│", (0..x.n()).map(
                        |j| format!("{:^2$} {}", entries[j*x.m() + i], if j == x.n()-1 {""} else {" "}, column_lengths[j])
                    ).collect::<String>()));
                }
                lines.push(format!("╰{}╯", (0..row_length).map(|_| ' ').collect::<String>()));
                lines
            }
            Object::LiteralExpression(x) => x.to_multline()
        }
    }

    pub fn to_expression(&self) -> Expression {
        match self {
            Object::Success | Object::Undefined => Expression::None, // This would be a syntax error
            Object::Real(x) => Expression::Number(*x),
            Object::Complex(x) => crate::expr_binop!(Expression::Number(x.real), Add, crate::expr_binop!(Expression::Number(x.imag), Mul, Expression::Identifier("i".to_string()))),
            Object::Tuple(v) => Expression::Tuple(v.iter().map(|o| o.to_expression()).collect()),
            Object::Vector(v) => Expression::Vector(v.values.iter().map(|entry| Expression::Number(*entry)).collect()),
            Object::Matrix(x) => Expression::Matrix(
                x.m(), x.n(),
                x.iter_values().map(|entry| Expression::Number(*entry)).collect()
            ),
            Object::LiteralExpression(e) => e.clone()
        }
    }

    /// Returns `Ok(x)` if `self` is `Object::Real(x)`, otherwise `Err`.
    /// 
    /// Works on `&self` because `f64` is `Copy`.
    pub fn expect_float(&self) -> Result<f64, String> {
        match self {
            Object::Real(x) => Ok(*x),
            other => Err(format!("Expected float, got {other}."))
        }
    }

    /// Returns `Ok(x as T)` if `self` is `Object::Real(x)` for `x` close to an integer of type `T`, otherwise `Err`.
    /// 
    /// Works on `&self` because `T` is assumed to be `Copy`.
    pub fn expect_int<T: NumCast + Copy>(&self) -> Result<T, String> {
        let f = self.expect_float()?;
        expect_int(f).ok_or_else(|| format!("Expected number close to integer; got {f}."))
    }

    /// Returns `Ok(x as usize)` if `self` is `Object::Real(x)` for `x` close to a non-negative integer, otherwise `Err`.
    /// 
    /// Works on `&self` because `usize` is `Copy`.
    pub fn expect_nonnegative_int(&self) -> Result<usize, String> {
        let i = self.expect_int::<i64>()?;
        if i >= 0 {
            Ok(i as usize)
        } else {
            Err(format!("Expected non-negative integer, got {i}."))
        }
    }

    /// Returns `Ok(x)` if `self` is `Object::Matrix(x)`, otherwise `Err`.
    pub fn expect_matrix(self) -> Result<Matrix, String> {
        match self {
            Object::Matrix(x) => Ok(x),
            other => Err(format!("Expected matrix, got {other}."))
        }
    }

    /// Returns `Ok(x as bool)` if `x` is `Object::Real(x)` for `x in {0, 1}`, otherwise `Err`.
    pub fn expect_bool(self) -> Result<bool, String> {
        match self {
            Object::Real(1.0) => Ok(true),
            Object::Real(0.0) => Ok(false),
            other => Err(format!("Expected 0 or 1 when evaluating condition, got {:?}.", other))
        }
    }

    pub fn get_type(&self) -> ObjType {
        match self {
            Object::Undefined | Object::Success => ObjType::NonObject,
            Object::Real(_) | Object::Complex(_) => ObjType::Scalar,
            Object::Vector(v) => ObjType::Vector(v.len()),
            Object::Matrix(x) => ObjType::Matrix(x.m(), x.n()),
            Object::Tuple(_) => ObjType::Tuple,
            Object::LiteralExpression(_) => ObjType::LiteralExpression
        }
    }
}

impl<'a> ops::Mul<&'a Object> for f64 {
    type Output = Object;
    fn mul(self, rhs: &'a Object) -> Self::Output {
        match rhs {
            Object::Success => Object::Success,
            Object::Undefined => Object::Undefined,
            Object::Real(x) => Object::Real(self * x),
            Object::Complex(x) => Object::Complex(Complex{real: self * x.real, imag: self * x.imag}),
            Object::Tuple(x) => Object::Tuple(x.iter().map(|o| self * o).collect()),
            Object::Vector(x) => Object::Vector(self * x),
            Object::Matrix(x) => Object::Matrix(self * x),
            Object::LiteralExpression(expr) => Object::LiteralExpression(Expression::BinaryOperation(
                Box::new(Expression::Number(self)),
                BinaryOperation::Mul,
                Box::new(expr.clone())
            )),
        }
    }
}
impl ops::Mul<f64> for Object {
    type Output = Object;
    fn mul(self, rhs: f64) -> Self::Output {
        rhs * &self
    }
}
// Binary operations are implemented via macro in `status.rs`
impl ops::Neg for &Object {
    type Output = Result<Object, String>;
    fn neg(self) -> Self::Output {
        match self {
            Object::Success => Ok(Object::Success),
            Object::Undefined => Err("Operation 'Neg' not valid for undefined operand.".to_string()),
            Object::Real(x) => Ok(Object::Real(-x)),
            Object::Complex(x) => Ok(Object::Complex(-x)),
            Object::Tuple(x) => Ok(Object::Tuple(x.iter().map(|o| -o).collect::<Result<Vec<_>, _>>()?)),
            Object::Vector(x) => Ok(Object::Vector(-x)),
            Object::Matrix(x) => Ok(Object::Matrix(-x)),
            Object::LiteralExpression(expr) => Ok(Object::LiteralExpression(crate::expr_unary_op!(Neg, expr.clone()))),
        }
    }
}
impl ops::Neg for Object {
    type Output = Result<Object, String>;
    fn neg(self) -> Self::Output {
        match self {
            Object::Success => Ok(Object::Success),
            Object::Undefined => Err("Operation 'Neg' not valid for undefined operand.".to_string()),
            Object::Real(x) => Ok(Object::Real(-x)),
            Object::Complex(x) => Ok(Object::Complex(-&x)),
            Object::Tuple(x) => Ok(Object::Tuple(x.iter().map(|o| -o).collect::<Result<Vec<_>, _>>()?)),
            Object::Vector(x) => Ok(Object::Vector(-&x)),
            Object::Matrix(x) => Ok(Object::Matrix(-&x)),
            Object::LiteralExpression(expr) => Ok(Object::LiteralExpression(crate::expr_unary_op!(Neg, expr.clone()))),
        }
    }
}
impl ops::Not for Object {
    type Output = Result<Object, String>;
    fn not(self) -> Self::Output {
        match self {
            Object::Success => Ok(Object::Success),
            Object::Undefined => Err("Operation 'Not' not valid for undefined operand.".to_string()),
            Object::Real(x) => Ok(Object::Real(if x == 0.0 {1.0} else {0.0})),
            Object::Complex(x) => Ok(Object::Real(if x.real == 0.0 && x.imag == 0.0 {1.0} else {0.0})),
            Object::Tuple(v) => Ok(Object::Tuple(v.into_iter().map(|o| !o).collect::<Result<Vec<_>, _>>()?)),
            Object::Vector(v) => Ok(Object::Vector(v.transform(|x| if x == 0.0 {1.0} else {0.0}))),
            Object::Matrix(m) => Ok(Object::Matrix(m.transform(|x| if x == 0.0 {1.0} else {0.0}))),
            Object::LiteralExpression(e) => Ok(Object::LiteralExpression(crate::expr_unary_op!(Not, e))),
        }
    }
}
impl Object {
    pub fn squared(self) -> ExtResult {
        try_operation(&self, &Object::Real(2.0), &BinaryOperation::Pow(false), None)
    }
}

/// Type abbreviation. A _direct function_ can take any number of `Object` and `Expression` and, optionally, a `VarStack` and an `Env`.
pub type DirectFunction = Box<dyn for<'a, 'b, 'c, 'd> Fn(&'a [Object], &'b [Expression], Option<(&'c VarStack, &'d mut Env)>) -> ExtResult + Send + Sync>;

/// Different representations for a function
#[derive(Clone)]
pub enum FunctionRepr {
    /// 1. The list of identifiers of the arguments (in order to parse the literal expression correctly).
    ///    These will be prefixed with `___tmp_` to avoid confusion with normal constants. Note that
    ///    the user is not allowed to define a variable whose name starts with three underscores.
    /// 2. E.g. `"5 * ___tmp_x + 2"` where `arguments` is `["___tmp_x"]`. The variable names here will already be prefixed.
    ByExpression(Vec<String>, Expression),
    /// Contains a reference to a default function as well as the corresponding argtype mask.
    /// 
    /// A _mask_ is a tuple `(m, n, k)` signifying the first `m` arguments should be evaluated, the next `n` arguments
    /// should not be evaluated. Let `x` be the number of arguments thereafter.
    /// The component `k` in the mask means that the first `n/k` of the remaining `n` arguments should be left unevaluated
    /// and the rest should be evaluated. If `k == 0`, this means all remaining arguments _should_ be evaluated.
    Direct(&'static DirectFunction, (usize, usize, usize))
}

impl fmt::Debug for FunctionRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionRepr::ByExpression(argnames, expr) => write!(f, "({}) ↦ {}", argnames.join(", "), expr),
            FunctionRepr::Direct(..) => write!(f, "<Closure>")
        }
    }
}


/// Simplifies notation in 'try_operation'. LHS and RHS should be a float on one side and a vector/matrix on the other side.
fn _op_mv_float<T, U, V>(lhs: T, rhs: U, op: &BinaryOperation) -> Result<V, String>
where T: std::ops::Mul<U, Output=V> + std::ops::Div<U, Output=V> + std::ops::Rem<U, Output=V> + Quo<U, Output=V> + fmt::Display, U: fmt::Display {
    match op {
        BinaryOperation::Mul => Ok(lhs * rhs),
        BinaryOperation::Div => Ok(lhs / rhs),
        BinaryOperation::Rem => Ok(lhs % rhs),
        BinaryOperation::Quo => Ok(Quo::quo(lhs, rhs)),
        // All other operations are not possible (again, I write them out explicitely to be forced to review this snippet if I add new operations)
        BinaryOperation::Add | BinaryOperation::Sub | BinaryOperation::Pow(_) | BinaryOperation::And | BinaryOperation::Or | BinaryOperation::Comp(..)
            => Err(format!("Operation '{}' invalid for operands {} and {}.", op, lhs, rhs))
    }
}

/// Returns 1 if the comparison succeeds, 0 otherwise
fn compare(x: f64, y: f64, comp: &Comparison) -> bool {
    match comp {
        Comparison::Eq => approx_eq(x, y),
        Comparison::Neq => !approx_eq(x, y),
        Comparison::Gt => x > y,
        Comparison::Lt => x < y,
        Comparison::Ge => x >= y || approx_eq(x, y),
        Comparison::Le => x < y || approx_eq(x, y)
    }
}
/// Returns Object::Real(1) if the comparison succeeds, Object::Real(0) if it doesn't, Object::Undefined if e.g. trying `z_1 < z_2`.
fn compare_complex(x: &Complex, y: &Complex, comp: &Comparison) -> Object {
    match comp {
        Comparison::Eq => Object::Real((approx_eq(x.real, y.real) && approx_eq(x.imag, y.imag)) as i8 as f64),
        Comparison::Neq => Object::Real(!(approx_eq(x.real, y.real) && approx_eq(x.imag, y.imag)) as i8 as f64),
        Comparison::Gt | Comparison::Lt | Comparison::Ge | Comparison::Le => Object::Undefined
    }
}

/// Attempts to perform the given operation 'op' on the given operands `lhs` and `rhs`.
/// 
/// `context` is a parameter that is only rarely required, namely when `op` is a comparison and at least one operand is a literal expression.
/// It is made optional to allow clean overloading of standard arithmetic operators for `Object` without having to pass a `VarStack` and an
/// `Env` that won't be used anyway.
/// 
/// I don't really see any significantly better way of doing this than to compare types, since we need to put the output into an `Object` too
/// and we must take care of possible dimension mismatches too.
pub fn try_operation(lhs: &Object, rhs: &Object, op: &BinaryOperation, context: Option<(&VarStack, &mut Env)>) -> ExtResult {
    match (lhs, rhs, op) {
        // Below types can't do any binary operations.
        (o @ (Object::Success | Object::Undefined | Object::Tuple(_)), ..) | (_, o @ (Object::Success | Object::Undefined | Object::Tuple(_)), _) => {
            Err(format!("Objects of type `{}` to not support binary operations.", o.get_type()))
        }
        (Object::LiteralExpression(lexpr), Object::LiteralExpression(rexpr), BinaryOperation::Comp(c, precision_expr)) => {
            context.ok_or_else(|| format!("Operation '{}' requires `VarStack` and `Env` when applied to a literal expression.", c))
            .and_then(
                |(varstack, env)|
                evaluator::compare_expressions(lexpr, rexpr, *c, precision_expr, varstack, env)
            )
        }
        (Object::LiteralExpression(lexpr), _, BinaryOperation::Comp(c, precision_expr)) => {
            context.ok_or_else(|| format!("Operation '{}' requires `VarStack` and `Env` when applied to a literal expression.", c))
            .and_then(
                |(varstack, env)|
                evaluator::Evaluable::from_expr(lexpr, varstack, env).and_then(|s| s.try_map_flatten(
                    |(lhs_ev, free_variables)| {
                        if let evaluator::Evaluable::Object(lhs_eval) = lhs_ev {
                            try_operation(
                                lhs_eval.as_ref(),
                                rhs,
                                &BinaryOperation::Comp(*c, None),
                                Some((varstack, env))
                            )
                        } else {
                            evaluator::test_function_equality(
                                &lhs_ev,
                                &evaluator::Evaluable::Object(Cow::Borrowed(rhs)),
                                &free_variables,
                                *c,
                                precision_expr,
                                varstack,
                                env
                            )
                        }
                    }
                ))
            )
        }
        (_, Object::LiteralExpression(rexpr), BinaryOperation::Comp(c, precision_expr)) => {
            context.ok_or_else(|| format!("Operation '{}' requires `VarStack` and `Env` when applied to a literal expression.", c))
            .and_then(
                |(varstack, env)|
                evaluator::Evaluable::from_expr(rexpr, varstack, env).and_then(|s| s.try_map_flatten(
                    |(rhs_ev, free_variables)| {
                        if let evaluator::Evaluable::Object(rhs_eval) = rhs_ev {
                            try_operation(
                                lhs,
                                rhs_eval.as_ref(),
                                &BinaryOperation::Comp(*c, None),
                                Some((varstack, env))
                            )
                        } else {
                            evaluator::test_function_equality(
                                &evaluator::Evaluable::Object(Cow::Borrowed(lhs)),
                                &rhs_ev,
                                &free_variables,
                                *c,
                                precision_expr,
                                varstack,
                                env
                            )
                        }
                    }
                ))
            )
        }
        // If at least one of both `lhs` and `rhs` is a literal expression, the binop should yield a literal expression too.
        (Object::LiteralExpression(_), _, op) | (_, Object::LiteralExpression(_), op) => {
            Ok(Status::ok(Object::LiteralExpression(expr_binop_from_enum!(lhs.to_expression(), op.clone(), rhs.to_expression()))))
        }

        (Object::Real(x), Object::Real(y), BinaryOperation::Add) => Ok(Status::ok(Object::Real(x + y))),
        (Object::Real(x), Object::Real(y), BinaryOperation::Sub) => Ok(Status::ok(Object::Real(x - y))),
        (Object::Real(x), Object::Real(y), BinaryOperation::Mul) => Ok(Status::ok(Object::Real(x * y))),
        (Object::Real(x), Object::Real(y), BinaryOperation::Div) => Ok(Status::ok(Object::Real(x / y))),
        (Object::Real(x), Object::Real(y), BinaryOperation::Rem) => Ok(Status::ok(Object::Real(x.rem_euclid(*y)))),
        (Object::Real(x), Object::Real(y), BinaryOperation::Quo) => Ok(Status::ok(Object::Real(quo(*x, *y)))),
        (Object::Real(x), Object::Real(y), BinaryOperation::Pow(_)) => Ok(Status::ok(Object::Real(x.powf(*y)))),
        (Object::Real(x), Object::Real(y), BinaryOperation::Comp(c, _)) => Ok(Status::ok(Object::Real(compare(*x, *y, c) as i8 as f64))),
        (Object::Real(x), Object::Real(y), BinaryOperation::And) => Ok(Status::ok(Object::Real(if *x != 0.0 && *y != 0.0 {1.0} else {0.0}))),
        (Object::Real(x), Object::Real(y), BinaryOperation::Or) => Ok(Status::ok(Object::Real(if *x != 0.0 || *y != 0.0 {1.0} else {0.0}))),

        (Object::Real(x), Object::Complex(z), BinaryOperation::Add)
        | (Object::Complex(z), Object::Real(x), BinaryOperation::Add) => Ok(Status::ok(Object::Complex(Complex { real: x + z.real, imag: z.imag }))),
        (Object::Real(x), Object::Complex(z), BinaryOperation::Sub) => Ok(Status::ok(Object::Complex(Complex { real: x - z.real, imag: -z.imag }))),
        (Object::Real(x), Object::Complex(z), BinaryOperation::Mul)
        | (Object::Complex(z), Object::Real(x), BinaryOperation::Mul) => Ok(Status::ok(Object::Complex(Complex { real: x * z.real, imag: x * z.imag }))),
        (Object::Real(x), Object::Complex(z), BinaryOperation::Div) => Ok(Status::ok({
            let inv = z.inv();
            Object::Complex(Complex { real: x * inv.real, imag: x * inv.imag })
        })),
        (Object::Real(x), Object::Complex(z), BinaryOperation::Pow(_)) => Ok(Status::ok(Object::Complex(Complex { real: *x, imag: 0.0 }.pow(z)))),
        (Object::Real(x), Object::Complex(z), BinaryOperation::Comp(c, _)) => Ok(Status::ok(compare_complex(&Complex { real: *x, imag: 0.0 }, z, c))),
        (Object::Real(x), Object::Complex(z), BinaryOperation::And)
        | (Object::Complex(z), Object::Real(x), BinaryOperation::And) => Ok(Status::ok(Object::Real(if *x != 0.0 && (z.real != 0.0 || z.imag != 0.0) {1.0} else {0.0}))),
        (Object::Real(x), Object::Complex(z), BinaryOperation::Or)
        | (Object::Complex(z), Object::Real(x), BinaryOperation::Or) => Ok(Status::ok(Object::Real(if *x != 0.0 || z.real != 0.0 || z.imag != 0.0 {1.0} else {0.0}))),

        (Object::Real(x), Object::Vector(y), _) => Ok(Status::ok(Object::Vector(_op_mv_float(*x, y, op)?))),
        (Object::Real(x), Object::Matrix(y), _) => Ok(Status::ok(Object::Matrix(_op_mv_float(*x, y, op)?))),

        (Object::Complex(z), Object::Real(x), BinaryOperation::Sub) => Ok(Status::ok(Object::Complex(Complex { real: z.real - x, imag: z.imag }))),
        (Object::Complex(z), Object::Real(x), BinaryOperation::Div) => Ok(Status::ok(Object::Complex(Complex { real: z.real / x, imag: z.imag / x }))),
        (Object::Complex(z), Object::Real(x), BinaryOperation::Rem) => Ok(Status::ok(Object::Complex(Complex { real: z.real.rem_euclid(*x), imag: z.imag.rem_euclid(*x) }))),
        (Object::Complex(z), Object::Real(x), BinaryOperation::Quo) => Ok(Status::ok(Object::Complex(Complex { real: quo(z.real, *x), imag: quo(z.imag, *x) }))),
        (Object::Complex(z), Object::Real(x), BinaryOperation::Pow(_)) => Ok(Status::ok(Object::Complex(z.pow(&Complex { real: *x, imag: 0.0 })))),
        (Object::Complex(z), Object::Real(x), BinaryOperation::Comp(c, _)) => Ok(Status::ok(compare_complex(z, &Complex { real: *x, imag: 0.0 }, c))),

        (Object::Complex(z), Object::Complex(w), BinaryOperation::Add) => Ok(Status::ok(Object::Complex(z + w))),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Sub) => Ok(Status::ok(Object::Complex(z - w))),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Mul) => Ok(Status::ok(Object::Complex(z * w))),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Div) => Ok(Status::ok(Object::Complex(z / w))),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Pow(_)) => Ok(Status::ok(Object::Complex(z.pow(w)))),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Comp(c, _)) => Ok(Status::ok(compare_complex(z, w, c))),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::And) => Ok(Status::ok(Object::Real(
            if (w.real != 0.0 || w.imag != 0.0) && (z.real != 0.0 || z.imag != 0.0) {1.0} else {0.0}
        ))),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Or) => Ok(Status::ok(Object::Real(
            if w.real != 0.0 || w.imag != 0.0 || z.real != 0.0 || z.imag != 0.0 {1.0} else {0.0}
        ))),

        (Object::Complex(_), Object::Vector(_) | Object::Matrix(_), _) | (Object::Vector(_) | Object::Matrix(_), Object::Complex(_), _) => {
            Err("Complex vectors/matrices aren't supported yet.".to_string())
        }

        (Object::Vector(x), Object::Real(y), _) => Ok(Status::ok(Object::Vector(_op_mv_float(x, *y, op)?))),
        (Object::Vector(x), Object::Vector(y), BinaryOperation::Add) if let Some(res) = x + y => Ok(Status::ok(Object::Vector(res))),
        (Object::Vector(x), Object::Vector(y), BinaryOperation::Sub) if let Some(res) = x - y => Ok(Status::ok(Object::Vector(res))),
        (Object::Vector(x), Object::Vector(y), BinaryOperation::Mul) if let Some(res) = x * y => Ok(Status::ok(Object::Real(res))),
        (Object::Vector(x), Object::Vector(y), BinaryOperation::Comp(c, _)) if x.len() == y.len() => Ok(Status::ok(Object::Real(
            if c.check_all() {
                x.iter().zip(y.iter()).all(|(a, b)| compare(a, b, c))
            } else {
                x.iter().zip(y.iter()).any(|(a, b)| compare(a, b, c))
            } as i8 as f64
        ))),
        (Object::Vector(x), Object::Matrix(y), BinaryOperation::Mul) if let Some(res) = x * y => Ok(Status::ok(Object::Vector(res))),

        (Object::Matrix(x), Object::Real(y), BinaryOperation::Pow(_)) if x.m() == x.n() && let Some(exponent) = expect_int::<i64>(*y) => {
            if exponent >= 0 {
                Ok(Status::ok(Object::Matrix(x.pow(exponent as u64).ok_or(format!("Matrix must be quadratic to apply `Pow` (got size {}x{})", x.m(), x.n()))?)))
            } else {
                let inv = x.inv().ok_or(format!("Matrix is not invertible: {:?}", x))?;
                Ok(Status::ok(Object::Matrix(inv.pow((-exponent) as u64).unwrap()))) // `unwrap` is safe since if `inv` exists, it is necessarily quadratic.
            }
        }
        (Object::Matrix(x), Object::Real(y), _) => Ok(Status::ok(Object::Matrix(_op_mv_float(x, *y, op)?))),
        (Object::Matrix(x), Object::Vector(y), BinaryOperation::Mul) if let Some(res) = x * y => Ok(Status::ok(Object::Vector(res))),
        (Object::Matrix(x), Object::Matrix(y), BinaryOperation::Add) if let Some(res) = x + y => Ok(Status::ok(Object::Matrix(res))),
        (Object::Matrix(x), Object::Matrix(y), BinaryOperation::Sub) if let Some(res) = x - y => Ok(Status::ok(Object::Matrix(res))),
        (Object::Matrix(x), Object::Matrix(y), BinaryOperation::Mul) if let Some(res) = x * y => Ok(Status::ok(Object::Matrix(res))),
        (Object::Matrix(x), Object::Matrix(y), BinaryOperation::Comp(c, _)) if x.m() == y.m() && x.n() == y.n() => Ok(Status::ok(Object::Real(
            if c.check_all() {
                x.iter().zip(y.iter()).all(|(a, b)| compare(a, b, c))
            } else {
                x.iter().zip(y.iter()).any(|(a, b)| compare(a, b, c))
            } as i8 as f64
        ))),

        _ => Err(format!("Operation '{}' invalid for operands {} and {}.", op, lhs, rhs))
    }
}