use num_traits::NumCast;
use std::borrow::Cow;
use std::fmt;

use crate::{expr_binop, expr_binop_from_enum, expr_unary_op, ok};
use crate::lang::evaluator;
use crate::math::{Complex, Env, Matrix, VarStack, Vector};
use crate::math::expressions::Expression;
use crate::math::operations::*;
use crate::math::traits::*;
use crate::math::utils::{all_res, any_res, approx_eq, expect_int};
use crate::status::{ExtResult, Status};


/// Here, objects are things an identifier (e.g. "x") can represent, that is:
/// - A numerical constant (in the type we always use, f64)
/// - A constant vector/matrix
/// - A tuple
/// - A literal expression (cf. documentation of corresponding variant)
#[derive(Debug, Clone)]
pub enum Object {
    /// Returned when a parsed expression is the definition of a function
    /// => No specific f64 value can be assigned to it, but we can signal a successful definition.
    Success,
    /// May be returned by a derivative if the given expression is not differentiable.
    Undefined,
    Real(f64),
    Complex(Complex<f64>),
    Tuple(Vec<Object>),
    Vector(VectorType),
    Matrix(MatrixType),
    /// It might seem strange to have this be a variant of `Object`, but fundamentally, `Object` is the
    /// return type of `eval(expression)` and the evaluation of `d/dx f(x)` will be an expression again.
    /// One _could_ create another enum `EvalReturnType {Object, Expression}`, but this would be
    /// unnecessarily verbose.
    LiteralExpression(Expression)
}

#[derive(Debug, Clone)]
pub enum VectorType {
    Real(Vector<f64>),
    Complex(Vector<Complex<f64>>)
}
#[derive(Debug, Clone)]
pub enum MatrixType {
    Real(Matrix<f64>),
    Complex(Matrix<Complex<f64>>)
}

impl Vector<f64> {
    pub fn wrap_in_type(self) -> VectorType {
        VectorType::Real(self)
    }
}
impl Vector<Complex<f64>> {
    pub fn wrap_in_type(self) -> VectorType {
        VectorType::Complex(self)
    }
}
impl Matrix<f64> {
    pub fn wrap_in_type(self) -> MatrixType {
        MatrixType::Real(self)
    }
}
impl Matrix<Complex<f64>> {
    pub fn wrap_in_type(self) -> MatrixType {
        MatrixType::Complex(self)
    }
}

#[macro_export]
macro_rules! dispatch_vector {
    ($self:expr, $v:ident => $body:expr) => {
        match $self {
            crate::math::objects::VectorType::Real($v) => $body,
            crate::math::objects::VectorType::Complex($v) => $body
        }
    }
}
/// Works on both vectors and matrices
macro_rules! compare_componentwise {
    ($v:expr, $w:expr, $c:expr, $closure:expr) => {
        ok!(Object::from_bool(
            if $c.check_all() {
                all_res(
                    $v.iter().zip($w.iter()),
                    $closure
                )?
            } else {
                any_res(
                    $v.iter().zip($w.iter()),
                    $closure
                )?
            }
        ))
    }
}
#[macro_export]
macro_rules! dispatch_matrix {
    ($self:expr, $m:ident => $body:expr) => {
        match $self {
            crate::math::objects::MatrixType::Real($m) => $body,
            crate::math::objects::MatrixType::Complex($m) => $body
        }
    }
}
#[macro_export]
macro_rules! map_mv {
    (($self:expr, $t:ident) -> $output_type:ty {$v:ident => $body:expr}) => {
        paste::paste!{
            match $self {
                crate::math::objects::[<$t Type>]::Real($v) => <$output_type>::Real($body),
                crate::math::objects::[<$t Type>]::Complex($v) => <$output_type>::Complex($body)
            }
        }
    }
}
#[macro_export]
macro_rules! try_map_mv {
    (($self:expr, $t:ident) -> $output_type:ty {$v:ident => $body:expr}) => {
        paste::paste!{
            match $self {
                crate::math::objects::[<$t Type>]::Real($v) => {$body}.map(<$output_type>::Real),
                crate::math::objects::[<$t Type>]::Complex($v) => {$body}.map(<$output_type>::Complex)
            }
        }
    }
}
macro_rules! try_map_two_mv {
    (($lhs:expr, $lhs_type:ident; $rhs:expr, $rhs_type:ident) -> $output_type:ty {($v:ident, $w:ident) => $body:expr}) => {
        paste::paste!{
            match ($lhs, $rhs) {
                (crate::math::objects::[<$lhs_type Type>]::Real($v), crate::math::objects::[<$rhs_type Type>]::Real($w)) => {$body}.map(<$output_type>::Real),
                (crate::math::objects::[<$lhs_type Type>]::Real($v), crate::math::objects::[<$rhs_type Type>]::Complex($w)) => {$body}.map(<$output_type>::Complex),
                (crate::math::objects::[<$lhs_type Type>]::Complex($v), crate::math::objects::[<$rhs_type Type>]::Real($w)) => {$body}.map(<$output_type>::Complex),
                (crate::math::objects::[<$lhs_type Type>]::Complex($v), crate::math::objects::[<$rhs_type Type>]::Complex($w)) => {$body}.map(<$output_type>::Complex),
            }
        }
    }
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
            ObjType::Vector(n) => Object::Vector(VectorType::Real(Vector::zeros(*n))),
            ObjType::Matrix(m, n) => Object::Matrix(MatrixType::Real(Matrix::zeros(*m, *n))),
            ObjType::Tuple => Object::Tuple(vec![]),
            ObjType::LiteralExpression => Object::LiteralExpression(Expression::None)
        }
    }

    /// Returns the corresponding multiplicative identity as `Object`.
    pub fn one(&self) -> Object {
        match self {
            ObjType::NonObject => Object::Undefined,
            ObjType::Scalar => Object::Real(1.0),
            ObjType::Vector(n) => Object::Vector(VectorType::Real(Vector { values: vec![1.0; *n] })), // Vector doesn't really have an identity
            ObjType::Matrix(m, _) => { // In order to get out an mxn matrix after multiplying this with an mxn matrix, this must be an mxm matrix
                Object::Matrix(MatrixType::Real(Matrix::identity(*m)))
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
            Object::Vector(x) => dispatch_vector!(x, v => write!(f, "{:?}", v)),
            Object::Matrix(x) => dispatch_matrix!(x, m => write!(f, "{:?}", m)),
            Object::LiteralExpression(x) => write!(f, "{:?}", x),
        }
    }
}
impl Object {
    pub fn from_bool(b: bool) -> Self {
        Object::Real(b as i8 as f64)
    }

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
            Object::Vector(x) => dispatch_vector!(x, v => vec![format!("[{}]", &v.values.iter().map(|x| x.format_trimmed(8)).collect::<Vec<String>>().join("; "))]),
            Object::Matrix(x) => dispatch_matrix!(x, m => {
                // First, we go through all element to know how much space each column needs.
                let mut column_lengths = Vec::<usize>::with_capacity(m.n());
                let mut entries = Vec::<String>::with_capacity(m.n() * m.m()); // Notice this is the transposed version of the typical flattened vector
                for j in 0..m.n() {
                    column_lengths.push((0..m.m()).map(
                        |i| {
                            let s = m.get(i, j).format_trimmed(8);
                            let len = s.chars().count();
                            entries.push(s);
                            len
                        }
                    ).max().unwrap_or(0))
                }
                // Cache locality isn't very important here since only so much can be displayed on a reasonable screen anyway
                let row_length = column_lengths.iter().sum::<usize>() + 2*m.n(); // Between two columns, add 2 spaces. Before the first columns and after the last one, only 1 space.
                let mut lines = vec![format!("╭{}╮", (0..row_length).map(|_| ' ').collect::<String>())];
                for i in 0..m.m() {
                    lines.push(format!("│ {}│", (0..m.n()).map(
                        |j| format!("{:^2$} {}", entries[j*m.m() + i], if j == m.n()-1 {""} else {" "}, column_lengths[j])
                    ).collect::<String>()));
                }
                lines.push(format!("╰{}╯", (0..row_length).map(|_| ' ').collect::<String>()));
                lines
            }),
            Object::LiteralExpression(x) => x.to_multline()
        }
    }

    pub fn to_expression(&self) -> Expression {
        match self {
            Object::Success | Object::Undefined => Expression::None, // This would be a syntax error
            Object::Real(x) => Expression::Number(*x),
            Object::Complex(x) => expr_binop!(Expression::Number(x.real), Add, expr_binop!(Expression::Number(x.imag), Mul, Expression::Identifier("i".to_string()))),
            Object::Tuple(v) => Expression::Tuple(v.iter().map(|o| o.to_expression()).collect()),
            Object::Vector(x) => dispatch_vector!(x, v => Expression::Vector(v.values.iter().map(|entry| entry.to_expression()).collect())),
            Object::Matrix(x) => dispatch_matrix!(x, m => Expression::Matrix(
                m.m(), m.n(),
                m.iter().map(|entry| entry.to_expression()).collect()
            )),
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

    /// Expects the given vector to only contain floats or complex numbers (a mixture of both is allowed).
    /// 
    /// On success, returns a tuple `(reals, complexes)`.
    fn expect_only_scalars(v: impl Iterator<Item=Result<Object, String>>) -> Result<(Vec<f64>, Vec<Complex<f64>>), String> {
        let mut reals = Vec::new();
        let mut complexes = Vec::new();
        for obj in v {
            match obj? {
                Object::Real(x) => {
                    if complexes.is_empty() {
                        reals.push(x)
                    } else {
                        complexes.push(x.to_complex())
                    }
                }
                Object::Complex(z) => {
                    if complexes.is_empty() {
                        complexes.extend(reals.iter().map(|&x| x.to_complex()));
                        reals.clear();
                    }
                    complexes.push(z);
                }
                other => return Err(format!("Expected float, got {other}."))
            }
        }
        Ok((reals, complexes))
    }

    /// Expects the given vector to only contain floats or complex numbers (a mixture of both is allowed).
    /// 
    /// Returns a `Object::Vector` which, if possible, it kept as the variant `VectorType::Real`.
    pub fn expect_scalar_vector(v: impl Iterator<Item=Result<Object, String>>) -> Result<Object, String> {
        Object::expect_only_scalars(v)
        .map(
            |(reals, complexes)| Object::Vector(
                if complexes.is_empty() {
                    VectorType::Real(Vector{values: reals})
                } else {
                    VectorType::Complex(Vector{values: complexes})
                }
            )
        )
    }
    /// Expects the given vector to only contain floats or complex numbers (a mixture of both is allowed).
    /// 
    /// Returns a `Object::Matrix` which, if possible, it kept as the variant `MatrixType::Real`.
    pub fn expect_scalar_matrix(m: usize, n: usize, v: impl Iterator<Item=Result<Object, String>>) -> Result<Object, String> {
        Object::expect_only_scalars(v)
        .map(
            |(reals, complexes)|
            Object::Matrix(
                if complexes.is_empty() {
                    MatrixType::Real(Matrix::from(m, n, reals))
                } else {
                    MatrixType::Complex(Matrix::from(m, n, complexes))
                }
            )
        )
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
    pub fn expect_matrix(self) -> Result<MatrixType, String> {
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
            Object::Vector(x) => dispatch_vector!(x, v => ObjType::Vector(v.len())),
            Object::Matrix(x) => dispatch_matrix!(x, m => ObjType::Matrix(m.m(), m.n())),
            Object::Tuple(_) => ObjType::Tuple,
            Object::LiteralExpression(_) => ObjType::LiteralExpression
        }
    }
}

impl<'a> Mul<&'a Object> for f64 {
    type Output = Object;
    fn mul(self, rhs: &'a Object) -> Self::Output {
        match rhs {
            Object::Success => Object::Success,
            Object::Undefined => Object::Undefined,
            Object::Real(x) => Object::Real(self * x),
            Object::Complex(x) => Object::Complex(Complex{real: self * x.real, imag: self * x.imag}),
            Object::Tuple(x) => Object::Tuple(x.iter().map(|o| self.mul(o)).collect()),
            Object::Vector(x) => Object::Vector(map_mv!((x, Vector) -> VectorType {v => self.mul(v)})),
            Object::Matrix(x) => Object::Matrix(map_mv!((x, Matrix) -> MatrixType {m => self.mul(m)})),
            Object::LiteralExpression(expr) => Object::LiteralExpression(expr_binop!(Expression::Number(self), Mul, expr.clone()))
        }
    }
}
impl Mul<f64> for Object {
    type Output = Object;
    fn mul(self, rhs: f64) -> Self::Output {
        rhs.mul(&self)
    }
}
impl Div<f64> for Object {
    type Output = Object;
    fn div(self, rhs: f64) -> Self::Output {
        self.mul(1.0 / rhs)
    }
}
// Binary operations are implemented via macro in `status.rs`
impl Neg for &Object {
    type Output = Result<Object, String>;
    fn neg(self) -> Self::Output {
        match self {
            Object::Success => Ok(Object::Success),
            Object::Undefined => Err("Operation 'Neg' not valid for undefined operand.".to_string()),
            Object::Real(x) => Ok(Object::Real(-x)),
            Object::Complex(x) => Ok(Object::Complex(x.neg())),
            Object::Tuple(x) => Ok(Object::Tuple(x.iter().map(|o| o.neg()).collect::<Result<Vec<_>, _>>()?)),
            Object::Vector(x) => Ok(Object::Vector(map_mv!((x, Vector) -> VectorType {v => v.neg()}))),
            Object::Matrix(x) => Ok(Object::Matrix(map_mv!((x, Matrix) -> MatrixType {m => m.neg()}))),
            Object::LiteralExpression(expr) => Ok(Object::LiteralExpression(expr_unary_op!(Neg, expr.clone()))),
        }
    }
}
impl Neg for Object {
    type Output = Result<Object, String>;
    fn neg(self) -> Self::Output {
        match self {
            Object::Success => Ok(Object::Success),
            Object::Undefined => Err("Operation 'Neg' not valid for undefined operand.".to_string()),
            Object::Real(x) => Ok(Object::Real(-x)),
            Object::Complex(x) => Ok(Object::Complex(x.neg())),
            Object::Tuple(x) => Ok(Object::Tuple(x.iter().map(|o| o.neg()).collect::<Result<Vec<_>, _>>()?)),
            Object::Vector(x) => Ok(Object::Vector(map_mv!((x, Vector) -> VectorType {v => v.neg()}))),
            Object::Matrix(x) => Ok(Object::Matrix(map_mv!((x, Matrix) -> MatrixType {m => m.neg()}))),
            Object::LiteralExpression(expr) => Ok(Object::LiteralExpression(expr_unary_op!(Neg, expr.clone()))),
        }
    }
}
impl std::ops::Not for Object {
    type Output = Result<Object, String>;
    fn not(self) -> Self::Output {
        match self {
            Object::Success => Ok(Object::Success),
            Object::Undefined => Err("Operation 'Not' not valid for undefined operand.".to_string()),
            Object::Real(x) => Ok(Object::Real(if x == 0.0 {1.0} else {0.0})),
            Object::Complex(x) => Ok(Object::Real(if x.real == 0.0 && x.imag == 0.0 {1.0} else {0.0})),
            Object::Tuple(v) => Ok(Object::Tuple(v.into_iter().map(|o| !o).collect::<Result<Vec<_>, _>>()?)),
            Object::Vector(x) => Ok(Object::Vector(map_mv!((x, Vector) -> VectorType {
                v => v.transform(|x| {
                    fn zero<T: Scalar>(_sample: T) -> T {T::zero()}
                    fn one<T: Scalar>(_sample: T) -> T {T::one()}
                    if x.is_zero() {
                        one(x)
                    } else {
                        zero(x)
                    }
                })
            }))),
            Object::Matrix(x) => Ok(Object::Matrix(map_mv!((x, Matrix) -> MatrixType {
                m => m.transform(|x| {
                    fn zero<T: Scalar>(_sample: T) -> T {T::zero()}
                    fn one<T: Scalar>(_sample: T) -> T {T::one()}
                    if x.is_zero() {
                        one(x)
                    } else {
                        zero(x)
                    }
                })
            }))),
            Object::LiteralExpression(e) => Ok(Object::LiteralExpression(expr_unary_op!(Not, e))),
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


// TODO rm
/// Simplifies notation in 'try_operation'. LHS and RHS should be a float on one side and a vector/matrix on the other side.
// fn _op_mv_float<T, U, V>(lhs: T, rhs: U, op: &BinaryOperation) -> Result<V, String>
// where T: std::Mul<U, Output=V> + std::Div<U, Output=V> + std::Rem<U, Output=V> + Quo<U, Output=V> + fmt::Display, U: fmt::Display {
//     match op {
//         BinaryOperation::Mul => Ok(lhs * rhs),
//         BinaryOperation::Div => Ok(lhs / rhs),
//         BinaryOperation::Rem => Ok(lhs % rhs),
//         BinaryOperation::Quo => Ok(Quo::quo(lhs, rhs)),
//         // All other operations are not possible (again, I write them out explicitely to be forced to review this snippet if I add new operations)
//         BinaryOperation::Add | BinaryOperation::Sub | BinaryOperation::Pow(_) | BinaryOperation::And | BinaryOperation::Or | BinaryOperation::Comp(..)
//             => Err(format!("Operation '{}' invalid for operands {} and {}.", op, lhs, rhs))
//     }
// }
// fn _op_mv_scalar<T, U, V>(lhs: T, rhs: U, op: &BinaryOperation) -> Result<V, String>
// where T: std::Mul<U, Output=V> + std::Div<U, Output=V> + fmt::Display, U: fmt::Display {
//     match op {
//         BinaryOperation::Mul => Ok(lhs * rhs),
//         BinaryOperation::Div => Ok(lhs / rhs),
//         // All other operations are not possible (again, I write them out explicitely to be forced to review this snippet if I add new operations)
//         BinaryOperation::Add | BinaryOperation::Sub | BinaryOperation::Pow(_) | BinaryOperation::Rem | BinaryOperation::Quo | BinaryOperation::And | BinaryOperation::Or | BinaryOperation::Comp(..)
//             => Err(format!("Operation '{}' invalid for operands {} and {}.", op, lhs, rhs))
//     }
// }

/// Returns 1 if the comparison succeeds, 0 otherwise
fn compare_reals(x: f64, y: f64, comp: Comparison) -> bool {
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
fn compare_complex<T: Float>(x: Complex<T>, y: Complex<T>, comp: Comparison) -> Result<bool, String> {
    match comp {
        Comparison::Eq => Ok(approx_eq(x.real, y.real) && approx_eq(x.imag, y.imag)),
        Comparison::Neq => Ok(!approx_eq(x.real, y.real) || !approx_eq(x.imag, y.imag)),
        Comparison::Gt | Comparison::Lt | Comparison::Ge | Comparison::Le => Err(format!("Comparison {} invalid for complex numbers.", comp))
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
            ok!(Object::LiteralExpression(expr_binop_from_enum!(lhs.to_expression(), op.clone(), rhs.to_expression())))
        }

        // Between real and real
        (Object::Real(x), Object::Real(y), BinaryOperation::Add) => ok!(Object::Real(x + y)),
        (Object::Real(x), Object::Real(y), BinaryOperation::Sub) => ok!(Object::Real(x - y)),
        (Object::Real(x), Object::Real(y), BinaryOperation::Mul) => ok!(Object::Real(x * y)),
        (Object::Real(x), Object::Real(y), BinaryOperation::Div) => ok!(Object::Real(x / y)),
        (Object::Real(x), Object::Real(y), BinaryOperation::Rem) => ok!(Object::Real(x.rem(*y))),
        (Object::Real(x), Object::Real(y), BinaryOperation::Quo) => ok!(Object::Real(x.quo(*y))),
        (Object::Real(x), Object::Real(y), BinaryOperation::Pow(_)) => ok!(Object::Real(x.pow(*y))),
        (Object::Real(x), Object::Real(y), BinaryOperation::Comp(c, _)) => ok!(Object::from_bool(compare_reals(*x, *y, *c))),
        (Object::Real(x), Object::Real(y), BinaryOperation::And) => ok!(Object::Real(if *x != 0.0 && *y != 0.0 {1.0} else {0.0})),
        (Object::Real(x), Object::Real(y), BinaryOperation::Or) => ok!(Object::Real(if *x != 0.0 || *y != 0.0 {1.0} else {0.0})),

        // Between complex and real
        (Object::Real(x), Object::Complex(z), BinaryOperation::Add)
        | (Object::Complex(z), Object::Real(x), BinaryOperation::Add) => ok!(Object::Complex(z.add(*x))),
        (Object::Real(x), Object::Complex(z), BinaryOperation::Sub) => ok!(Object::Complex(x.sub(*z))),
        (Object::Real(x), Object::Complex(z), BinaryOperation::Mul)
        | (Object::Complex(z), Object::Real(x), BinaryOperation::Mul) => ok!(Object::Complex(z.mul(*x))),
        (Object::Real(x), Object::Complex(z), BinaryOperation::Div) => ok!(Object::Complex(x.div(*z))),
        (Object::Real(x), Object::Complex(z), BinaryOperation::Pow(_)) => ok!(Object::Complex(x.pow(*z))),
        (Object::Real(x), Object::Complex(z), BinaryOperation::Comp(c, _)) => ok!(Object::from_bool(compare_complex(Complex { real: *x, imag: 0.0 }, *z, *c)?)),
        (Object::Real(x), Object::Complex(z), BinaryOperation::And)
        | (Object::Complex(z), Object::Real(x), BinaryOperation::And) => ok!(Object::Real(if *x != 0.0 && (z.real != 0.0 || z.imag != 0.0) {1.0} else {0.0})),
        (Object::Real(x), Object::Complex(z), BinaryOperation::Or)
        | (Object::Complex(z), Object::Real(x), BinaryOperation::Or) => ok!(Object::Real(if *x != 0.0 || z.real != 0.0 || z.imag != 0.0 {1.0} else {0.0})),
        (Object::Complex(z), Object::Real(x), BinaryOperation::Sub) => ok!(Object::Complex(z.sub(*x))),
        (Object::Complex(z), Object::Real(x), BinaryOperation::Div) => ok!(Object::Complex(z.div(*x))),
        (Object::Complex(z), Object::Real(x), BinaryOperation::Rem) => ok!(Object::Complex(z.rem(*x))),
        (Object::Complex(z), Object::Real(x), BinaryOperation::Quo) => ok!(Object::Complex(z.quo(*x))),
        (Object::Complex(z), Object::Real(x), BinaryOperation::Pow(_)) => ok!(Object::Complex(z.pow(*x))),
        (Object::Complex(z), Object::Real(x), BinaryOperation::Comp(c, _)) => ok!(Object::from_bool(compare_complex(*z, Complex { real: *x, imag: 0.0 }, *c)?)),
        
        // Between complex and complex
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Add) => ok!(Object::Complex(z.add(*w))),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Sub) => ok!(Object::Complex(z.sub(*w))),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Mul) => ok!(Object::Complex(z.mul(*w))),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Div) => ok!(Object::Complex(z.div(*w))),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Pow(_)) => ok!(Object::Complex(z.pow(*w))),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Comp(c, _)) => ok!(Object::from_bool(compare_complex(*z, *w, *c)?)),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::And) => ok!(Object::Real(
            if (w.real != 0.0 || w.imag != 0.0) && (z.real != 0.0 || z.imag != 0.0) {1.0} else {0.0}
        )),
        (Object::Complex(z), Object::Complex(w), BinaryOperation::Or) => ok!(Object::Real(
            if w.real != 0.0 || w.imag != 0.0 || z.real != 0.0 || z.imag != 0.0 {1.0} else {0.0}
        )),

        // Between real and vector
        (Object::Real(x), Object::Vector(y), BinaryOperation::Mul) => ok!(Object::Vector(map_mv!((y, Vector) -> VectorType {v => x.mul(v)}))),
        (Object::Real(x), Object::Vector(y), BinaryOperation::Div) => ok!(Object::Vector(map_mv!((y, Vector) -> VectorType {v => x.div(v)}))),
        (Object::Real(x), Object::Vector(y), BinaryOperation::Rem) => ok!(Object::Vector(map_mv!((y, Vector) -> VectorType {v => x.rem(v)}))),
        (Object::Real(x), Object::Vector(y), BinaryOperation::Quo) => ok!(Object::Vector(map_mv!((y, Vector) -> VectorType {v => x.quo(v)}))),
        (Object::Vector(y), Object::Real(x), BinaryOperation::Mul) => ok!(Object::Vector(map_mv!((y, Vector) -> VectorType {v => v.mul(*x)}))),
        (Object::Vector(y), Object::Real(x), BinaryOperation::Div) => ok!(Object::Vector(map_mv!((y, Vector) -> VectorType {v => v.div(*x)}))),
        (Object::Vector(y), Object::Real(x), BinaryOperation::Rem) => ok!(Object::Vector(map_mv!((y, Vector) -> VectorType {v => v.rem(*x)}))),
        (Object::Vector(y), Object::Real(x), BinaryOperation::Quo) => ok!(Object::Vector(map_mv!((y, Vector) -> VectorType {v => v.quo(*x)}))),
        // Between complex and vector
        (Object::Complex(x), Object::Vector(y), BinaryOperation::Mul) => ok!(Object::Vector(VectorType::Complex(dispatch_vector!(y, v => x.mul(v))))),
        (Object::Complex(x), Object::Vector(y), BinaryOperation::Div) => ok!(Object::Vector(VectorType::Complex(dispatch_vector!(y, v => x.div(v))))),
        (Object::Complex(x), Object::Vector(y), BinaryOperation::Rem) => ok!(Object::Vector(VectorType::Complex(dispatch_vector!(y, v => x.rem(v))))),
        (Object::Complex(x), Object::Vector(y), BinaryOperation::Quo) => ok!(Object::Vector(VectorType::Complex(dispatch_vector!(y, v => x.quo(v))))),
        (Object::Vector(y), Object::Complex(x), BinaryOperation::Mul) => ok!(Object::Vector(VectorType::Complex(dispatch_vector!(y, v => v.mul(*x))))),
        (Object::Vector(y), Object::Complex(x), BinaryOperation::Div) => ok!(Object::Vector(VectorType::Complex(dispatch_vector!(y, v => v.div(*x))))),
        (Object::Vector(y), Object::Complex(x), BinaryOperation::Rem) => ok!(Object::Vector(VectorType::Complex(dispatch_vector!(y, v => v.rem(*x))))),
        (Object::Vector(y), Object::Complex(x), BinaryOperation::Quo) => ok!(Object::Vector(VectorType::Complex(dispatch_vector!(y, v => v.quo(*x))))),

        // Between real and matrix
        (Object::Real(x), Object::Matrix(y), BinaryOperation::Mul) => ok!(Object::Matrix(map_mv!((y, Matrix) -> MatrixType {m => x.mul(m)}))),
        (Object::Real(x), Object::Matrix(y), BinaryOperation::Div) => try_map_mv!((y, Matrix) -> MatrixType {
            m => x.div(m).ok_or("Matrix is not invertible.".to_string())
        }).map(|res| Status::ok(Object::Matrix(res))),
        (Object::Real(x), Object::Matrix(y), BinaryOperation::Rem) => ok!(Object::Matrix(map_mv!((y, Matrix) -> MatrixType {m => x.rem(m)}))),
        (Object::Real(x), Object::Matrix(y), BinaryOperation::Quo) => ok!(Object::Matrix(map_mv!((y, Matrix) -> MatrixType {m => x.quo(m)}))),
        (Object::Matrix(y), Object::Real(x), BinaryOperation::Mul) => ok!(Object::Matrix(map_mv!((y, Matrix) -> MatrixType {m => m.mul(*x)}))),
        (Object::Matrix(y), Object::Real(x), BinaryOperation::Div) => ok!(Object::Matrix(map_mv!((y, Matrix) -> MatrixType {m => m.div(*x)}))),
        (Object::Matrix(y), Object::Real(x), BinaryOperation::Rem) => ok!(Object::Matrix(map_mv!((y, Matrix) -> MatrixType {m => m.rem(*x)}))),
        (Object::Matrix(y), Object::Real(x), BinaryOperation::Quo) => ok!(Object::Matrix(map_mv!((y, Matrix) -> MatrixType {m => m.quo(*x)}))),
        (Object::Matrix(x), Object::Real(y), BinaryOperation::Pow(_))
        if dispatch_matrix!(x, m => m.m() == m.n()) && let Some(exponent) = expect_int::<i64>(*y) => ok!(Object::Matrix(
            map_mv!((x, Matrix) -> MatrixType {
                m => {
                    if exponent >= 0 {
                        m.pow(exponent as u64).unwrap() // Safe because if-guard protects us from non-square matrices
                    } else {
                        let inv = m.inv().ok_or(format!("Matrix is not invertible: {:?}", m))?;
                        inv.pow((-exponent) as u64).unwrap()
                    }
                }
            })
        )),
        // Between complex and matrix
        (Object::Complex(x), Object::Matrix(y), BinaryOperation::Mul) => ok!(Object::Matrix(MatrixType::Complex(dispatch_matrix!(y, v => x.mul(v))))),
        (Object::Complex(x), Object::Matrix(y), BinaryOperation::Div) => dispatch_matrix!(
            y, m => x.div(m).ok_or("Matrix is not invertible.".to_string())
        ).map(|res| Status::ok(Object::Matrix(MatrixType::Complex(res)))),
        (Object::Complex(x), Object::Matrix(y), BinaryOperation::Rem) => ok!(Object::Matrix(MatrixType::Complex(dispatch_matrix!(y, v => x.rem(v))))),
        (Object::Complex(x), Object::Matrix(y), BinaryOperation::Quo) => ok!(Object::Matrix(MatrixType::Complex(dispatch_matrix!(y, v => x.quo(v))))),
        (Object::Matrix(y), Object::Complex(x), BinaryOperation::Mul) => ok!(Object::Matrix(MatrixType::Complex(dispatch_matrix!(y, v => v.mul(*x))))),
        (Object::Matrix(y), Object::Complex(x), BinaryOperation::Div) => ok!(Object::Matrix(MatrixType::Complex(dispatch_matrix!(y, v => v.div(*x))))),
        (Object::Matrix(y), Object::Complex(x), BinaryOperation::Rem) => ok!(Object::Matrix(MatrixType::Complex(dispatch_matrix!(y, v => v.rem(*x))))),
        (Object::Matrix(y), Object::Complex(x), BinaryOperation::Quo) => ok!(Object::Matrix(MatrixType::Complex(dispatch_matrix!(y, v => v.quo(*x))))),
        
        // Between vector and vector (add/sub/mul/compare)
        (Object::Vector(vt), Object::Vector(wt), BinaryOperation::Add)
        if let Some(res) = dispatch_vector!(vt, v => dispatch_vector!(wt, w => v.add(w).map(|r| r.wrap_in_type())))
        => ok!(Object::Vector(res)),
        (Object::Vector(vt), Object::Vector(wt), BinaryOperation::Sub)
        if let Some(res) = dispatch_vector!(vt, v => dispatch_vector!(wt, w => v.sub(w).map(|r| r.wrap_in_type())))
        => ok!(Object::Vector(res)),
        (Object::Vector(vt), Object::Vector(wt), BinaryOperation::Mul)
        if let Some(res) = try_map_two_mv!((vt, Vector; wt, Vector) -> Object {(v, w) => v.mul(w)})
        => ok!(res),
        // Comparisons between vectors
        (Object::Vector(VectorType::Real(v)), Object::Vector(VectorType::Real(w)), BinaryOperation::Comp(c, _))
        if v.len() == w.len() => compare_componentwise!(v, w, c, |(a, b)| Ok::<bool, String>(compare_reals(*a, *b, *c))),
        (Object::Vector(VectorType::Real(v)), Object::Vector(VectorType::Complex(w)), BinaryOperation::Comp(c, _))
        if v.len() == w.len() => compare_componentwise!(v, w, c, |(a, b)| compare_complex(a.to_complex(), *b, *c)),
        (Object::Vector(VectorType::Complex(v)), Object::Vector(VectorType::Real(w)), BinaryOperation::Comp(c, _))
        if v.len() == w.len() => compare_componentwise!(v, w, c, |(a, b)| compare_complex(*a, b.to_complex(), *c)),
        (Object::Vector(VectorType::Complex(v)), Object::Vector(VectorType::Complex(w)), BinaryOperation::Comp(c, _))
        if v.len() == w.len() => compare_componentwise!(v, w, c, |(a, b)| compare_complex(*a, *b, *c)),

        // Between matrix and matrix (add/sub/mul/compare)
        (Object::Matrix(vt), Object::Matrix(wt), BinaryOperation::Add)
        if let Some(res) = dispatch_matrix!(vt, v => dispatch_matrix!(wt, w => v.add(w).map(|r| r.wrap_in_type())))
        => ok!(Object::Matrix(res)),
        (Object::Matrix(vt), Object::Matrix(wt), BinaryOperation::Sub)
        if let Some(res) = dispatch_matrix!(vt, v => dispatch_matrix!(wt, w => v.sub(w).map(|r| r.wrap_in_type())))
        => ok!(Object::Matrix(res)),
        (Object::Matrix(vt), Object::Matrix(wt), BinaryOperation::Mul)
        if let Some(res) = try_map_two_mv!((vt, Matrix; wt, Matrix) -> MatrixType {(v, w) => v.mul(w)})
        => ok!(Object::Matrix(res)),
        // Comparisons between matrices
        (Object::Matrix(MatrixType::Real(v)), Object::Matrix(MatrixType::Real(w)), BinaryOperation::Comp(c, _))
        if v.m() == w.m() && v.n() == w.n() => compare_componentwise!(v, w, c, |(a, b)| Ok::<bool, String>(compare_reals(*a, *b, *c))),
        (Object::Matrix(MatrixType::Real(v)), Object::Matrix(MatrixType::Complex(w)), BinaryOperation::Comp(c, _))
        if v.m() == w.m() && v.n() == w.n() => compare_componentwise!(v, w, c, |(a, b)| compare_complex(a.to_complex(), *b, *c)),
        (Object::Matrix(MatrixType::Complex(v)), Object::Matrix(MatrixType::Real(w)), BinaryOperation::Comp(c, _))
        if v.m() == w.m() && v.n() == w.n() => compare_componentwise!(v, w, c, |(a, b)| compare_complex(*a, b.to_complex(), *c)),
        (Object::Matrix(MatrixType::Complex(v)), Object::Matrix(MatrixType::Complex(w)), BinaryOperation::Comp(c, _))
        if v.m() == w.m() && v.n() == w.n() => compare_componentwise!(v, w, c, |(a, b)| compare_complex(*a, *b, *c)),

        // Between vectors and matrices
        (Object::Vector(xt), Object::Matrix(yt), BinaryOperation::Mul)
        if let Some(res) = try_map_two_mv!((xt, Vector; yt, Matrix) -> MatrixType {(x, y) => x.mul(y)})
        => ok!(Object::Matrix(res)),
        (Object::Matrix(xt), Object::Vector(yt), BinaryOperation::Mul)
        if let Some(res) = try_map_two_mv!((xt, Matrix; yt, Vector) -> VectorType {(x, y) => x.mul(y)})
        => ok!(Object::Vector(res)),

        _ => Err(format!("Operation '{}' invalid for operands {} and {}.", op, lhs, rhs))
    }
}