use std::ops;
use std::fmt;

use crate::math::matrices_and_vectors::{Matrix, Vector};
use crate::math::expressions::Expression;
use crate::math::operations::*;
use crate::math::utils;
use crate::math::utils::approx_eq;

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
    Float(f64),
    Tuple(Vec<Object>),
    /// Vector/matrix operations are implemented for references to vectors/matrices anyway, so only
    /// using references to Vector/Matrix makes sense here.
    Vector(Vector),
    Matrix(Matrix),
    LiteralExpression(Expression)
}
impl fmt::Display for Object {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Object::Success => write!(f, "Object::Success"),
            Object::Undefined => write!(f, "Undefined"),
            Object::Float(x) => write!(f, "{}", x),
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
            Object::Float(x) => vec![x.to_string()],
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
            Object::Vector(v) => vec![format!("({})", &v.values.iter().map(|x| utils::format_trimmed(*x, 8)).collect::<Vec<String>>().join(", "))],
            Object::Matrix(x) => {
                // First, we go through all element to know how much space each column needs.
                let mut column_lengths = Vec::<usize>::with_capacity(x.n);
                let mut entries = Vec::<String>::with_capacity(x.n * x.m); // Notice this is the transposed version of the typical flattened vector
                for j in 0..x.n {
                    column_lengths.push((0..x.m).map(
                        |i| {
                            let s = utils::format_trimmed(x.get(i, j), 8);
                            let len = s.len();
                            entries.push(s);
                            len
                        }
                    ).max().unwrap_or(0))
                }
                // Cache locality isn't very important here since only so much can be displayed on a reasonable screen anyway
                let row_length = column_lengths.iter().sum::<usize>() + 2*x.n; // Between two columns, add 2 spaces. Before the first columns and after the last one, only 1 space.
                let mut lines = vec![format!("╭{}╮", (0..row_length).map(|_| ' ').collect::<String>())];
                for i in 0..x.m {
                    lines.push(format!("│ {}│", (0..x.n).map(
                        |j| format!("{:^2$} {}", entries[j*x.m + i], if j == x.n-1 {""} else {" "}, column_lengths[j])
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
            Object::Float(x) => Expression::Number(*x),
            Object::Tuple(v) => Expression::Tuple(v.iter().map(|o| o.to_expression()).collect()),
            Object::Vector(v) => Expression::Vector(v.values.iter().map(|entry| Expression::Number(*entry)).collect()),
            Object::Matrix(x) => Expression::Matrix(
                x.m, x.n,
                x.iter_values().map(|entry| Expression::Number(*entry)).collect()
            ),
            Object::LiteralExpression(e) => e.clone()
        }
    }

    pub fn expect_float(self) -> Result<f64, String> {
        match self {
            Object::Float(x) => Ok(x),
            other => Err(format!("Expected float, got {other}."))
        }
    }

    pub fn expect_int(self) -> Result<f64, String> {
        let f = self.expect_float()?;
        let i = f.round();
        if approx_eq(f, i) {
            Ok(i)
        } else {
            Err(format!("Expected number close to integer; got {f}."))
        }
    }
}
impl<'a> ops::Mul<&'a Object> for f64 {
    type Output = Object;
    fn mul(self, rhs: &'a Object) -> Self::Output {
        match rhs {
            Object::Success => Object::Success,
            Object::Undefined => Object::Undefined,
            Object::Float(x) => Object::Float(self * x),
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
impl ops::Add<Object> for Object {
    type Output = Result<Object, String>;
    fn add(self, rhs: Object) -> Self::Output {
        try_operation(&self, &rhs, &BinaryOperation::Add)
    }
}
impl ops::Neg for &Object {
    type Output = Result<Object, String>;
    fn neg(self) -> Self::Output {
        match self {
            Object::Success => Ok(Object::Success),
            Object::Undefined => Err("Operation 'Neg' not valid for undefined operand.".to_string()),
            Object::Float(x) => Ok(Object::Float(-x)),
            Object::Tuple(x) => Ok(Object::Tuple(x.iter().map(|o| -o).collect::<Result<Vec<_>, _>>()?)),
            Object::Vector(x) => Ok(Object::Vector(-x)),
            Object::Matrix(x) => Ok(Object::Matrix(-x)),
            Object::LiteralExpression(expr) => Ok(Object::LiteralExpression(crate::expr_neg!(expr.clone()))),
        }
    }
}
impl ops::Neg for Object {
    type Output = Result<Object, String>;
    fn neg(self) -> Self::Output {
        (&self).neg()
    }
}
impl ops::Not for &Object {
    type Output = Result<Object, String>;
    fn not(self) -> Self::Output {
        match self {
            Object::Success => Ok(Object::Success),
            Object::Undefined => Err("Operation 'Not' not valid for undefined operand.".to_string()),
            Object::Float(x) => Ok(Object::Float(if *x == 0.0 {1.0} else {0.0})),
            Object::Tuple(v) => Ok(Object::Tuple(v.iter().map(|o| !o).collect::<Result<Vec<_>, _>>()?)),
            Object::Vector(v) => Ok(Object::Vector(v.transform(|x| if x == 0.0 {1.0} else {0.0}))),
            Object::Matrix(m) => Ok(Object::Matrix(m.transform(|x| if x == 0.0 {1.0} else {0.0}))),
            Object::LiteralExpression(e) => Ok(Object::LiteralExpression(Expression::UnaryOperation(UnaryOperation::Not, Box::new(e.clone())))),
        }
    }
}

pub type DirectFunction = Box<dyn Fn(&[Object]) -> Result<Object, String>>;

/// Different representations for a function
pub enum FunctionRepr {
    /// 1. The list of identifiers of the arguments (in order to parse the literal expression correctly).
    ///    These will be prefixed with `___tmp_` to avoid confusion with normal constants. Note that
    ///    the user is not allowed to define a variable whose name starts with three underscores.
    /// 2. E.g. `"5 * ___tmp_x + 2"` where `arguments` is `["___tmp_x"]`. The variable names here will already be prefixed.
    ByExpression(Vec<String>, Expression),
    Direct(DirectFunction)
}

impl fmt::Debug for FunctionRepr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FunctionRepr::ByExpression(argnames, expr) => write!(f, "({}) ↦ {}", argnames.join(", "), expr),
            FunctionRepr::Direct(_) => write!(f, "<Closure>")
        }
    }
}


/// Simplifies notation in 'try_operation'. LHS and RHS should be a float on one side and a vector/matrix on the other side.
fn _op_mv_float<T, U, V>(lhs: T, rhs: U, op: &BinaryOperation) -> Result<V, String>
where T: std::ops::Mul<U, Output=V> + std::ops::Div<U, Output=V> + std::ops::Rem<U, Output=V> + fmt::Debug, U: fmt::Debug {
    match op {
        BinaryOperation::Mul => Ok(lhs * rhs),
        BinaryOperation::Div => Ok(lhs / rhs),
        BinaryOperation::Rem => Ok(lhs % rhs),
        // All other operations are not possible (again, I write them out explicitely to be forced to review this snippet if I add new operations)
        BinaryOperation::Add | BinaryOperation::Sub | BinaryOperation::Quo | BinaryOperation::Pow(_) | BinaryOperation::And | BinaryOperation::Or | BinaryOperation::Comp(..)
            => Err(format!("Operation {} invalid for operands {:?} and {:?}.", op, lhs, rhs))
    }
}

/// Returns 1 if the comparison succeeds, 0 otherwise
fn compare(x: &f64, y: &f64, comp: &Comparison) -> bool {
    match comp {
        Comparison::Eq => approx_eq(*x, *y),
        Comparison::Neq => !approx_eq(*x, *y),
        Comparison::Gt => x > y,
        Comparison::Lt => x < y,
        Comparison::Ge => x >= y || approx_eq(*x, *y),
        Comparison::Le => x < y || approx_eq(*x, *y)
    }
}

/// Attempts to perform the given operation 'op' on the given operands 'lhs' and 'rhs'.
/// On success, returns 'Some(lhs op rhs)'. On failure, returns 'None'.
/// 
/// I don't really see any significantly better way of doing this than to compare types, since we need to put the output into an 'Object' too
/// and we must take care of possible dimension mismatches too.
/// I'd go as far as saying this is fine since there are (currently) only 4 different types.
pub fn try_operation(lhs: &Object, rhs: &Object, op: &BinaryOperation) -> Result<Object, String> {
    let err_msg = || format!("Operation {} invalid for operands {:?} and {:?}.", op, lhs, rhs); // Simplifies typing in the following match block
    let err = || Err(err_msg());
    match lhs {
        Object::Success | Object::Undefined | Object::Tuple(_) => err(), // You can't do any operation with 'Success'
        Object::Float(x) => {
            match rhs {
                Object::Float(y) => Ok(Object::Float(
                    match op {
                        BinaryOperation::Add => x+y,
                        BinaryOperation::Sub => x-y,
                        BinaryOperation::Mul => x*y,
                        BinaryOperation::Div => x/y,
                        BinaryOperation::Rem => x.rem_euclid(*y),
                        // The following result should in fact already be an integer, the `.round()` only converts it to int while accounting for small errors.
                        BinaryOperation::Quo => ((x - (x.rem_euclid(*y))) / y).round(),
                        BinaryOperation::Pow(_) => x.powf(*y),
                        BinaryOperation::Comp(comp, _) => compare(x, y, comp) as i8 as f64,
                        BinaryOperation::Or => if *x != 0.0 || *y != 0.0 {1.0} else {0.0},
                        BinaryOperation::And => if *x != 0.0 && *y != 0.0 {1.0} else {0.0},
                    }
                )),
                Object::Vector(y) => {
                    Ok(Object::Vector(_op_mv_float(*x, y, op)?))
                }
                Object::Matrix(y) => {
                    Ok(Object::Matrix(_op_mv_float(*x, y, op)?))
                }
                Object::Tuple(_) | Object::Success | Object::Undefined | Object::LiteralExpression(_) => err()
            }
        }
        Object::Vector(x) => {
            match rhs {
                Object::Float(y) => {
                    Ok(Object::Vector(_op_mv_float(x, *y, op)?))
                }
                Object::Vector(y) => {
                    match op { // Shorter, since Vector operations have different return types including Option<...>
                        BinaryOperation::Add => {
                            (x+y).map(Object::Vector).ok_or_else(err_msg)
                        }
                        BinaryOperation::Sub => {
                            (x-y).map(Object::Vector).ok_or_else(err_msg)
                        }
                        BinaryOperation::Mul => {
                            (x*y).map(Object::Float).ok_or_else(err_msg)
                        }
                        BinaryOperation::Comp(c, _) => {
                            let n = x.len();
                            if n == y.len() {
                                Ok(Object::Float((0..n).all(|i| compare(&x[i], &y[i], c)) as i8 as f64))
                            }
                            else {
                                err()
                            }
                        }
                        _ => err()
                    }
                }
                Object::Matrix(y) if *op == BinaryOperation::Mul => { // Only possible operation between matrix and vector
                    (x*y).map(Object::Vector).ok_or_else(err_msg)
                }
                _ => err()
            }
        }
        Object::Matrix(x) => {
            match rhs {
                Object::Float(y) => {
                    if let BinaryOperation::Pow(_) = op {
                        // Matrix exponentiation is only accepted when the exponent is an integer (a.k.a. approximately equal to an integer)
                        let exponent = y.round();
                        if x.m == x.n && approx_eq(exponent, *y) {
                            if exponent >= 0.0 {
                                Ok(Object::Matrix(x.pow(exponent as u64).ok_or(format!("Matrix must be quadratic to apply `Pow` (got size {}x{})", x.m, x.n))?))
                            } else {
                                let inv = x.inv().ok_or(format!("Matrix is not invertible: {:?}", x))?;
                                Ok(Object::Matrix(inv.pow((-exponent) as u64).unwrap())) // `unwrap` is safe since if `inv` exists, it is necessarily quadratic.
                            }
                        }
                        else {err()}
                    }
                    else {
                        Ok(Object::Matrix(_op_mv_float(x, *y, op)?))
                    }
                }
                Object::Vector(y) if *op == BinaryOperation::Mul => {
                    (x*y).map(Object::Vector).ok_or_else(err_msg)
                }
                Object::Matrix(y) => {
                    if let BinaryOperation::Comp(c, _) = op {
                        let m = x.m; let n = x.n;
                        if m == y.m && n == y.n {
                            Ok(Object::Float((0..m).all(
                                |i| (0..n).all(
                                    |j| compare(&x.get(i, j), &y.get(i, j), c)
                                )
                            ) as i8 as f64))
                        }
                        else {
                            err()
                        }
                    }
                    else {
                        match op {
                            BinaryOperation::Add => x+y,
                            BinaryOperation::Sub => x-y,
                            BinaryOperation::Mul => x*y,
                            _ => None
                        }
                            .map(Object::Matrix).ok_or_else(err_msg)
                    }
                }
                _ => err()
            }
        }
        Object::LiteralExpression(_) => err()
    }
}