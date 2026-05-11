use std::ops;
use std::fmt;

use crate::math::matrices_and_vectors::{Matrix, Vector};
use crate::math::expressions::Expression;
use crate::math::operations::*;
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
            Object::Vector(x) => write!(f, "{:?}", x),
            Object::Matrix(x) => write!(f, "{:?}", x),
            Object::LiteralExpression(x) => write!(f, "{:?}", x),
            //Object::Boolean(x) => write!(f, "{}", if *x { "True" } else { "False" })
        }
    }
}
impl Object {
    /// Formats an object to a string that may stretch over multiple lines.
    /// The lines will be returned as a vector of strings, not as a single string containing newline chars.
    pub fn to_multline(&self) -> Vec<String> {
        match self {
            Object::Success => vec!["Assignment successful".to_string()],
            Object::Undefined => vec!["Undefined".to_string()],
            Object::Float(x) => vec![x.to_string()],
            Object::Vector(x) => vec![format!("Vec<{}>: {:?}", x.len(), &x.values)],
            Object::Matrix(x) => {
                // First, we go through all element to know how much space each column needs.
                let column_lengths: Vec<usize> = (0..x.n).map(
                    |j| (0..x.m).map(
                        |i| x
                        .get(i, j)
                        .to_string()
                        .len()
                    ).max().unwrap_or(0)
                ).collect();
                let row_length = column_lengths.iter().sum::<usize>() + 2*x.n; // Between two columns, add 2 spaces. Before the first columns and after the last one, only 1 space.
                let mut lines = vec![format!("╭{}╮", (0..row_length).map(|_| ' ').collect::<String>())];
                for i in 0..x.m {
                    lines.push(format!("│ {}│", (0..x.n).map(
                        |j| format!("{:^2$} {}", x.get(i, j), if j == x.n-1 {""} else {" "}, column_lengths[j])
                    ).collect::<String>()));
                }
                lines.push(format!("╰{}╯", (0..row_length).map(|_| ' ').collect::<String>()));
                lines
            }
            Object::LiteralExpression(x) => x.to_multline()
            //Object::Boolean(x) => vec![(if *x { "True" } else { "False" }).to_string()]
        }
    }
}
/// This operation is only derived to simplify typing in `directional_derivative`.
impl<'a> ops::Mul<&'a Object> for f64 {
    type Output = Object;
    fn mul(self, rhs: &'a Object) -> Self::Output {
        match rhs {
            Object::Success => Object::Success,
            Object::Undefined => Object::Undefined,
            Object::Float(x) => Object::Float(self * x),
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

impl ops::Neg for &Object {
    type Output = Object;
    fn neg(self) -> Self::Output {
        match self {
            Object::Success => Object::Success,
            Object::Undefined => Object::Undefined,
            Object::Float(x) => Object::Float(-x),
            Object::Vector(x) => Object::Vector(-x),
            Object::Matrix(x) => Object::Matrix(-x),
            Object::LiteralExpression(expr) => Object::LiteralExpression(Expression::UnaryOperation(
                UnaryOperation::Neg,
                Box::new(expr.clone())
            )),
        }
    }
}

pub type DirectFunction = Box<dyn Fn(&Vec<Object>) -> Result<Object, String>>;

/// Different representations for a function
pub enum FunctionRepr {
    /// 1. The list of identifiers of the arguments (in order to parse the literal expression correctly).
    ///    These will be prefixed with three underscores to avoid confusion with normal constants. Note that
    ///    the user is not allowed to define a variable whose name starts with three underscores. From our
    ///    perspective, this prefix is good because it allows us to simply add a few constants temporarily
    ///    when evaluating a function instead of having to remember which variables to revert or even
    ///    copying the HashMap `constants` entirely.
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
        BinaryOperation::Add | BinaryOperation::Sub | BinaryOperation::Quo | BinaryOperation::Pow | BinaryOperation::Comp(..)
            => Err(format!("Operation {} invalid for operands {:?} and {:?}.", op, lhs, rhs))
    }
}

/// Returns 1 if the comparison succeeds, 0 otherwise
fn compare(x: &f64, y: &f64, comp: &Comparison) -> bool {
    match comp {
        Comparison::Eq => approx_eq(x, y),
        Comparison::Gt => x > y,
        Comparison::Lt => x < y,
        Comparison::Ge => x >= y || approx_eq(x, y),
        Comparison::Le => x < y || approx_eq(x, y)
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
        Object::Success | Object::Undefined => err(), // You can't do any operation with 'Success'
        Object::Float(x) => {
            match rhs {
                Object::Float(y) => Ok(Object::Float(
                    match op {
                        BinaryOperation::Add => x+y,
                        BinaryOperation::Sub => x-y,
                        BinaryOperation::Mul => x*y,
                        BinaryOperation::Div => x/y,
                        BinaryOperation::Rem => x%y,
                        // The following result should in fact already be an integer, the `.round()` only converts it to int while accounting for small errors.
                        BinaryOperation::Quo => ((x - (x % y)) / y).round(),
                        BinaryOperation::Pow => x.powf(*y),
                        BinaryOperation::Comp(comp, _) => compare(x, y, comp) as i8 as f64
                    }
                )),
                Object::Vector(y) => {
                    Ok(Object::Vector(_op_mv_float(*x, y, op)?))
                }
                Object::Matrix(y) => {
                    Ok(Object::Matrix(_op_mv_float(*x, y, op)?))
                }
                Object::Success | Object::Undefined | Object::LiteralExpression(_) => err()
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
                    if let BinaryOperation::Pow = op {
                        // Matrix exponentiation is only accepted when the exponent is an integer (a.k.a. approximately equal to an integer)
                        let exponent = y.round();
                        if approx_eq(&exponent, y) && *y >= 0.0 {
                            // I didn't use `fold` here because if one step of the operation fails, `None` should be returned (instead of panicking).
                            (0..exponent as i64)
                                .try_fold(Matrix::identity(x.m), |acc, _| (&acc * x).ok_or_else(err_msg))
                                .map(Object::Matrix)
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