use std::fmt;
use crate::math::operations::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    None, // Used as the LHS of unary operations
    Identifier(String),
    Number(f64),
    /// This also doubles as a container for a function's arguments when the function isn't defined yet (cf. `Assignment` block in `eval`).
    Vector(Vec<Expression>), // As for functions
    /// Dimensions of the matrix and list of entries in flattened version.
    Matrix(usize, usize, Vec<Expression>), // Same
    UnaryOperation(UnaryOperation, Box<Expression>),
    /// Comparisons are interpreted as binary operations too.
    BinaryOperation(Box<Expression>, BinaryOperation, Box<Expression>),
    /// Respectively: function's name and list of arguments passed.
    Function(String, Vec<Expression>),
    /// Format: LHS := RHS
    Assignment(Box<Expression>, Box<Expression>),
    /// Compute the partial derivative of the given expression w.r.t. the given identifier. The direction to differentiate in is set to 1.0.
    PartialDerivative(String, Box<Expression>),
    /// Compute the directional derivative of `SecondArg` at point `ThirdArg` in direction `FourthArg` where the variables w.r.t. which we differentiate are `first_args`.
    DirectionalDerivative(Vec<String>, Box<Expression>, Vec<Expression>, Vec<Expression>)
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::None => write!(f, "None"),
            Expression::Identifier(s) => write!(f, "{}", s),
            Expression::Number(x) => write!(f, "{}", x),
            Expression::Vector(x) => write!(f, "[{}]", x.iter().map(|y| format!("{}", y)).collect::<Vec<String>>().join(", ")),
            Expression::Matrix(m, n, x) => write!(f, "[{}]", (0..*m).map(|i| (0..*n).map(|j| format!("{}", x[i*n+j])).collect::<Vec<String>>().join(", ")).collect::<Vec<String>>().join("; ")),
            Expression::UnaryOperation(op, r) => {
                match op {
                    UnaryOperation::Neg => write!(f, "(-({}))", r),
                    UnaryOperation::Abs => write!(f, "|{}|", r),
                }
            },
            Expression::BinaryOperation(l, op, r) => write!(f, "({} {} {})", l, op, r),
            Expression::Function(name, args)
                => write!(f, "{}({})", name, args.iter().map(|x| format!("{:?}", x)).collect::<Vec<String>>().join(", ")),
            Expression::Assignment(lhs, rhs) => write!(f, "{} := {}", lhs, rhs),
            Expression::PartialDerivative(wrt, expr) => write!(f, "d/d{} ({})", wrt, expr),
            Expression::DirectionalDerivative(vars, expr, point, direction) => write!(f, "D_{{{}}} ({})({:?})[{:?}]", vars.join(", "), expr, point, direction),
        }
    }
}
impl Expression {
    /// Formats an object to a string that may stretch over multiple lines.
    /// The lines will be returned as a vector of strings, not as a single string containing newline chars.
    pub fn to_multline(&self) -> Vec<String> {
        match self {
            Expression::None => vec!["None".to_string()],
            Expression::Identifier(s) => vec![format!("{}", s)],
            Expression::Number(x) => vec![format!("{}", x)],
            Expression::Vector(x) => vec![format!("[{}]", x.iter().map(|y| format!("{}", y)).collect::<Vec<String>>().join(", "))],
            Expression::Matrix(m, n, x) => {
                let values = x.iter().map(|b| format!("{}", b)).collect::<Vec<String>>();
                let column_lengths: Vec<usize> = (0..*n).map(
                    |j| (0..*m).map(
                        |i| values[i*n+j]
                        .len()
                    ).max().unwrap_or(0)
                ).collect();
                let row_length = column_lengths.iter().sum::<usize>() + 2*n; // Between two columns, add 2 spaces. Before the first columns and after the last one, only 1 space.
                let mut lines = vec![format!("╭{}╮", (0..row_length).map(|_| ' ').collect::<String>())];
                for i in 0..*m {
                    lines.push(format!("│ {}│", (0..*n).map(
                        |j| format!("{:^2$} {}", values[i*n+j], if j == n-1 {""} else {" "}, column_lengths[j])
                    ).collect::<String>()));
                }
                lines.push(format!("╰{}╯", (0..row_length).map(|_| ' ').collect::<String>()));
                lines
            },
            Expression::UnaryOperation(op, r) => {
                match op {
                    UnaryOperation::Neg => vec![format!("(-({}))", r)],
                    UnaryOperation::Abs => vec![format!("|{}|", r)],
                }
            },
            Expression::BinaryOperation(l, op, r) => vec![format!("({} {} {})", l, op, r)],
            Expression::Function(name, args)
                => vec![format!("{}({})", name, args.iter().map(|x| format!("{}", x)).collect::<Vec<String>>().join(", "))],
            Expression::Assignment(lhs, rhs) => vec![format!("{} := {}", lhs, rhs)],
            Expression::PartialDerivative(wrt, expr) => vec![format!("d/d{} ({})", wrt, expr)],
            Expression::DirectionalDerivative(vars, expr, point, direction) => vec![format!("D_{{{}}} ({})({:?})[{:?}]", vars.join(", "), expr, point, direction)],
        }
    }
}

/// Allows to simplify literal expressions.
/// 
/// If `lhs` is zero, returns `rhs`. If `rhs` is zero, returns `lhs`. Otherwise, returns `lhs + rhs`.
pub fn simplify_add(lhs: Expression, rhs: Expression) -> Expression {
    match (lhs, rhs) {
        (Expression::Number(0.0), other) | (other, Expression::Number(0.0)) => other,
        (lhs, rhs) => Expression::BinaryOperation(Box::new(lhs), BinaryOperation::Add, Box::new(rhs))
    }
}
/// Allows to simplify literal expressions.
/// 
/// If `lhs` and `rhs` are both numbers, subtract and return the wrapped result.
/// If `lhs` is zero, returns `-rhs`. If `rhs` is zero, returns `lhs`. Otherwise, returns `lhs - rhs`.
pub fn simplify_sub(lhs: Expression, rhs: Expression) -> Expression {
    match (lhs, rhs) {
        (Expression::Number(x), Expression::Number(y)) => Expression::Number(x-y),
        (Expression::Number(0.0), rhs) => Expression::UnaryOperation(UnaryOperation::Neg, Box::new(rhs)),
        (lhs, Expression::Number(0.0)) => lhs,
        (lhs, rhs) => Expression::BinaryOperation(Box::new(lhs), BinaryOperation::Sub, Box::new(rhs))
    }
}
/// Allows to simplify literal expressions.
/// 
/// If one term is `0`, returns `0`. If one term is `1`, returns the other one. Otherwise, returns `lhs * rhs`.
pub fn simplify_mul(lhs: Expression, rhs: Expression) -> Expression {
    match (lhs, rhs) {
        (Expression::Number(0.0), _) | (_, Expression::Number(0.0)) => Expression::Number(0.0),
        (Expression::Number(1.0), x) | (x, Expression::Number(1.0)) => x,
        (lhs, rhs) => Expression::BinaryOperation(Box::new(lhs), BinaryOperation::Mul, Box::new(rhs))
    }
}
/// Allows to simplify literal expressions.
/// 
/// If `rhs` is `1`, returns `lhs`. Otherwise, returns `lhs / rhs`.
pub fn simplify_div(lhs: Expression, rhs: Expression) -> Expression {
    if let Expression::Number(1.0) = rhs {
        lhs
    }
    else {
        Expression::BinaryOperation(Box::new(lhs), BinaryOperation::Div, Box::new(rhs))
    }
}
/// Allows to simplify literal expressions.
/// 
/// If `rhs` is `1`, returns `lhs`. If `rhs` is `0` or `lhs` is `1`, returns `1`. Otherwise, returns `lhs ^ rhs`.
pub fn simplify_pow(lhs: Expression, rhs: Expression) -> Expression {
    if let Expression::Number(1.0) = rhs {
        lhs
    }
    else if let Expression::Number(0.0) = rhs {
        Expression::Number(1.0)
    }
    else if let Expression::Number(1.0) = lhs {
        lhs
    }
    else {
        Expression::BinaryOperation(Box::new(lhs), BinaryOperation::Pow, Box::new(rhs))
    }
}


impl Expression {
    /// Parses itself recursively and replaces every encountered `ident` by `by`. Ignores the LHS of assignment operators.
    pub fn replace_identifiers(&mut self, ident: &String, by: &Expression) {
        match self {
            Expression::Identifier(x) if x == ident => {
                *self = by.clone();
            }
            Expression::Vector(v) | Expression::Matrix(.., v) | Expression::Function(_, v)
                => {v.iter_mut().for_each(|x| x.replace_identifiers(ident, by));}
            Expression::UnaryOperation(_, x) | Expression::PartialDerivative(_, x) | Expression::Assignment(_, x) // Ignore LHS of assigment operator
                => x.replace_identifiers(ident, by),
            Expression::BinaryOperation(x, _, y)
                => {x.replace_identifiers(ident, by); y.replace_identifiers(ident, by);}
            Expression::DirectionalDerivative(_, x, point, direction) => {
                x.replace_identifiers(ident, by);
                point.iter_mut().for_each(|y| y.replace_identifiers(ident, by));
                direction.iter_mut().for_each(|y| y.replace_identifiers(ident, by));
            }
            _ => {}
        }
    }
}