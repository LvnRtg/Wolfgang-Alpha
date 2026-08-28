use std::fmt;

use crate::math::Expression;

#[derive(Clone, Debug, PartialEq)]
pub enum UnaryOperation {
    Neg,
    Not,
    Factorial,
    Abs,
    Norm(Option<Box<Expression>>),
}

impl fmt::Display for UnaryOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            UnaryOperation::Neg => write!(f, "-"),
            UnaryOperation::Not => write!(f, "!"),
            UnaryOperation::Factorial => write!(f, "!"),
            UnaryOperation::Abs => write!(f, "|_|"),
            UnaryOperation::Norm(opt) => write!(f, "||_||{}", format_optional_subscript(opt)),
        }
    }
}
impl UnaryOperation {
    /// Example: applied to `UnaryOperation::Neg` and some vector `v`,
    /// adds '-' at the beginning of `v[0]`.
    pub fn format_with_multline_expr(&self, expr: &mut [String]) {
        match self {
            UnaryOperation::Neg => expr[0].insert(0, '-'),
            UnaryOperation::Not => expr[0].insert(0, '!'),
            UnaryOperation::Factorial => expr.last_mut().unwrap().push('!'),
            UnaryOperation::Abs => {
                expr[0].insert(0, '|');
                expr.last_mut().unwrap().push('|');
            }
            UnaryOperation::Norm(opt) => {
                expr[0].insert_str(0, "||");
                expr.last_mut().unwrap().push_str(format!("||{}", format_optional_subscript(opt)).as_str());
            }
        }
    }
}

/// If `opt` is `Some(x)`, returns "_x" if x is an identifier or a number and "_{x}" otherwise.
/// If `opt` is `None`, returns an empty string.
pub fn format_optional_subscript(opt: &Option<Box<Expression>>) -> String {
    if let Some(e) = opt {
        match &**e {
            Expression::Number(x) => format!("_{x}"),
            Expression::Identifier(x) => format!("_{x}"),
            other => format!("_{{{other}}}"),
        }
    } else {String::new()}
}