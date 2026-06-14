use std::fmt;

use crate::math::{Expression, Object};

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

#[derive(Clone, Copy, PartialEq)]
pub enum Comparison { Eq, Gt, Ge, Lt, Le }
impl Comparison {
    pub fn as_str(&self) -> &str {
        match self {
            Comparison::Eq => "=",
            Comparison::Gt => ">",
            Comparison::Ge => ">=",
            Comparison::Lt => "<",
            Comparison::Le => "<=",
        }
    }
}
impl fmt::Display for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
impl fmt::Debug for Comparison {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Clone, PartialEq)]
pub enum BinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    Quo,
    Rem,
    Pow,
    And,
    Or,
    Comp(Comparison, Option<Box<Expression>>)
}
impl BinaryOperation {
    pub fn as_str(&self) -> &str {
        match self {
            BinaryOperation::Add => "+",
            BinaryOperation::Sub => "-",
            BinaryOperation::Mul => "*",
            BinaryOperation::Div => "/",
            BinaryOperation::Quo => "//",
            BinaryOperation::Rem => "%",
            BinaryOperation::Pow => "^",
            BinaryOperation::And => "&&",
            BinaryOperation::Or => "||",
            BinaryOperation::Comp(c, _) => c.as_str(),
        }
    }
    pub fn priority(&self) -> u8 {
        match self {
            BinaryOperation::Add => 5,
            BinaryOperation::Sub => 5,
            BinaryOperation::Mul => 6,
            BinaryOperation::Div => 6,
            BinaryOperation::Quo => 6,
            BinaryOperation::Rem => 6,
            BinaryOperation::Pow => 7,
            BinaryOperation::And => 2,
            BinaryOperation::Or => 1,
            BinaryOperation::Comp(..) => 4,
        }
    }
}
impl fmt::Display for BinaryOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
impl fmt::Debug for BinaryOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

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

/// Any operation for which an operator of the type `sum_{i=1}^n ...` is implemented.
#[derive(Clone, Debug, PartialEq)]
pub enum FoldedOperation {
    Sum,
    Product
}
impl fmt::Display for FoldedOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FoldedOperation::Sum => write!(f, "sum"),
            FoldedOperation::Product => write!(f, "prod"),
        }
    }
}

impl FoldedOperation {
    pub fn priority(&self) -> u8 {
        match self {
            FoldedOperation::Sum => BinaryOperation::Add.priority(),
            FoldedOperation::Product => BinaryOperation::Mul.priority(),
        }
    }

    pub fn underlying_binop(&self) -> BinaryOperation {
        match self {
            FoldedOperation::Sum => BinaryOperation::Add,
            FoldedOperation::Product => BinaryOperation::Mul
        }
    }

    pub fn valid_string(str: &str) -> bool {
        str == "sum" || (str.starts_with("sum") && str.chars().nth(3) == Some('_'))
        || str == "prod" || (str.starts_with("prod") && str.chars().nth(4) == Some('_')) 
    }

    pub fn from_string(str: &str) -> Option<FoldedOperation> {
        if str == "sum" || (str.starts_with("sum") && str.chars().nth(3) == Some('_')) {
            Some(FoldedOperation::Sum)
        } else if str == "prod" || (str.starts_with("prod") && str.chars().nth(4) == Some('_')) {
            Some(FoldedOperation::Product)
        } else {
            None
        }
    }

    /// Returns the value of an empty folded operation of type `self` (e.g. 0 for sums, 1 for products).
    pub fn if_empty(&self) -> Object {
        match self {
            FoldedOperation::Sum => Object::Float(0.0),
            FoldedOperation::Product => Object::Float(1.0)
        }
    }
}