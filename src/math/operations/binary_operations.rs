use std::fmt;

use crate::math::Expression;

#[derive(Clone, PartialEq)]
pub enum BinaryOperation {
    Add,
    Sub,
    Mul,
    Div,
    Quo,
    Rem,
    /// The bool is only needed during parsing and is completely disregarded afterwards.
    /// It signifies whether this power operation is supposed to be right-associative (true)
    /// or left-associative (false).
    Pow(bool),
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
            BinaryOperation::Pow(_) => "^",
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
            BinaryOperation::Pow(_) => 7,
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


#[derive(Clone, Copy, PartialEq)]
pub enum Comparison { Eq, Neq, Gt, Ge, Lt, Le }
impl Comparison {
    pub fn as_str(&self) -> &str {
        match self {
            Comparison::Eq => "=",
            Comparison::Neq => "!=",
            Comparison::Gt => ">",
            Comparison::Ge => ">=",
            Comparison::Lt => "<",
            Comparison::Le => "<=",
        }
    }
    /// Assume you want to evaluate `v comp w` where `u, v` are vectors/matrices of the same size.
    /// If `comp == Eq`, then this is equivalent to `all(v[i] comp w[i])`, but if `comp == Neq`,
    /// then it should be `any(v[i] comp w[i])` instead. This function returns `true` iff `all`
    /// should be used.
    pub fn check_all(&self) -> bool {
        !matches!(self, Comparison::Neq)
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