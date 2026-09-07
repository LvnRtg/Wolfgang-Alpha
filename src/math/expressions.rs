pub mod display;
pub mod macros;
pub mod simplification;
pub mod traversing;
pub mod type_checking;

use crate::math::{Env, VarStack};
use crate::math::operations::{BinaryOperation, FoldedOperation, UnaryOperation};
use crate::status::Status;

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
    /// E.g. `sum_{i=1, i != 3}^n f(i)` will become `FoldedOperation(Sum, "i", 1, [i != j], n, f(i))`.
    /// There can be as many conditions as desired, including none at all.
    FoldedOperation(FoldedOperation, String, Box<Expression>, Vec<Expression>, Box<Expression>, Box<Expression>),
    /// Respectively: function's name and list of arguments passed.
    Function(String, Vec<Expression>),
    /// A collection of comma-separated expressions between parentheses.
    Tuple(Vec<Expression>),
    /// Format: LHS := RHS
    Assignment(Box<Expression>, Box<Expression>),
    /// The partial derivative of the boxed expression w.r.t. the given identifiers in order.
    /// 
    /// For example, if the given `Vec` is `[(x, 2), (y, 1)]`, then the expression is first differentiated twice w.r.t. `x` and then once w.r.t. `y`.
    /// 
    /// Note: one could also use a single `String` instead of `Vec<String>` and express higher derivatives as nested `PartialDerivative` enums.
    /// However, this doesn't allow for expressions like `d^n/dx^n f(x)` if `n` is not an explicit integer.
    PartialDerivative(Vec<(String, Repeat)>, Box<Expression>),
    /// The directional derivative of `SecondArg` at point `ThirdArg` in direction `FourthArg` where the variables w.r.t. which we differentiate are `FirstArg`.
    DirectionalDerivative(Vec<String>, Box<Expression>, Vec<Expression>, Vec<Expression>),
    /// E.g. `int_a^b f(x) dx` gives `Integral(f(x), a, b, x)`.
    Integral(Box<Expression>, Box<Expression>, Box<Expression>, String),
    /// `if (FirstArg) { SecondArg } else { ThirdArg }`
    IfElse(Box<Expression>, Box<Expression>, Box<Expression>)
}

#[derive(Debug, Clone, PartialEq)]
pub enum Repeat {
    Default,
    Number(usize),
    Expression(Expression)
}

impl Repeat {
    pub fn get(&self, varstack: &VarStack, env: &mut Env) -> Result<Status<usize>, String> {
        match self {
            Repeat::Default => Ok(Status::ok(1)),
            Repeat::Number(n) => Ok(Status::ok(*n)),
            Repeat::Expression(e) => {
                crate::lang::eval(e, varstack, env)
                .and_then(|s| s.try_map(|o| o.expect_nonnegative_int()))
            }
        }
    }
}

impl Default for Expression {
    fn default() -> Self {
        Expression::None
    }
}

impl Expression {
    pub fn expect_ident(&self) -> Result<&String, String> {
        match self {
            Expression::Identifier(id) => Ok(id),
            other => Err(format!("Expected identifier, found `{:?}`.", other))
        }
    }
}