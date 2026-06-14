//! The explanation of the syntax can be found in `README.md`.

pub mod lexer;
pub mod parser;
pub mod evaluator;

pub use lexer::tokenize;
pub use parser::Parser;
pub use evaluator::eval;