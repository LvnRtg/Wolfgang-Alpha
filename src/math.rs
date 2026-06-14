//! Aggregates all submodules related to math, namely:
//! - `matrices_and_vectors`: implements operations and various functions for matrices and vectors with variable dimensions.
//! - `operations`: contains enums of various binary/unary operations as well as rudimentary implementations associated with them.
//! - `objects`: contains definitions and basic implementations of `Object` and `FunctionRepr`.
//! - `expressions`: contains definition and basic implementations of `Expression`.
//! - `differentiation`: contains functions to analytically or numerically differentiate expressions/functions (either partially or directionally).
//! - `utils`: a collection of small helper functions. This module lies at the very bottom in the hierachy.
//! 
//! Some common enums/structs/etc. are made directly accessible, e.g. `Matrix` and `Vector`.

use std::collections::HashMap;

pub mod differentiation;
pub mod expressions;
pub mod matrices_and_vectors;
pub mod objects;
pub mod operations;
pub mod optimization;
pub mod utils;

pub use crate::math::expressions::Expression;
pub use crate::math::matrices_and_vectors::{Matrix, Vector};
pub use crate::math::objects::{Object, DirectFunction, FunctionRepr};
pub use crate::math::operations::{Comparison, BinaryOperation, UnaryOperation, FoldedOperation};

pub struct Env {
    pub constants: HashMap<String, Object>,
    pub functions: HashMap<String, FunctionRepr>
}