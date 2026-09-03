//! Aggregates all submodules related to math, namely:
//! - `matrices_and_vectors`: implements operations and various functions for matrices and vectors with variable dimensions.
//! - `operations`: contains enums of various binary/unary operations as well as rudimentary implementations associated with them.
//! - `objects`: contains definitions and basic implementations of `Object` and `FunctionRepr`.
//! - `expressions`: contains definition and basic implementations of `Expression`.
//! - `differentiation`: contains functions to analytically or numerically differentiate expressions/functions (either partially or directionally).
//! - `utils`: a collection of small helper functions. This module lies at the very bottom in the hierachy.
//! 
//! Some common enums/structs/etc. are made directly accessible, e.g. `Matrix` and `Vector`.

pub mod complex;
pub mod differentiation;
pub mod env;
pub mod expressions;
pub mod integration;
pub mod matrices_and_vectors;
pub mod objects;
pub mod operations;
pub mod optimization;
pub mod utils;

pub use crate::math::complex::Complex;
pub use crate::math::env::{Env, VarStack, VarStackLookup};
pub use crate::math::expressions::Expression;
pub use crate::math::matrices_and_vectors::{Matrix, Vector};
pub use crate::math::objects::{DirectFunction, FunctionRepr, Object, ObjType};

/// Set this constant such that `BLOCK^2 * 8` fits in your L1 Cache. Find out the capacity of the latter by running `sudo lshw -C memory`.
/// 
/// My L1 Cache is 512 KiB bit, so I set the constant to 128 (256 would theoretically fit, but I want to leave some space for potential other things).
pub const BLOCK_SIZE: usize = 64;