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

pub mod complex;
pub mod differentiation;
pub mod expressions;
pub mod integration;
pub mod matrices_and_vectors;
pub mod objects;
pub mod operations;
pub mod optimization;
pub mod utils;

pub use crate::math::complex::Complex;
pub use crate::math::expressions::Expression;
pub use crate::math::matrices_and_vectors::{Matrix, Vector};
pub use crate::math::objects::{Object, DirectFunction, FunctionRepr};
pub use crate::math::operations::{Comparison, BinaryOperation, UnaryOperation, FoldedOperation};

/// Set this constant such that `BLOCK^2 * 8` fits in your L1 Cache. Find out the capacity of the latter by running `sudo lshw -C memory`.
/// 
/// My L1 Cache is 512 KiB bit, so I set the constant to 128 (256 would theoretically fit, but I want to leave some space for potential other things).
pub const BLOCK_SIZE: usize = 64;

#[derive(Clone)]
pub struct Env {
    pub constants: HashMap<String, Object>,
    pub functions: HashMap<String, FunctionRepr>
}

impl Env {
    /// For every non-`DirectFunction`-entry in `other`, updates `self` to correspond to that entry.
    pub fn update(&mut self, other: Env) {
        for (s, c) in other.constants {
            self.constants.insert(s, c);
        }
        for (s, f) in other.functions {
            if let FunctionRepr::ByExpression(..) = &f {
                self.functions.insert(s, f);
            }
        }
    }
}

#[derive(Debug)]
pub enum VarStack<'a> {
    Empty,
    Frame {
        vars: &'a HashMap<&'a String, &'a Object>,
        parent: &'a VarStack<'a>,
    },
}

impl<'a> VarStack<'a> {
    pub fn lookup(&self, key: &String) -> Option<&Object> {
        match self {
            VarStack::Empty => None,
            VarStack::Frame { vars, parent } => {
                vars.get(key).copied().or_else(|| parent.lookup(key))
            }
        }
    }
}