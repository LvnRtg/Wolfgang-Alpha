//! Contains definition and basic member functions for vector.

use std::fmt;
use std::fmt::{Debug, Display};

use crate::math::traits::Scalar;

mod norms;
mod ops;

pub use norms::VectorNorm;


#[derive(Clone, PartialEq)]
pub struct Vector<T> {
    pub values: Vec<T>
}


impl<T: Debug> Display for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}", &self.values)
    }
}
impl<T: Debug> Debug for Vector<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "Vec<{}>:\n{:#?}", self.values.len(), &self.values)
        }
        else {
            write!(f, "Vec<{}>: {:?}", self.values.len(), &self.values)
        }
    }
}

impl<T: Default> Default for Vector<T> {
    fn default() -> Vector<T> {
        Vector { values: vec![T::default()] }
    }
}

impl<T> Vector<T> {
    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Creates a new vector by applying f to every element of `self` while consuming `self`.
    pub fn into_new<U, F>(self, f: F) -> Vector<U> where F: Fn(T) -> U {
        Vector{values: self.values.into_iter().map(f).collect()}
    }

    pub fn iter(&self) -> impl Iterator<Item=&T> {
        self.values.iter()
    }
}

impl<T: Copy> Vector<T> {
    /// Replaces every component `x` of the vector by `f(x)`.
    pub fn transform_in_place<F>(&mut self, f: F) where F: Fn(T) -> T {
        for x in self.values.iter_mut() {
            *x = f(*x);
        }
    }
    /// Maps every component `x` of `self` to `f(x)`, returning a new vector.
    pub fn transform<U, F>(&self, f: F) -> Vector<U> where F: Fn(T) -> U {
        Vector{values: self.values.iter().map(|x| f(*x)).collect()}
    }
}


impl<T: Scalar> Vector<T> {
    pub fn zeros(n: usize) -> Vector<T> {
        Vector { values: vec![T::zero(); n] }
    }
}