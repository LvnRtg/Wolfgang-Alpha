//! Implements basic arithmetic operations for matrices and vectors.

use std::ops;
use std::slice::SliceIndex;

use std::cmp::min;
use crate::math::{Matrix, Vector};

// Indexing just operates on the values directly
impl<I> ops::Index<I> for Vector where I: SliceIndex<[f64]> {
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output {
        &self.values[index]
    }
}
impl<I> ops::IndexMut<I> for Vector where I: SliceIndex<[f64]> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.values[index]
    }
}

// In most implementations of operations (e.g. std::Add), we do not need to transfer ownership, since we only need to read
// the values of LHS and RHS. Therefore, we will first implement the operation "&a + &b", which is realizable without
// transfer of ownership.
// Theoretically, "a + b" may still be useful for ergonomic purposes (e.g. "myvec + Vector {...}", where the RHS is a throwaway vector
// and myvec isn't used again later). However, so are "a + &b" and "&a + b" (e.g. "&myvec + Vector {...}" when we do want to
// reuse 'myvec' later), which one would then have to implement too. The initial idea of defining "a+b := &a + &b" might be inefficient
// because the compiler may not be able to optimize it away. On the contrary, rewriting the full implementation for
// "a+b", "&a+b" and "a+&b" will increase the code size by a lot (and thus probably the binary size too).
// => For the moment, we only implement "&a + &b" and rely on the fact that the developer will not be too lazy to add two '&' in every call.

impl ops::Add<&Vector> for &Vector {
    type Output = Option<Vector>;
    fn add(self, rhs: &Vector) -> Self::Output {
        if self.values.len() != rhs.values.len() {
            return None;
        }
        Some(Vector {
            values: self.values.iter().zip(&rhs.values).map(|(x, y)| x+y).collect::<Vec<f64>>()
        })
    }
}
/// Behavior:
/// - If the RHS is shorter than the LHS, treat it as if it were extended by zeros.
/// - If the RHS is longer than the LHS, ignore the trailing values.
impl ops::AddAssign<&Vector> for Vector {
    fn add_assign(&mut self, rhs: &Vector) {
        for i in 0..min(self.values.len(), rhs.values.len()) {
            self.values[i] += rhs.values[i];
        }
    }
}
impl ops::Sub<&Vector> for &Vector {
    type Output = Option<Vector>;
    fn sub(self, rhs: &Vector) -> Self::Output {
        if self.values.len() != rhs.values.len() {
            return None;
        }
        Some(Vector {
            values: self.values.iter().zip(&rhs.values).map(|(x, y)| x-y).collect::<Vec<f64>>()
        })
    }
}
/// Behavior:
/// - If the RHS is shorter than the LHS, treat it as if it were extended by zeros.
/// - If the RHS is longer than the LHS, ignore the trailing values.
impl ops::SubAssign<&Vector> for Vector {
    fn sub_assign(&mut self, rhs: &Vector) {
        for i in 0..min(self.values.len(), rhs.values.len()) {
            self.values[i] -= rhs.values[i];
        }
    }
}
/// Multiplication with a "constant" (i.e. a float) is done component-wise.
/// Operator is implemented as commutative.
// The same goes for division, negation and modulo.
impl ops::Mul<f64> for &Vector {
    type Output = Vector;
    fn mul(self, rhs: f64) -> Self::Output {
        Vector {
            values: self.values.iter().map(|x| x * rhs).collect::<Vec<f64>>()
        }
    }
}
impl<'a> ops::Mul<&'a Vector> for f64 {
    type Output = Vector;
    fn mul(self, rhs: &'a Vector) -> Self::Output {
        Vector {
            values: rhs.values.iter().map(|x| self * x).collect::<Vec<f64>>()
        }
    }
}
impl ops::MulAssign<f64> for Vector {
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..self.values.len() {
            self.values[i] *= rhs;
        }
    }
}
/// Note: for the operations vector/constant and constant/vector, order matters.
impl ops::Div<f64> for &Vector {
    type Output = Vector;
    fn div(self, rhs: f64) -> Self::Output {
        Vector {
            values: self.values.iter().map(|x| x / rhs).collect::<Vec<f64>>()
        }
    }
}
impl<'a> ops::Div<&'a Vector> for f64 {
    type Output = Vector;
    fn div(self, rhs: &'a Vector) -> Self::Output {
        Vector {
            values: rhs.values.iter().map(|x| self / x).collect::<Vec<f64>>()
        }
    }
}
impl ops::DivAssign<f64> for Vector {
    fn div_assign(&mut self, rhs: f64) {
        for i in 0..self.values.len() {
            self.values[i] /= rhs;
        }
    }
}
impl ops::Rem<f64> for &Vector {
    type Output = Vector;
    fn rem(self, rhs: f64) -> Self::Output {
        Vector {
            values: self.values.iter().map(|x| x.rem_euclid(rhs)).collect::<Vec<f64>>()
        }
    }
}
impl<'a> ops::Rem<&'a Vector> for f64 {
    type Output = Vector;
    fn rem(self, rhs: &'a Vector) -> Self::Output {
        Vector {
            values: rhs.values.iter().map(|x| self.rem_euclid(*x)).collect::<Vec<f64>>()
        }
    }
}
impl ops::RemAssign<f64> for Vector {
    fn rem_assign(&mut self, rhs: f64) {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].rem_euclid(rhs);
        }
    }
}
impl ops::Neg for &Vector {
    type Output = Vector;
    fn neg(self) -> Self::Output {
        Vector {
            values: self.values.iter().map(|x| -x).collect::<Vec<f64>>()
        }
    }
}


// Same rationale for matrix operations as for vectors
/// Slightly more inefficient than the version with type parameter since we have to check whether the dimensions match
impl ops::Add<&Matrix> for &Matrix {
    type Output = Option<Matrix>;
    fn add(self, rhs: &Matrix) -> Self::Output {
        if self.m != rhs.m || self.n != rhs.n {
            return None;
        }
        Some(Matrix {
            m: self.m,
            n: self.n,
            values: self.values.iter().zip(&rhs.values).map(|(x, y)| x+y).collect()
        })
    }
}
/// Panics if dimensions mismatch.
impl ops::AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &Matrix) {
        assert!(self.m == rhs.m && self.n == rhs.n, "Dimension mismatch in add_assign between {:?} and {:?}", self, *rhs);
        for i in 0..self.values.len() {
            self.values[i] += rhs.values[i];
        }
    }
}
impl ops::Sub<&Matrix> for &Matrix {
    type Output = Option<Matrix>;
    fn sub(self, rhs: &Matrix) -> Self::Output {
        if self.m != rhs.m || self.n != rhs.n {
            return None;
        }
        Some(Matrix {
            m: self.m,
            n: self.n,
            values: self.values.iter().zip(&rhs.values).map(|(x, y)| x-y).collect()
        })
    }
}
/// Panics if dimensions mismatch.
impl ops::SubAssign<&Matrix> for Matrix {
    fn sub_assign(&mut self, rhs: &Matrix) {
        assert!(self.m == rhs.m && self.n == rhs.n, "Dimension mismatch in sub_assign between {:?} and {:?}", self, *rhs);
        for i in 0..self.values.len() {
            self.values[i] -= rhs.values[i];
        }
    }
}
/// Multiplication with a "constant" (i.e. a float) is done component-wise.
/// Operator is implemented as commutative.
// The same goes for division, negation and modulo.
impl ops::Mul<f64> for &Matrix {
    type Output = Matrix;
    fn mul(self, rhs: f64) -> Self::Output {
        Matrix {
            m: self.m,
            n: self.n,
            values: self.values.iter().map(|x| x * rhs).collect()
        }
    }
}
impl<'a> ops::Mul<&'a Matrix> for f64 {
    type Output = Matrix;
    fn mul(self, rhs: &'a Matrix) -> Self::Output {
        Matrix {
            m: rhs.m,
            n: rhs.n,
            values: rhs.values.iter().map(|x| self * x).collect()
        }
    }
}
impl ops::MulAssign<f64> for Matrix {
    fn mul_assign(&mut self, rhs: f64) {
        for i in 0..self.values.len() {
            self.values[i] *= rhs;
        }
    }
}
/// Note: for the operations vector/constant and constant/vector, order matters.
impl ops::Div<f64> for &Matrix {
    type Output = Matrix;
    fn div(self, rhs: f64) -> Self::Output {
        Matrix {
            m: self.m,
            n: self.n,
            values: self.values.iter().map(|x| x / rhs).collect()
        }
    }
}
impl<'a> ops::Div<&'a Matrix> for f64 {
    type Output = Matrix;
    fn div(self, rhs: &'a Matrix) -> Self::Output {
        Matrix {
            m: rhs.m,
            n: rhs.n,
            values: rhs.values.iter().map(|x| self / x).collect()
        }
    }
}
impl ops::DivAssign<f64> for Matrix {
    fn div_assign(&mut self, rhs: f64) {
        for i in 0..self.values.len() {
            self.values[i] /= rhs;
        }
    }
}
impl ops::Rem<f64> for &Matrix {
    type Output = Matrix;
    fn rem(self, rhs: f64) -> Self::Output {
        Matrix {
            m: self.m,
            n: self.n,
            values: self.values.iter().map(|x| x.rem_euclid(rhs)).collect()
        }
    }
}
impl<'a> ops::Rem<&'a Matrix> for f64 {
    type Output = Matrix;
    fn rem(self, rhs: &'a Matrix) -> Self::Output {
        Matrix {
            m: rhs.m,
            n: rhs.n,
            values: rhs.values.iter().map(|x| self.rem_euclid(*x)).collect()
        }
    }
}
impl ops::RemAssign<f64> for Matrix {
    fn rem_assign(&mut self, rhs: f64) {
        for i in 0..self.values.len() {
            self.values[i] = self.values[i].rem_euclid(rhs);
        }
    }
}
impl ops::Neg for &Matrix {
    type Output = Matrix;
    fn neg(self) -> Self::Output {
        Matrix {
            m: self.m,
            n: self.n,
            values: self.values.iter().map(|x| -x).collect()
        }
    }
}

/// Multiplication of vectors is treated as the euclidian inner product (i.e. v * v := v^t * v).
/// This makes it more convenient to obtain the inner product.
/// 
/// Returns None in case the dimensions mismatch.
impl ops::Mul<&Vector> for &Vector {
    type Output = Option<f64>;
    fn mul(self, rhs: &Vector) -> Self::Output {
        if self.values.len() != rhs.values.len() {
            None
        }
        else {
            Some(self.values.iter().zip(&rhs.values).map(|(x, y)| x*y).sum())
        }
    }
}
/// Returns None in case the dimensions mismatch.
impl ops::Mul<&Vector> for &Matrix {
    type Output = Option<Vector>;
    fn mul(self, rhs: &Vector) -> Self::Output {
        if self.n != rhs.values.len() {
            None
        }
        else {
            Some(Vector{ values: (0..self.m).map(|i|
                Vector::unchecked_dot(self.row_slice(i), &rhs.values)
            ).collect()})
        }
    }
}
/// This is mathematically not perfectly accurate, because one can only multiply a flipped vector with a matrix,
/// but this slight lack of rigorousness is less expensive than re-implementing all functions for a new type 'FlippedVector' or using a 1xn-matrix.
/// 
/// Returns `None` in case the dimensions mismatch.
impl ops::Mul<&Matrix> for &Vector {
    type Output = Option<Vector>;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        if self.values.len() != rhs.m {
            None
        }
        else {
            let mut res = Vector::zeros(rhs.n);
            for k in 0..rhs.m { // Iterate in this order for better cache locality
                for i in 0..rhs.n {
                    res.values[i] += self.values[k] * rhs.get(k, i);
                }
            }
            Some(res)
        }
    }
}