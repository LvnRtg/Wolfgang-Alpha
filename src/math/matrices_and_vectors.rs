//! Most things in this module are written such that user interaction is smoother,
//! e.g. by having vectors and matrices of variable size.
//! However, this makes computation slightly longer, so this file shouldn't be
//! used for intensive computations.
//! It is tailored to the use case of a calculator where performance doesn't matter
//! that much and flexibility to define variables as desired is more important.

use std::ops;
use std::slice::SliceIndex;
use std::fmt;
use std::cmp::min;

use crate::math::utils;


pub struct Vector {
    pub values: Vec<f64>
}
pub struct Matrix {
    pub m: usize,
    pub n: usize,
    values: Vec<f64>
}

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

impl Clone for Vector {
    fn clone(&self) -> Self {
        Vector { values: self.values.clone() }
    }
}

impl PartialEq for Vector {
    fn eq(&self, other: &Self) -> bool {
        self.values.len() == other.values.len()
        && (0..self.values.len()).all(|i| self.values[i] == other.values[i])
    }
}
impl Eq for Vector {}

// Current output format: Vec<5>: [3, 1, 4, 1, 5]
impl fmt::Debug for Vector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            write!(f, "Vec<{}>:\n{:#?}", self.values.len(), &self.values)
        }
        else {
            write!(f, "Vec<{}>: {:?}", self.values.len(), &self.values)
        }
    }
}

impl Default for Vector {
    fn default() -> Vector {
        Vector { values: vec![0.0] }
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
            values: (0..self.values.len()).map(|i| self.values[i] + rhs.values[i]).collect::<Vec<f64>>()
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
            values: (0..self.values.len()).map(|i| self.values[i] - rhs.values[i]).collect::<Vec<f64>>()
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
            values: self.values.iter().map(|x| x % rhs).collect::<Vec<f64>>()
        }
    }
}
impl<'a> ops::Rem<&'a Vector> for f64 {
    type Output = Vector;
    fn rem(self, rhs: &'a Vector) -> Self::Output {
        Vector {
            values: rhs.values.iter().map(|x| self % x).collect::<Vec<f64>>()
        }
    }
}
impl ops::RemAssign<f64> for Vector {
    fn rem_assign(&mut self, rhs: f64) {
        for i in 0..self.values.len() {
            self.values[i] %= rhs;
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


// impl<I> ops::Index<I> for Matrix where I: SliceIndex<[Vec<f64>]> {
//     type Output = I::Output;
//     fn index(&self, index: I) -> &Self::Output {
//         &self.values[index]
//     }
// }
// impl<I> ops::IndexMut<I> for Matrix where I: SliceIndex<[Vec<f64>]> {
//     fn index_mut(&mut self, index: I) -> &mut Self::Output {
//         &mut self.values[index]
//     }
// }

impl Matrix {
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[i * self.n + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.values[i * self.n + j] = value;
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        Matrix { m: self.m, n: self.n, values: self.values.clone() }
    }
}

impl PartialEq for Matrix {
    fn eq(&self, other: &Self) -> bool {
        self.m == other.m
        && self.n == other.n
        && (0..self.m*self.n).all(|i| self.values[i] == other.values[i])
    }
}
impl Eq for Matrix {}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if f.alternate() {
            let _ = write!(f, "Matrix ({}x{}):\n[", self.m, self.n);
            for i in 0..self.m {
                let _ = write!(f, "\n    {:?}{}", &self.values[(i*self.n)..(i+1)*self.n], if i == self.m-1 {""} else {","});
            }
            write!(f, "\n]")
        }
        else {
            write!(f, "Matrix ({}x{}): {:?}", self.m, self.n, &self.values)
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
            // `self.values.len()` is faster than `self.m * self.n` since the length is already stored in the vector's metadata
            values: (0..self.values.len()).map(|i| self.values[i] + rhs.values[i]).collect()
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
            values: (0..self.values.len()).map(|i| self.values[i] - rhs.values[i]).collect()
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
            values: self.values.iter().map(|x| x % rhs).collect()
        }
    }
}
impl<'a> ops::Rem<&'a Matrix> for f64 {
    type Output = Matrix;
    fn rem(self, rhs: &'a Matrix) -> Self::Output {
        Matrix {
            m: rhs.m,
            n: rhs.n,
            values: rhs.values.iter().map(|x| self % x).collect()
        }
    }
}
impl ops::RemAssign<f64> for Matrix {
    fn rem_assign(&mut self, rhs: f64) {
        for i in 0..self.values.len() {
            self.values[i] %= rhs;
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

/// Multiplication of vectors is treated as the euclidian inner product (i.e. v * v := v^\top * v).
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
            Some((0..self.values.len()).map(|i| self.values[i] * rhs.values[i]).sum())
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
            Some(Vector{ values: (0..self.m).map(
                |i| (0..self.n).map(|k| self.get(i, k) * rhs.values[k]).sum()
            ).collect()})
        }
    }
}
/// This is mathematically not perfectly accurate, because one can only multiply a flipped vector with a matrix,
/// but this slight lack of rigorousness is less expensive than re-implementing all functions for a new type 'FlippedVector' or using a 1xn-matrix.
/// 
/// Returns None in case the dimensions mismatch.
impl ops::Mul<&Matrix> for &Vector {
    type Output = Option<Vector>;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        if self.values.len() != rhs.m {
            None
        }
        else {
            Some(Vector{ values: (0..rhs.n).map(
                |i| (0..rhs.m).map(|k| self.values[k] * rhs.get(k, i)).sum()
            ).collect()})
        }
    }
}
/// Returns None in case the dimensions mismatch.
impl ops::Mul<&Matrix> for &Matrix {
    type Output = Option<Matrix>;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        if self.n != rhs.m {
            None
        }
        else {
            let mut values = Vec::<f64>::with_capacity(self.m * rhs.n);
            for i in 0..self.m {
                for j in 0..rhs.n {
                    values.push((0..self.n).map(|k| self.get(i, k) * rhs.get(k, j)).sum())
                }
            }
            Some(Matrix{ m: self.m, n: rhs.n, values })
        }
    }
}

impl Vector {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Returns the standard euclidian R^n-norm of the vector.
    pub fn norm(&self) -> f64 {
        self.values.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
    }
}

impl Matrix {
    pub fn zeros(m: usize, n: usize) -> Matrix {
        Matrix {
            m, n,
            values: vec![0.0; m*n]
        }
    }

    /// Returns `self.values.iter()`. Encapsulated in order to keep the field `values` private.
    pub fn iter_values(&self) -> std::slice::Iter<'_, f64> {
        self.values.iter()
    }

    /// Encapsulated in order to keep the field `values` private.
    pub fn from(m: usize, n: usize, values: Vec<f64>) -> Matrix {
        Matrix{m, n, values}
    }

    pub fn row(&self, i: usize) -> Vector {
        Vector { values: self.values[(i*self.n)..(i+1)*self.n].to_vec() }
    }
    pub fn col(&self, j: usize) -> Vector {
        Vector { values: (0..self.m).map(|i| self.get(i, j)).collect() }
    }

    pub fn transpose(&self) -> Matrix {
        let mut values = Vec::<f64>::with_capacity(self.values.len());
        for j in 0..self.n {
            for i in 0..self.m {
                values.push(self.get(i, j));
            }
        }
        Matrix {m: self.n, n: self.m, values}
    }

    pub fn identity(n: usize) -> Matrix {
        let mut values = Vec::<f64>::with_capacity(n*n);
        values.push(1.0);
        for _ in 0..n-1 {
            values.extend(std::iter::repeat_n(0.0, n));
            values.push(1.0);
        }
        Matrix{m: n, n, values}
    }

    /// Standard algorithm to compute the LU decomposition.
    /// Time complexity: 2/3 * n^3 + O(n^2)
    pub fn lu_decomposition(&self) -> Option<(Matrix, Matrix)> {
        // First, ensure the matrix is square.
        if self.m != self.n {
            None
        }
        else {
            let mut l = vec![0.0; self.m * self.m];
            let mut u = Vec::<f64>::with_capacity(self.m * self.m);
            for i in 0..self.m {
                u.extend(std::iter::repeat_n(0.0, i));
                for k in i..self.m {
                    let newval = if i > 0 {
                        self.get(i, k) - (0..i).map(|j| l[i*self.m + j] * u[j * self.m + k]).sum::<f64>()
                    } else { self.get(i, k) };
                    u.push(newval); // Can't condense into single line because of ownership of u
                }
                for k in i..self.m {
                    if utils::approx_eq(&u[i*self.m + i], &0.0) {
                        return None; // No LU decomposition
                    }
                    l[k*self.m + i] = (if i > 0 {
                        self.get(k, i) - (0..i).map(|j| l[k*self.m + j] * u[j*self.m + i]).sum::<f64>()
                    } else { self.get(k, i) }) / u[i*self.m + i];
                }
            }
            Some((Matrix{m: self.m, n: self.m, values: l}, Matrix{m: self.m, n: self.m, values: u}))
        }
    }

    /// Returns the product of all diagonal entries of the matrix.
    fn diag_product(&self) -> f64 {
        (0..self.m).fold(1.0, |acc, i| acc * self.get(i, i))
    }

    /// Returns the determinant of the matrix. Currently, this is computed via an LU-decomposition,
    /// with a time complexity of 2/3 * n^3 + O(n^2).
    pub fn det(&self) -> f64 {
        if let Some((l, u)) = self.lu_decomposition() {
            l.diag_product() * u.diag_product()
        } else {
            // If no LU-decomposition exists, there exists some linear dependency between rows.
            // This immediately implies that the matrix is not invertible, that is, it has determinant zero.
            0.0
        }
    }
}