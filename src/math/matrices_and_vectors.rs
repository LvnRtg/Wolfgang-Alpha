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
use std::f64::consts::PI;

use crate::math::utils;

/// Set this constant such that `BLOCK^2 * 8` fits in your L1 Cache. Find out the capacity of the latter by running `sudo lshw -C memory`.
/// 
/// My L1 Cache is 512 KiB bit, so I set the constant to 128 (256 would theoretically fit, but I want to leave some space for potential other things).
const BLOCK: usize = 64;


#[derive(Clone, PartialEq)]
pub struct Vector {
    pub values: Vec<f64>
}
#[derive(Clone, PartialEq)]
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
            let rhs_transposed = rhs.transpose(); // Improves Cache locality, only O(n²)
            let mut values = Vec::<f64>::with_capacity(self.m * rhs.n);
            for i in 0..self.m {
                for j in 0..rhs.n {
                    values.push(
                        (0..self.n)
                            .map(|k| {
                                self.values[i * self.n + k]
                                    * rhs_transposed.values[j * rhs_transposed.n + k]
                            })
                            .sum(),
                    )
                }
            }
            Some(Matrix {
                m: self.m,
                n: rhs.n,
                values,
            })
        }
    }
}


pub enum VectorNorm {
    P(f64)
}
pub enum MatrixNorm {
    P(f64),
    Frobenius,
}


impl Vector {
    pub fn zeros(n: usize) -> Vector {
        Vector { values: vec![0.0; n] }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Replaces every component `x` of the vector by `f(x)`.
    pub fn transform<F>(&mut self, f: F) where F: Fn(f64) -> f64 {
        for x in self.values.iter_mut() {
            *x = f(*x);
        }
    }
    /// Creates a new vector by applying f to every element of `self` while consuming `self`.
    pub fn into_new<F>(self, f: F) -> Vector where F: Fn(f64) -> f64 {
        Vector{values: self.values.into_iter().map(f).collect()}
    }

    /// Returns the norm of this vector (w.r.t. the given norm).
    pub fn norm(&self, norm_type: &VectorNorm) -> f64 {
        match norm_type {
            VectorNorm::P(f64::INFINITY) => utils::max_abs(self.values.iter()),
            VectorNorm::P(p) => self.values.iter().map(|x| x.powf(*p)).sum::<f64>().powf(1.0 / *p),
        }
    }

    /// Returns the dual of `self` w.r.t. the p-norm. Since (l^p)^* and l^q are isometrically isomorphic for q s.t. 1/p+1/q=1
    /// (standard result from functional analysis), the dual of `self` can be identified by a vector `v*` s.t.
    /// `<v*, self> = ||self||_p`.
    /// 
    /// In this function, we return that `v` with the additional constraint `||v*||_q = 1`.
    /// 
    /// # Panics
    /// Panics if `p<1`.
    pub fn dual(&self, p: f64) -> Vector {
        assert!(p >= 1.0, "p must be >= 1, got {p}");
    
        let n = self.len();
        let supnorm = self.norm(&VectorNorm::P(f64::INFINITY));
        if supnorm == 0.0 {
            return self.clone();
        }
    
        if p == 1.0 {
            // Then, `q = \infty`, so the dual is simply `self.values.map(sign)`.
            Vector { values: self.values.iter().map(|x| x.signum()).collect() }
        } else if p == f64::INFINITY {
            // Then, `q = 1`, so the dual is simply the unit vector pointing in direction `argmax_i |self[i]|`.
            let mut i: usize = 0; let mut highest_abs = 0.0;
            for (j, x) in self.values.iter().enumerate() {
                if *x > highest_abs {
                    highest_abs = *x;
                    i = j;
                }
            }
            let mut dual = Vector::zeros(n);
            dual[i] = self[i].signum();
            dual
        } else {
            let q = 1.0 / (1.0 - 1.0 / p);
            let mut dual = Vector {values: self.values.iter().map(|x| x.signum() * (x / supnorm).abs().powf(p - 1.0)).collect()};
            dual /= dual.norm(&VectorNorm::P(q));
            dual
        }
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

    /// Replaces every component `x` of the matrix by `f(x)`.
    pub fn transform<F>(&mut self, f: F) where F: Fn(f64) -> f64 {
        for x in self.values.iter_mut() {
            *x = f(*x);
        }
    }
    /// Creates a new matrix by applying f to every element of `self` while consuming `self`.
    pub fn into_new<F>(self, f: F) -> Matrix where F: Fn(f64) -> f64 {
        Matrix{m: self.m, n: self.n, values: self.values.into_iter().map(f).collect()}
    }

    pub fn row(&self, i: usize) -> Vector {
        Vector { values: self.values[(i*self.n)..(i+1)*self.n].to_vec() }
    }
    pub fn col(&self, j: usize) -> Vector {
        Vector { values: (0..self.m).map(|i| self.get(i, j)).collect() }
    }

    pub fn transpose(&self) -> Matrix {
        // Cache friendly. By definition, a `BLOCK x BLOCK` tile of the matrix fits into the L1 cache entirely.
        // Hence, we transpose tile by tile.
        let mut values = vec![0.0f64; self.values.len()];
        for i_block in (0..self.m).step_by(BLOCK) {
            for j_block in (0..self.n).step_by(BLOCK) {
                let i_end = (i_block + BLOCK).min(self.m);
                let j_end = (j_block + BLOCK).min(self.n);
                for i in i_block..i_end {
                    for j in j_block..j_end {
                        // Read: row-major access within tile (sequential)
                        // Write: row-major in output (sequential within tile)
                        values[j * self.m + i] = self.values[i * self.n + j];
                    }
                }
            }
        }

        Matrix { m: self.n, n: self.m, values }
    }

    pub fn identity(n: usize) -> Matrix {
        if n == 0 {return Matrix{m: 0, n: 0, values: Vec::<f64>::new()};}
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
    
    pub fn norm(&self, norm_type: &MatrixNorm) -> Result<f64, String> {
        match norm_type {
            // The sup-norm is simply the highest row sum, i.e. \max_i \sum_{j=1}^n |a_{i,j}|
            MatrixNorm::P(f64::INFINITY) => Ok(utils::max(
                (0..self.m).map(
                    |i| (0..self.n).map(
                        |j| self.get(i, j).abs()
                    ).sum()
                )
            )),
            // The 1-norm is the highest column sum. We take a different approach than above to improve cache locality.
            MatrixNorm::P(1.0) => {
                let mut sums = vec![0.0; self.n];
                for i in 0..self.m {
                    sums.iter_mut().enumerate().for_each(|(j, x)| *x += self.get(i, j).abs());
                }
                Ok(utils::max(sums.into_iter()))
            }
            MatrixNorm::P(2.0) => {
                // TODO
                unimplemented!()
            }
            // This implementation is based on the article "Estimating the matrix p-norm" by Nicholas Highham,
            // https://link.springer.com/article/10.1007/BF01396242. For explanations, see the article.
            MatrixNorm::P(p) if *p >= 1.0 => {
                // All `unwrap`s below are safe because the dimensions of the operands fit.
                let q = if *p == 1.0 {
                    f64::INFINITY
                } else {
                    1.0 / (1.0 - 1.0 / p)
                };
                let samples = 9; // Could theoretically be increased or reduced (until 2), but the default is 9.
                let tolerance = 1e-10;
                let mut y = Vector::zeros(self.m);
                let mut x = Vector::zeros(self.n);

                // Initialisation: block power method with angle sampling.
                for k in 0..self.n {
                    let (c, s) = if k == 0 {
                        (1.0, 0.0)
                    } else {
                        let col_k = self.col(k);
                        let mut best_f = 0.0_f64;
                        let mut best_c = 1.0_f64;
                        let mut best_s = 0.0_f64;
                        for i in 0..samples {
                            let th = PI * i as f64 / (samples - 1) as f64;
                            let mut cs = Vector{values: vec![th.cos(), th.sin()]};
                            cs /= cs.norm(&VectorNorm::P(*p));
                            let w_cs = (&(cs[0] * &col_k) + &(cs[1] * &y)).unwrap();
                            let f = w_cs.norm(&VectorNorm::P(*p));
                            if f > best_f {
                                best_f = f;
                                best_c = cs[0];
                                best_s = cs[1];
                            }
                        }
                        (best_c, best_s)
                    };
                    x[k] = c;
                    y = (&(c * &self.col(k)) + &(s * &y)).unwrap();
                    if k > 0 {
                        for xi in x.values.iter_mut().take(k) {
                            *xi *= s;
                        }
                    }
                }

                // Refinement: power iteration with dual vectors.
                let mut est = y.norm(&VectorNorm::P(*p));
                for iter in 1usize.. {
                    y = (self * &x).unwrap();
                    let eo = est;
                    est = y.norm(&VectorNorm::P(*p));
                    let dv_y = y.dual(*p);
                    // Slightly hacky; instead of `self^\top * dv_y`, we write `dv_y * self`, which I implemented as `(dv_y^\top * self)^\top` for convenience
                    // (the other operation would be undefined anyway), which in turn is mathematically exactly `self^\top * dv_y`.
                    let z = (&dv_y * self).unwrap();
                    let z_q_norm = z.norm(&VectorNorm::P(q));
                    if iter > 1 && (z_q_norm < (&z * &x).unwrap() || (est - eo).abs() <= tolerance * est) {
                        break;
                    }
                    x = z.dual(q);
                }
                Ok(est)
            }
            MatrixNorm::P(other) => Err(format!("Parameter `p` must be at least 1 (got {other}).")),
            MatrixNorm::Frobenius => Ok(self.values.iter().map(|x| x.powi(2)).sum::<f64>().sqrt())
        }
    }
}