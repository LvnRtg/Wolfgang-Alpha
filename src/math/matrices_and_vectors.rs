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

use crate::lang;
use crate::math::utils;
use crate::math::{Env, Expression, Object};

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
                                self.get(i, k) * rhs_transposed.get(j, k)
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
impl VectorNorm {
    /// If `opt` is `None`, use the euclidian 2-norm. If `opt` is "inf" or "infty", use the supremum norm.
    /// Otherwise, evaluate `opt` and use the corresponding p-norm.
    pub fn from_expr(opt: &Option<Box<Expression>>, extra_vars: &lang::evaluator::VarStack, env: &mut Env) -> Result<VectorNorm, String> {
        if let Some(inner) = opt {match &**inner {
            Expression::Identifier(ident) if ident == "inf" || ident == "infty"
                => Ok(VectorNorm::P(f64::INFINITY)),
            other => {
                if let Object::Float(z) = lang::eval(other, extra_vars, env)? {
                    Ok(VectorNorm::P(z))
                }
                else {
                    Err(format!("Couldn't evaluate {other} to float."))
                }
            }
        }} else {Ok(VectorNorm::P(2.0))}
    }
}

pub enum MatrixNorm {
    P(f64),
    Frobenius,
}
impl MatrixNorm {
    /// If `opt` is `None`, use the euclidian 2-norm. If `opt` is "inf" or "infty", use the supremum norm.
    /// Otherwise, evaluate `opt` and use the corresponding p-norm.
    pub fn from_expr(opt: &Option<Box<Expression>>, extra_vars: &lang::evaluator::VarStack, env: &mut Env) -> Result<MatrixNorm, String> {
        if let Some(inner) = opt {match &**inner {
            Expression::Identifier(ident) if ident == "inf" || ident == "infty"
                => Ok(MatrixNorm::P(f64::INFINITY)),
            Expression::Identifier(ident) if ident.starts_with('f')
                => Ok(MatrixNorm::Frobenius),
            other => {
                if let Object::Float(z) = lang::eval(other, extra_vars, env)? {
                    Ok(MatrixNorm::P(z))
                }
                else {
                    Err(format!("Couldn't evaluate {other} to float."))
                }
            }
        }} else {Ok(MatrixNorm::P(2.0))}
    }
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

    /// Constructs a diagonal matrix with diagonal entries `diag_values`.
    pub fn diag(diag_values: &[f64]) -> Matrix {
        let n = diag_values.len();
        if n == 0 {return Matrix{m: 0, n: 0, values: Vec::<f64>::new()};}
        let mut values = Vec::<f64>::with_capacity(n * n);
        values.push(diag_values[0]);
        for v in diag_values.iter().skip(1) {
            values.extend(std::iter::repeat_n(0.0, n));
            values.push(*v);
        }
        Matrix{m: n, n, values}
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

    /// Multiplies `self` with `rhs^t`. Returns `None` if the dimensions don't match.
    pub fn mul_with_transposed(self, rhs: &Matrix) -> Option<Matrix> {
        if self.n != rhs.n {
            None
        }
        else {
            let mut values = Vec::<f64>::with_capacity(self.m * rhs.m);
            for i in 0..self.m {
                for j in 0..rhs.m {
                    values.push(
                        (0..self.n)
                            .map(|k| {
                                self.get(i, k) * rhs.get(j, k)
                            })
                            .sum(),
                    )
                }
            }
            Some(Matrix {
                m: self.m,
                n: rhs.m,
                values,
            })
        }
    }

    /// Applies the given permutation to the rows of `self`.
    /// 
    /// `permutation` should be a permutation of the vector `[0, ..., n-1]` for some `n`.
    /// Effectively, the row `i` of the new matrix will be the row `permutation[i]` of `self`.
    pub fn permute_rows(&self, permutation: &[usize]) -> Matrix {
        let mut values = Vec::<f64>::with_capacity(self.values.len());
        for p in permutation {
            values.extend(&self.values[p * self.n .. (p + 1) * self.n]);
        }
        Matrix { m: self.m, n: self.n, values }
    }
    /// Applies the given permutation to columns of `self`.
    /// 
    /// `permutation` should be a permutation of the vector `[0, ..., n-1]` for some `n`.
    /// Effectively, the column `i` of the new matrix will be the column `permutation[i]` of `self`.
    pub fn permute_columns(&self, permutation: &[usize]) -> Matrix {
        // To avoid cache thrashing (inevitable in a naive implementation), we first transpose `self`,
        // apply the permutation as row permutation to it, and finally transpose back.
        // Mathematically, this works because A*P = (P^t * A^t)^t.
        let self_t = self.transpose();
        let mut values = Vec::<f64>::with_capacity(self_t.values.len());
        for p in permutation {
            values.extend(&self_t.values[p * self_t.n .. (p + 1) * self_t.n]);
        }
        Matrix { m: self_t.m, n: self_t.n, values }.transpose()
    }

    /// Computes the LU decomposition of `self`.
    /// 
    /// Time complexity: 2/3 * n^3 + O(n^2)
    pub fn lu_decomposition(&self) -> Option<(Matrix, Matrix)> {
        // First, ensure the matrix is square.
        if self.m != self.n { return None; }
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

    /// Computes the full-pivot LU decomposition of `self`.
    ///
    /// Returns `Some((L, U, p, q))` such that `P * self * Q = L * U`, where
    /// `P` and `Q` are the permutation matrices encoded by the index vectors
    /// `p` and `q` (i.e. `P[i][p[i]] = 1` and `Q[q[j]][j] = 1`).
    ///
    /// Returns `None` if the matrix is not square. Notice that the full-pivot LU decomposition exists for any square matrix.
    pub fn lu_decomposition_full_pivot(&self) -> Option<(Matrix, Matrix, Vec<usize>, Vec<usize>)> {
        if self.m != self.n { return None; }
        let n = self.m;
        let mut a = self.clone(); // Working copy of `self` we modify in place
        let mut l = Matrix::identity(n);
        let mut u = Matrix::zeros(n, n);
        let mut p: Vec<usize> = (0..n).collect();
        let mut q: Vec<usize> = (0..n).collect();

        for i in 0..n {
            // Find the pivot, i.e. the `r, c` corresponding to the largest `|a[r][c]|` in the trailing submatrix
            let mut max_abs = 0.0f64;
            let mut pivot_row = i;
            let mut pivot_col = i;
            for r in i..n {
                for c in i..n {
                    let v = a.get(r, c).abs();
                    if v > max_abs {
                        max_abs = v;
                        pivot_row = r;
                        pivot_col = c;
                    }
                }
            }
            if max_abs == 0.0 {
                break; // Remaining submatrix is zero; decomposition is complete
            }

            // Bring the pivot row up to row `i` both in `a` and in `l` (for `l`, we only treat already-filled columns)
            if pivot_row != i {
                for c in 0..n {
                    a.values.swap(i * n + c, pivot_row * n + c);
                }
                for c in 0..i {
                    l.values.swap(i * n + c, pivot_row * n + c);
                }
                p.swap(i, pivot_row);
            }
            // Bring the pivot column left to column `i` both in `a` and `u`
            if pivot_col != i {
                for r in 0..n {
                    a.values.swap(r * n + i, r * n + pivot_col);
                }
                for r in 0..i {
                    u.values.swap(r * n + i, r * n + pivot_col);
                }
                q.swap(i, pivot_col);
            }

            // Fill row `i` of `u` and column `i` of `l`:
            // `U[i][j] = a[i][j]` for `j >= i` (current reduced row).
            for j in i..n {
                u.set(i, j, a.get(i, j));
            }
            // `L[k][i] = a[k][i] / pivot`, then eliminate below.
            let pivot = a.get(i, i); // N.b.: this must be non-zero at this point.
            for k in (i + 1)..n {
                l.set(k, i, a.get(k, i) / pivot);
                for j in i+1..n {
                    a.values[k * n + j] -= l.get(k, i) * a.get(i, j);
                }
            }
        }
        Some((l, u, p, q))
    }

    /// Returns the inverse of `self` assuming that `self` is an upper triangular matrix.
    /// 
    /// If `self` is singular or non-square, returns `None`.
    pub fn inv_for_upper_triangular(&self) -> Option<Matrix> {
        let n = self.n;
        if self.m != n || (0..n).any(|i| self.get(i, i) == 0.0) {return None;}
        let mut res = Matrix::zeros(n, n);
        for j in 0..n {
            res.set(j, j, 1.0 / self.get(j, j));
            for i in (0..j).rev() {
                res.set(i, j, -(i+1..n).map(|k| self.get(i, k) * res.get(k, j)).sum::<f64>() / self.get(i, i));
            }
        }
        Some(res)
    }
    /// Returns the inverse of `self` assuming that `self` is a lower triangular matrix.
    /// 
    /// If `self` is singular or non-square, returns `None`.
    pub fn inv_for_lower_triangular(&self) -> Option<Matrix> {
        let n = self.n;
        if self.m != n || (0..n).any(|i| self.get(i, i) == 0.0) {return None;}
        let mut res = Matrix::zeros(n, n);
        for j in 0..n {
            res.set(j, j, 1.0 / self.get(j, j));
            for i in j+1..n {
                res.set(i, j, -(0..i).map(|k| self.get(i, k) * res.get(k, j)).sum::<f64>() / self.get(i, i));
            }
        }
        Some(res)
    }

    /// Returns the adjugate matrix of `self`. Returns `None` if `self` is not square.
    /// 
    /// We decompose `A := self` into `PAQ = LDU` via the full-pivot LU decomposition where `L` and `U` only
    /// have `1`s on their respective diagonal. Then, `A = XDY` for `X = P^t L` and `Y = U Q^t`.
    /// 
    /// Then, `adj(A) = adj(Y) adj(D) adj(X) = det(P) det(Q) Q adj(U) adj(D) adj(L) P`, which can be computed in O(n^3).
    pub fn adj(&self) -> Option<Matrix> {
        let (l, mut u, pt, qt) = self.lu_decomposition_full_pivot()?;
        let n = self.n;
        // Currently, U doesn't necessarily have diagonal entries (1, ..., 1). We still have to extract D.
        let d = Matrix::diag(&(0..n).map(|i| u.get(i, i)).collect::<Vec<f64>>());
        for i in 0..n {
            if !utils::approx_eq(&d.get(i, i), &0.0) {
                for j in i..n {
                    u.values[i*n+j] /= d.get(i, i);
                }
            }
        }
        // Now, apply the formula from the documentation.
        let det_p = if utils::permutation_parity(&pt) {1.0} else {-1.0};
        let det_q = if utils::permutation_parity(&qt) {1.0} else {-1.0};
        let adj_d = d.adj_for_diagonal_matrix();
        &(det_p * det_q
        * &(&u.adj_for_upper_triangular_matrix().permute_rows(&utils::transpose_permutation(&qt))
        * &adj_d)?)
        * &l.transpose().adj_for_upper_triangular_matrix()
            .permute_rows(&utils::transpose_permutation(&pt))
            .transpose()
    }

    /// Returns the adjugate matrix of `self` assuming that `self` is a diagonal matrix `D`. Then, `adj(D)` is the diagonal matrix with
    /// entries `adj(D)_{i,i} = \prod_{j \neq i} D_{j,j}`.
    /// 
    /// Interestingly enough, this function runs in only O(n) of time (see implementation).
    pub fn adj_for_diagonal_matrix(&self) -> Matrix {
        // Compute the prefix products p_i := \prod_{k=0}^i D_{k,k} and the suffix products s_i := \prod_{k=i}^{n-1} D_{k,k}.
        let mut p = Vec::<f64>::with_capacity(self.n);
        p.push(self.get(0, 0)); // p_0 = D_{0,0}
        for i in 1..self.n {
            p.push(p[i-1] * self.get(i, i));
        }
        let mut s_reverse = Vec::<f64>::with_capacity(self.n);
        s_reverse.push(self.get(self.n-1, self.n-1)); // s_{n-1} = D_{n-1,n-1}
        for i in 1..self.n {
            s_reverse.push(s_reverse[i-1] * self.get(self.n - i - 1, self.n - i - 1));
        }
        // Now, adj(D)_{i,i} = \prod_{j \neq i} D_{j,j} = p_{i-1} * s_{i+1} = p_{i-1} * s_reverse_{n-i}
        let mut diag = Vec::<f64>::with_capacity(self.n);
        diag.push(s_reverse[self.n-2]);
        for i in 1..self.n-1 {
            diag.push(p[i-1] * s_reverse[self.n - i - 2]);
        }
        diag.push(p[self.n-2]);
        Matrix::diag(&diag)
    }
    
    /// Returns the adjugate matrix of `self` assuming that `self` is a square upper triangular matrix `U`.
    /// 
    /// Then, `adj(U)` is also an upper triangular matrix with non-trivial entries
    /// `adj(U)_{i,j} = (-1)^{i+j} (\prod_{p<i} U_{p,p}) (\prod_{q>j} U_{q,q}) det(H^{(i,j)})`
    /// where `H` is the `(j-i)x(j-i)` Hessenberg matrix given by (indexing from zero)
    /// `H^{(i,j)}_{k,l} = u_{i+k,i+1+l}`.
    /// 
    /// This function runs in `O(n^3)`.
    pub fn adj_for_upper_triangular_matrix(&self) -> Matrix {
        let n = self.n;
        // Compute the prefix products p_i := \prod_{k=0}^i U_{k,k} and the suffix products s_i := \prod_{k=i}^{n-1} U_{k,k}.
        let mut p = Vec::<f64>::with_capacity(n);
        p.push(self.get(0, 0)); // p_0 = D_{0,0}
        for i in 1..n {
            p.push(p[i-1] * self.get(i, i));
        }
        let mut s_reverse = Vec::<f64>::with_capacity(n);
        s_reverse.push(self.get(n-1, n-1)); // s_{n-1} = D_{n-1,n-1}
        for i in 1..n {
            s_reverse.push(s_reverse[i-1] * self.get(n - i - 1, n - i - 1));
        }

        // Notice `adj(U)` is an upper triangular matrix too
        let mut adj = Matrix::zeros(n, n);

        for i in 0..n {
            // Fix i.
            // To compute the determinant of H := H_{i,j}, we generally proceed as in `det_for_hessenberg_matrix`.
            // However, we can share the subdeterminants to reduce time complexity from O(n^4) to O(n^3).
            // Specifically, when indexing from 1, we'd have with d_r := det H_{i, i+r} (r=0,...,n-i) that
            // d_0 = 1, d_k = \sum_{r=1}^k (-1)^{k-r} U_{i+r-1,i+k} (\prod_{t=r}^{k-1} U_{i+t,i+t}) d_{r-1}.
            // Since indexing actually starts at zero, the code will have some indices shifted.
            // We first compute all D_i in order (total complexity: O(n^2)).
            let mut d = vec![0.0; n-i+1];
            d[0] = 1.0;
            for k in 1..n-i {
                let mut chain = 1.0;
                for r in (1..=k).rev() {
                    d[k] += (if (k-r) % 2 == 0 {1.0} else {-1.0}) * self.get(i+r-1, i+k) * chain * d[r-1];
                    if r > 1 {
                        chain *= self.get(i+r-1, i+r-1);
                    }
                }
            }
            for j in i..n {
                adj.set(i, j,
                    (if (i+j) % 2 == 0 {1.0} else {-1.0})
                    * (if i > 0 {p[i-1]} else {1.0})
                    * (if j < n-1 {s_reverse[n-j-2]} else {1.0})
                    * d[j-i]
                )
            }
        }
        adj
    }

    /// Returns the product of all diagonal entries of `self`.
    fn diag_product(&self) -> f64 {
        (0..self.m).fold(1.0, |acc, i| acc * self.get(i, i))
    }

    /// Returns the sum of all diagonal entries of `self`.
    pub fn tr(&self) -> f64 {
        (0..self.m).map(|i| self.get(i, i)).sum()
    }

    /// Returns the determinant of `self` via an LU-decomposition.
    /// 
    /// Runs in 2/3 * n^3 + O(n^2).
    pub fn det(&self) -> Option<f64> {
        if self.m != self.n { return None; }
        if let Some((l, u)) = self.lu_decomposition() {
            Some(l.diag_product() * u.diag_product())
        } else {
            // If no LU-decomposition exists, there exists some linear dependency between rows.
            // This immediately implies that the matrix is not invertible, that is, it has determinant zero.
            Some(0.0)
        }
    }

    /// Returns the determinant of `self` assuming that `self` is an `nxn` Hessenberg matrix.
    /// 
    /// This algorithm is based on the fact that (indexing from 1) with `d_k := det(H_{1:k, 1:k}` and `d_0 := 1`,
    /// we have the recursive formula `d_k = \sum_{r=0}^k (-1)^{k-r} H_{r,k} (\prod_{t=r}^{k-1} H_{t+1,t}) d_{r-1}`.
    /// 
    /// Runs in O(n^2).
    pub fn det_for_hessenberg_matrix(&self) -> f64 {
        let mut d = vec![0.0; self.n+1];
        d[0] = 1.0;
        for k in 1..=self.n {
            let mut chain = 1.0;
            for r in (1..=k).rev() {
                d[k] += (if (k-r) % 2 == 0 {1.0} else {-1.0}) * self.get(r-1, k-1) * chain * d[r-1];
                if r > 1 {
                    chain *= self.get(r-1, r-2);
                }
            }
        }
        d[self.n]
    }

    /// Computes the matrix `G` such that `G * self` rotates the plane spanned by the rows `i` and `j`
    /// precisely such that `A[j][col]` becomes zero.
    /// 
    /// Assumes that `self` is quadratic (if this function becomes public, this may be changed).
    fn givens_matrix(&self, i: usize, j: usize, col: usize) -> Matrix {
        let mut g = Matrix::identity(self.m);
        let x = (self.get(i, col).powi(2) + self.get(j, col).powi(2)).sqrt();
        g.set(i, i, self.get(i, col) / x);
        g.set(j, j, self.get(i, col) / x);
        g.set(i, j, self.get(j, col) / x);
        g.set(j, i, -self.get(j, col) / x);
        g
    }

    /// Computes the matrix `G` such that `G * self` rotates the plane spanned by the rows `i` and `j`
    /// precisely such that `A[j][col]` becomes zero.
    /// 
    /// Assumes that `self` is quadratic (if this function becomes public, this may be changed).
    fn upper_hessenberg(&self) -> (Matrix, Matrix) {
        let mut h = self.clone();
        let mut pg = Matrix::identity(self.m);
        if self.m > 1 {
            for col in 0..self.m-2 {
                for row in col+2..self.m {
                    let g = h.givens_matrix(col+1, row, col);
                    h = (&g * &h).unwrap().mul_with_transposed(&g).unwrap();
                    pg = (&g * &pg).unwrap();
                }
            }
        }
        (h, pg)
    }

    /// Computes the QR decomposition of `self`. Only works if `self` is a Hessenberg matrix.
    fn qr_decomposition_for_hessenberg_matrix(&self) -> (Matrix, Matrix) {
        let mut r = self.clone();
        let mut q = Matrix::identity(self.m);
        for i in 0..self.m-1 {
            let g = r.givens_matrix(i, i+1, i);
            q = (&g * &q).unwrap();
            r = (&g * &r).unwrap();
        }
        (q.transpose(), r)
    }
    /// Computes the QR decomposition of `self` in O(n^3) using Hessenberg matrices and Givens rotations.
    /// 
    /// Returns `None` if `self` is not quadratic. Otherwise, returns `(eigenvalues, q, r)`.
    pub fn qr_decomposition(&self) -> Option<(Vec<f64>, Matrix, Matrix)> {
        if self.m != self.n {
            return None;
        }
        let a = self.clone();
        let mut u_k = Matrix::identity(self.m);
        let (mut h_k, pg) = a.upper_hessenberg();
        for _ in 0..100 {
            let (inverse_pg, r_k) = h_k.qr_decomposition_for_hessenberg_matrix();
            h_k = (&r_k * &inverse_pg).unwrap();
            u_k = (&u_k * &inverse_pg).unwrap();
        }
        u_k = (&pg.transpose() * &u_k).unwrap();
        Some(((0..self.m).map(|i| h_k.get(i, i)).collect(), h_k, u_k))
    }

    /// Returns `self^t * self`.
    pub fn gram_matrix(&self) -> Matrix {
        let mut res = Matrix::zeros(self.n, self.n);
        for i in 0..self.n {
            for j in 0..=i {
                let value = (0..self.n)
                    .map(|k| self.get(k, i) * self.get(k, j))
                    .sum();
                res.set(i, j, value);
                if i != j {
                    res.set(j, i, value);
                }
            }
        }
        res
    }

    /// Approximates the operator norm of `self` induced by the `p`-norm up to an error of at most `tolerance`.
    /// 
    /// This implementation is based on the [article](https://link.springer.com/article/10.1007/BF01396242)
    /// "Estimating the matrix p-norm" by Nicholas Highham. For explanations, see the article.
    /// 
    /// `p` must be at least one and may be `f64::INFINITY`. The matrix `self` does not have any constraints. `tolerance` is recommended to be `1e-10`.
    /// 
    /// This method does not treat the cases `p=1.0`, `p=2.0` and `p=f64::INFINITY` separately. For those, call `self.norm`.
    fn pnorm(&self, p: f64, tolerance: f64) -> Result<f64, String> {
        // All `unwrap`s below are safe because the dimensions of the operands fit.
        let q = if p == 1.0 {
            f64::INFINITY
        } else {
            1.0 / (1.0 - 1.0 / p)
        };
        let samples = 9; // Could theoretically be increased or reduced (until 2), but the default is 9.
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
                    cs /= cs.norm(&VectorNorm::P(p));
                    let w_cs = (&(cs[0] * &col_k) + &(cs[1] * &y)).unwrap();
                    let f = w_cs.norm(&VectorNorm::P(p));
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
        let mut est = y.norm(&VectorNorm::P(p));
        for iter in 1usize.. {
            y = (self * &x).unwrap();
            let eo = est;
            est = y.norm(&VectorNorm::P(p));
            let dv_y = y.dual(p);
            // Slightly hacky; instead of `self^t * dv_y`, we write `dv_y * self`, which I implemented as `(dv_y^t * self)^t` for convenience
            // (the other operation would be undefined anyway), which in turn is mathematically exactly `self^t * dv_y`.
            let z = (&dv_y * self).unwrap();
            let z_q_norm = z.norm(&VectorNorm::P(q));
            if iter > 1 && (z_q_norm < (&z * &x).unwrap() || (est - eo).abs() <= tolerance * est) {
                break;
            }
            x = z.dual(q);
        }
        Ok(est)
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
                match self.gram_matrix().qr_decomposition() {
                    Some((eigenvalues, ..)) => Ok(utils::max_abs(eigenvalues.iter()).sqrt()),
                    None => Err(format!("Matrix must be quadratic (got {}x{}).", self.m, self.n))
                }
            }
            MatrixNorm::P(p) if *p >= 1.0 => {
                self.pnorm(*p, 1e-10)
            }
            MatrixNorm::P(other) => Err(format!("Parameter `p` must be at least 1 (got {other}).")),
            MatrixNorm::Frobenius => Ok(self.values.iter().map(|x| x.powi(2)).sum::<f64>().sqrt())
        }
    }
}