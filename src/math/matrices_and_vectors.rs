//! The implementations matrices and vectors below are tailored to the use within a calculator,
//! where flexibility to define variables as desired is more important than performance.
//! 
//! However, this makes computation slightly longer, so this file shouldn't be
//! used without modification for intensive computations. Nevertheless, algorithms
//! are written with view to larger matrices too, including optimization strategies
//! like tiling (for better cache locality) and parallelization.

use std::fmt;

mod adj;
mod lu;
mod matmul;
mod norms;
mod ops;
mod qr;
mod transposition;

pub use norms::{MatrixNorm, VectorNorm};

use crate::math::utils;


#[derive(Clone, PartialEq)]
pub struct Vector {
    pub values: Vec<f64>
}
#[derive(Clone, PartialEq)]
pub struct Matrix {
    m: usize,
    n: usize,
    values: Vec<f64>
}

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

impl Default for Vector {
    fn default() -> Vector {
        Vector { values: vec![0.0] }
    }
}


impl Matrix {
    /// For the sake of efficiency, the responsibility of bound-checking is delegated to the caller.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.values[i * self.n + j]
    }

    /// For the sake of efficiency, the responsibility of bound-checking is delegated to the caller.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: f64) {
        self.values[i * self.n + j] = value;
    }

    pub fn m(&self) -> usize {self.m}
    pub fn n(&self) -> usize {self.n}
}


impl Vector {
    pub fn zeros(n: usize) -> Vector {
        Vector { values: vec![0.0; n] }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Replaces every component `x` of the vector by `f(x)`.
    pub fn transform_in_place<F>(&mut self, f: F) where F: Fn(f64) -> f64 {
        for x in self.values.iter_mut() {
            *x = f(*x);
        }
    }
    /// Maps every component `x` of `self` to `f(x)`, returning a new vector.
    pub fn transform<F>(&self, f: F) -> Vector where F: Fn(f64) -> f64 {
        Vector{values: self.values.iter().map(|x| f(*x)).collect()}
    }
    /// Creates a new vector by applying f to every element of `self` while consuming `self`.
    pub fn into_new<F>(self, f: F) -> Vector where F: Fn(f64) -> f64 {
        Vector{values: self.values.into_iter().map(f).collect()}
    }

    /// Contiguous slice dot product. Does not check if the dimensions of `a, b` match.
    #[inline]
    fn unchecked_dot(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    }
    /// Contiguous slice dot product. Does not check if the dimensions of `a, b` match.
    #[inline]
    fn unchecked_dot_iter(a: std::iter::Map<std::ops::Range<usize>, impl FnMut(usize) -> f64>, b: &[f64]) -> f64 {
        a.zip(b.iter()).map(|(x, y)| x * y).sum()
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
    pub fn transform_in_place<F>(&mut self, f: F) where F: Fn(f64) -> f64 {
        for x in self.values.iter_mut() {
            *x = f(*x);
        }
    }
    /// Maps every component `x` of `self` to `f(x)`, returning a new vector.
    pub fn transform<F>(&self, f: F) -> Matrix where F: Fn(f64) -> f64 {
        Matrix{m: self.m, n: self.n, values: self.values.iter().map(|x| f(*x)).collect()}
    }
    /// Creates a new matrix by applying f to every element of `self` while consuming `self`.
    pub fn into_new<F>(self, f: F) -> Matrix where F: Fn(f64) -> f64 {
        Matrix{m: self.m, n: self.n, values: self.values.into_iter().map(f).collect()}
    }

    pub fn row(&self, i: usize) -> Vector {
        Vector { values: self.values[(i*self.n)..(i+1)*self.n].to_vec() }
    }
    #[inline]
    pub fn row_slice(&self, i: usize) -> &[f64] {
        &self.values[i * self.n .. (i+1) * self.n]
    }
    pub fn col(&self, j: usize) -> Vector {
        // Impossible to do in a cache-friendly way without changing the method by which `Matrix` stores its values.
        Vector { values: utils::col(&self.values, j, self.m, self.n).collect() }
    }

    /// Applies the given permutation to the rows of `self`.
    /// 
    /// `permutation` should be a permutation of the vector `[0, ..., self.m-1]`.
    /// Effectively, the row `i` of the new matrix will be the row `permutation[i]` of `self`.
    /// 
    /// Returns `None` if the permutation's size doesn't match the matrices size.
    pub fn permute_rows(&self, permutation: &[usize]) -> Option<Matrix> {
        if permutation.len() != self.m || permutation.iter().any(|p| *p >= self.m) {return None;}
        let mut values = Vec::<f64>::with_capacity(self.values.len());
        for p in permutation {
            // Writes are sequential and reads are sequential within each row. By the nature
            // of permutations, this is optimal in the sense that even with tiling, you may
            // have to jump between rows either when reading or when writing.
            values.extend(&self.values[p * self.n .. (p + 1) * self.n]);
        }
        Some(Matrix { m: self.m, n: self.n, values })
    }
    /// Applies the given permutation to columns of `self`.
    /// 
    /// `permutation` should be a permutation of the vector `[0, ..., self.n-1]`.
    /// Effectively, the column `j` of the new matrix will be the column `permutation[j]` of `self`.
    /// 
    /// Returns `None` if the permutation's size doesn't match the matrices size.
    pub fn permute_columns(&self, permutation: &[usize]) -> Option<Matrix> {
        if permutation.len() != self.n || permutation.iter().any(|p| *p >= self.n) {return None;}
        let mut values = Vec::<f64>::with_capacity(self.values.len());
        for i in 0..self.m {
            // As for row permutations, writes are sequential and reads are sequential within each row.
            permutation.iter().for_each(|p| values.push(self.get(i, *p)));
        }
        Some(Matrix { m: self.m, n: self.n, values })
    }
    

    /// Returns the inverse of `self` in O(n^3).
    pub fn inv(&self) -> Option<Matrix> {
        if let Some((p, l, u)) = self.plu_decomposition() {
            (&u.inv_for_upper_triangular()? * &l.inv_for_lower_triangular()?)?.permute_columns(&p)
        } else {None}
    }

    /// Returns the inverse of `self` assuming that `self` is an upper triangular matrix.
    /// 
    /// If `self` is singular or non-square, returns `None`.
    pub fn inv_for_upper_triangular(&self) -> Option<Matrix> {
        let n = self.n;
        if self.m != n || (0..n).any(|i| self.get(i, i) == 0.0) {return None;}
        // For cache locality, we build the transposed inverse first
        let mut inv_t = vec![0.0; n*n];
        for j in 0..n {
            inv_t[j*n + j] = 1.0 / self.get(j, j);
            for i in (0..j).rev() {
                inv_t[j*n + i] = -Vector::unchecked_dot(&self.values[i*n + i+1 .. i*n + j+1], &inv_t[j*n + i+1 .. j*n + j+1]) / self.get(i, i);
                //             = -sum_{k=i+1}^{n-1} self[i, k] * inv_t[j, k]                                                   / self[i, i]
                //             = -sum_{k=i+1}^j self[i, k] * inv_t[j, k]                                                       / self[i, i]
                //               since inv_t[j, k] = 0 for k > j
            }
        }
        Some(Matrix::from(n, n, inv_t).transpose())
    }
    /// Returns the inverse of `self` assuming that `self` is a lower triangular matrix.
    /// 
    /// If `self` is singular or non-square, returns `None`.
    pub fn inv_for_lower_triangular(&self) -> Option<Matrix> {
        let n = self.n;
        if self.m != n || (0..n).any(|i| self.get(i, i) == 0.0) {return None;}
        // For cache locality, we build the transposed inverse first
        let mut inv_t = Vec::<f64>::with_capacity(n * n);
        for j in 0..n {
            inv_t.extend(std::iter::repeat_n(0.0, j));
            inv_t.push(1.0 / self.get(j, j));
            for i in j+1..n {
                inv_t.push(-Vector::unchecked_dot(&self.values[i*n + j .. i*n + i], &inv_t[j*n + j..]) / self.get(i, i));
                //       = -sum_{k=0}^{i-1} self[i, k] * inv_t[j, k]                                        / self[i, i]
                //       = -sum_{k=j}^{i-1} self[i, k] * inv_t[j, k]                                        / self[i, i]
                //         since inv_t[j, k] = 0 for k < j
            }
        }
        Some(Matrix::from(n, n, inv_t).transpose())
    }


    /// Returns the product of all diagonal entries of `self`.
    fn diag_product(&self) -> f64 {
        (0..self.m).fold(1.0, |acc, i| acc * self.get(i, i))
    }

    /// Returns the sum of all diagonal entries of `self`. Returns `Err` if the matrix isn't square.
    pub fn tr(&self) -> Result<f64, String> {
        if self.m != self.n {
            return Err("Can't compute the trace of a non-square matrix.".to_string());
        }
        Ok((0..self.m).map(|i| self.get(i, i)).sum())
    }


    /// Returns the determinant of `self` via a full-pivot LU-decomposition.
    /// 
    /// Runs in 2/3 * n^3 + O(n^2).
    pub fn det(&self) -> Option<f64> {
        self.lu_decomposition_full_pivot().map(
            |(l, u, p, q)|
            // We now have `self = p^T * L * U * q^T` so
            // det(self) = det(p) det(L) det(U) det(q)
            if utils::permutation_parity(&p) {1.0} else {-1.0}
            * if utils::permutation_parity(&q) {1.0} else {-1.0}
            * l.diag_product()
            * u.diag_product()
        )
    }

    /// Returns the determinant of `self` requiring that `self` is an `nxn` Hessenberg matrix (`None` if `self` isn't square).
    /// 
    /// This algorithm is based on the fact that (indexing from 1) with `d_k := det(H_{1:k, 1:k}` and `d_0 := 1`,
    /// we have the recursive formula `d_k = \sum_{r=0}^k (-1)^{k-r} H_{r,k} (\prod_{t=r}^{k-1} H_{t+1,t}) d_{r-1}`.
    /// 
    /// Runs in O(n^2).
    pub fn det_for_hessenberg_matrix(&self) -> Option<f64> {
        if self.m != self.n { return None; }
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
        Some(d[self.n])
    }
}