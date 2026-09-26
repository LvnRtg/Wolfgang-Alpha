//! Contains definition and basic member functions for the class `Matrix`.
//! 
//! The implementation of all further operations is delegated to submodules.
//! 
//! The implementations matrices and vectors below are tailored to the use within a calculator,
//! where flexibility to define variables as desired is more important than performance.
//! 
//! However, this makes computation slightly longer, so this file shouldn't be
//! used without modification for intensive computations. Nevertheless, algorithms
//! are written with view to larger matrices too, including optimization strategies
//! like tiling (for better cache locality) and parallelization.

use num_traits::{One, Zero};
use std::fmt;
use std::fmt::{Debug, Display};

use crate::math::{Complex, Vector};
use crate::math::objects::MatrixType;
use crate::math::traits::*;
use crate::math::utils;

mod adj;
mod lu;
mod matmul;
mod norms;
mod ops;
mod qr;
mod schur;
mod square_root;
mod sylvester;
mod tests;
mod transposition;
mod views;

pub use matmul::mul_views_into;
pub use norms::MatrixNorm;
pub use views::{MatrixView, MatrixViewMut};


pub struct Matrix<T> {
    m: usize,
    n: usize,
    values: Vec<T>
}

impl<T: Clone> Clone for Matrix<T> {
    fn clone(&self) -> Matrix<T> {
        Matrix::<T> {
            m: self.m,
            n: self.n,
            values: self.values.clone()
        }
    }
}
impl<T: PartialEq> PartialEq for Matrix<T> {
    fn eq(&self, other: &Self) -> bool {
        self.m == other.m && self.n == other.n && self.values.iter().zip(other.values.iter()).all(|(x, y)| x == y)
    }
}
impl<T: ToString> Display for Matrix<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "[{}]",
            (0..self.m).map(
                |i| self.values[i * self.n .. (i+1) * self.n].iter().map(
                    |x| x.to_string()
                ).collect::<Vec<_>>().join(", ")
            ).collect::<Vec<_>>().join("; "))
    }
}
impl<T: Debug> Debug for Matrix<T>  {
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

impl<T> Matrix<T> {
    /// Encapsulated in order to keep the field `values` private.
    pub fn from(m: usize, n: usize, values: Vec<T>) -> Matrix<T> {
        Matrix{m, n, values}
    }

    #[inline]
    pub fn get_ref(&self, i: usize, j: usize) -> &T {
        &self.values[i * self.n + j]
    }

    /// For the sake of efficiency, the responsibility of bound-checking is delegated to the caller.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: T) {
        self.values[i * self.n + j] = value;
    }

    #[inline] pub fn m(&self) -> usize {self.m}
    #[inline] pub fn n(&self) -> usize {self.n}
    #[inline] pub fn total_size(&self) -> usize {self.values.len()} // Can spare us an unnecessary multiplication

    /// Returns `self.values.iter()`. Encapsulated in order to keep the field `values` private.
    pub fn iter(&self) -> impl Iterator<Item=&T> {
        self.values.iter()
    }

    /// Creates a new matrix by applying f to every element of `self` while consuming `self`.
    pub fn into_new<U, F>(self, f: F) -> Matrix<U> where F: Fn(T) -> U {
        Matrix{m: self.m, n: self.n, values: self.values.into_iter().map(f).collect()}
    }

    #[inline]
    pub fn row_slice(&self, i: usize) -> &[T] {
        &self.values[i * self.n .. (i+1) * self.n]
    }
}


impl<T: Copy> Matrix<T> {
    /// For the sake of efficiency, the responsibility of bound-checking is delegated to the caller.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T {
        self.values[i * self.n + j]
    }

    /// Maps every component `x` of `self` to `f(x)`, returning a new matrix.
    #[inline]
    pub fn transform<U, F>(&self, f: F) -> Matrix<U> where F: Fn(T) -> U {
        Matrix{m: self.m, n: self.n, values: self.values.iter().map(|x| f(*x)).collect()}
    }
    /// Replaces every component `x` of the matrix by `f(x)`.
    pub fn transform_in_place<F>(&mut self, f: F) where F: Fn(T) -> T {
        for x in self.values.iter_mut() {
            *x = f(*x);
        }
    }

    pub fn row(&self, i: usize) -> Vector<T> {
        Vector::<T> { values: self.values[(i*self.n)..(i+1)*self.n].to_vec() }
    }
    pub fn col(&self, j: usize) -> Vector<T> {
        // Impossible to do in a cache-friendly way without changing the method by which `Matrix` stores its values.
        Vector::<T> { values: utils::col(&self.values, j, self.m, self.n).collect() }
    }
}

impl<T: Copy + std::iter::Sum<T>> Matrix<T> {
    /// Returns the sum of all diagonal entries of `self`. Returns `Err` if the matrix isn't square.
    pub fn tr(&self) -> Result<T, String> {
        if self.m != self.n {
            return Err("Can't compute the trace of a non-square matrix.".to_string());
        }
        Ok((0..self.m).map(|i| self.get(i, i)).sum())
    }
}

impl<T: Real> Matrix<T> {
    #[inline]
    pub fn to_complex(self) -> Matrix<Complex<T>> {
        self.transform(|x| x.to_complex())
    }
}

impl<T: Real> Matrix<Complex<T>> {
    pub fn conjugate(self) -> Matrix<Complex<T>> {
        self.transform(|z| z.conjugate())
    }
}

impl<T: Scalar> Matrix<T> {
    pub fn zeros(m: usize, n: usize) -> Matrix<T> {
        Matrix {
            m, n,
            values: vec![T::zero(); m*n]
        }
    }
    
    pub fn identity(n: usize) -> Matrix<T> {
        if n == 0 {return Matrix{m: 0, n: 0, values: Vec::<T>::new()};}
        let mut values = Vec::<T>::with_capacity(n*n);
        values.push(T::one());
        for _ in 0..n-1 {
            values.extend(std::iter::repeat_n(T::zero(), n));
            values.push(T::one());
        }
        Matrix{m: n, n, values}
    }

    /// Constructs a diagonal matrix with diagonal entries `diag_values`.
    pub fn diag(diag_values: &[T]) -> Matrix<T> {
        let n = diag_values.len();
        if n == 0 {return Matrix{m: 0, n: 0, values: Vec::<T>::new()};}
        let mut values = Vec::<T>::with_capacity(n * n);
        values.push(diag_values[0]);
        for v in diag_values.iter().skip(1) {
            values.extend(std::iter::repeat_n(T::zero(), n));
            values.push(*v);
        }
        Matrix{m: n, n, values}
    }

    pub fn approx_eq(&self, other: &Matrix<T>) -> bool {
        self.m == other.m && self.n == other.n && self.iter().zip(other.iter()).all(|(&x, &y)| utils::approx_eq(x, y))
    }
    
    /// Returns the inverse of `self` in O(n^3). Returns `None` if the inverse doesn't exist.
    pub fn inv(&self) -> Option<Matrix<T>> {
        self.plu_decomposition().and_then(
            |(p, l, u)|
            p.apply_to_matrix_columns(&(u.inv_for_upper_triangular()?.mul(l.inv_for_lower_triangular()?))?)
        )
    }

    /// Returns the inverse of `self` assuming that `self` is an upper triangular matrix.
    /// 
    /// If `self` is singular or non-square, returns `None`.
    pub fn inv_for_upper_triangular(&self) -> Option<Matrix<T>> {
        let n = self.n;
        if self.m != n || (0..n).any(|i| self.get(i, i).is_zero()) {return None;}
        // For cache locality, we build the transposed inverse first
        let mut inv_t = vec![T::zero(); n*n];
        for j in 0..n {
            inv_t[j*n + j] = T::one().div(self.get(j, j));
            for i in (0..j).rev() {
                inv_t[j*n + i] = utils::unchecked_dot(&self.values[i*n + i+1 .. i*n + j+1], &inv_t[j*n + i+1 .. j*n + j+1]).neg().div(self.get(i, i));
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
    pub fn inv_for_lower_triangular(&self) -> Option<Matrix<T>> {
        let n = self.n;
        if self.m != n || (0..n).any(|i| self.get(i, i).is_zero()) {return None;}
        // For cache locality, we build the transposed inverse first
        let mut inv_t = Vec::<T>::with_capacity(n * n);
        for j in 0..n {
            inv_t.extend(std::iter::repeat_n(T::zero(), j));
            inv_t.push(T::one().div(self.get(j, j)));
            for i in j+1..n {
                inv_t.push(utils::unchecked_dot(&self.values[i*n + j .. i*n + i], &inv_t[j*n + j..]).neg().div(self.get(i, i)));
                //       = -sum_{k=0}^{i-1} self[i, k] * inv_t[j, k]                                        / self[i, i]
                //       = -sum_{k=j}^{i-1} self[i, k] * inv_t[j, k]                                        / self[i, i]
                //         since inv_t[j, k] = 0 for k < j
            }
        }
        Some(Matrix::from(n, n, inv_t).transpose())
    }
    
    /// Returns the product of all diagonal entries of `self`.
    fn diag_product(&self) -> T {
        (0..self.m).fold(T::one(), |acc, i| acc.mul(self.get(i, i)))
    }
    
    /// Returns the determinant of `self` via a full-pivot LU-decomposition.
    /// 
    /// Runs in 2/3 * n^3 + O(n^2).
    pub fn det(&self) -> Option<T> {
        self.lu_decomposition_full_pivot().map(
            |(l, u, p, q)|
            // We now have `self = p^T * L * U * q^T` so
            // det(self) = det(p) det(L) det(U) det(q)
            if p.parity() {T::one()} else {T::one().neg()}
            .mul(if q.parity() {T::one()} else {T::one().neg()})
            .mul(l.diag_product())
            .mul(u.diag_product())
        )
    }
}

impl<T> Matrix<T>
where T:
    Copy
    + One
    + Zero
    + Neg<Output=T>
    + AddAssign<T>
    + MulAssign<T>
{
    /// Returns the determinant of `self` requiring that `self` is an `nxn` Hessenberg matrix (`None` if `self` isn't square).
    /// 
    /// This algorithm is based on the fact that (indexing from 1) with `d_k := det(H_{1:k, 1:k}` and `d_0 := 1`,
    /// we have the recursive formula `d_k = \sum_{r=0}^k (-1)^{k-r} H_{r,k} (\prod_{t=r}^{k-1} H_{t+1,t}) d_{r-1}`.
    /// 
    /// Runs in O(n^2).
    pub fn det_for_hessenberg_matrix(&self) -> Option<T> {
        if self.m != self.n { return None; }
        let mut d = vec![T::zero(); self.n+1];
        d[0] = T::one();
        for k in 1..=self.n {
            let mut chain = T::one();
            for r in (1..=k).rev() {
                let local_storage = d[r-1];
                d[k].add_assign(
                    if (k-r) % 2 == 0 {T::one()} else {T::one().neg()}
                    .mul(self.get(r-1, k-1))
                    .mul(chain)
                    .mul(local_storage)
                );
                if r > 1 {
                    chain.mul_assign(self.get(r-1, r-2));
                }
            }
        }
        Some(d[self.n])
    }
}

impl Matrix<Complex<f64>> {
    pub fn try_to_real(self) -> MatrixType {
        let mut reals = Vec::new();
        for z in self.values.iter() {
            if utils::approx_eq(z.imag, 0.0) {
                reals.push(z.real);
            } else {
                return MatrixType::Complex(self)
            }
        }
        MatrixType::Real(Matrix::from(self.m, self.n, reals))
    }
}