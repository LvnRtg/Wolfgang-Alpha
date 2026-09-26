//! Implements matrix multiplication and adjacent operations (e.g. raising a matrix to a power `n \in \N`)
//! using tiling and parallelization. As for transpositions, we provide a simpler method for small matrices.

use rayon::prelude::*;

use crate::math::{BLOCK_SIZE, utils};
use crate::math::traits::{Mul, Scalar};
use super::*;


/// Tiling will only be used if the dimension exceeds this threshold, i.e. `max(m, n) >= TILING_THRESHOLD`.
const TILING_THRESHOLD: usize = 256;


/// Computes `self * rhs` in the natural way using parallelization.
/// 
/// This method is used when matrices are small enough that pairs `self.row(i), rhs.row(i)`
/// fit comfortably in the cache without explicit blocking.
fn mul_simple_parallel<T, U, V>(
    a: MatrixView<T>, b_t: MatrixView<U>, mut out: MatrixViewMut<V>,
) where
    T: Copy + Sync + Mul<U, Output = V>,
    U: Copy + Sync,
    V: Send + std::iter::Sum<V>,
{
    out.par_rows_mut().enumerate().for_each(|(i, out_row)| {
        let a_row = a.row_slice(i);
        for (j, o) in out_row.iter_mut().enumerate() {
            *o = utils::unchecked_dot(a_row, b_t.row_slice(j));
        }
    });
}

/// Computes `self * rhs` using parallelization and tiling: the output is
/// processed in tiles so that the working set of `self` rows and `rhs_t`
/// rows involved in a tile stays resident in cache across the inner iterations,
/// cutting down on repeated DRAM traffic for `rhs_t`.
/// 
/// This method is used for large matrices.
fn mul_tiled_parallel<T, U, V>(
    a: MatrixView<T>, b_t: MatrixView<U>, mut out: MatrixViewMut<V>,
) where
    T: Copy + Sync + Mul<U, Output = V>,
    U: Copy + Sync,
    V: Send + Sync + std::iter::Sum<V>,
{
    let cols = out.cols();
    let block_rows = BLOCK_SIZE.min(a.rows()).max(1);
    out.par_rows_mut()
    .chunks(block_rows)
    .enumerate()
    .for_each(|(blk, mut out_rows)| {
        let i_start = blk * block_rows;
        let rows_in_block = out_rows.len();
        for jj in (0..cols).step_by(BLOCK_SIZE) {
            let j_end = (jj + BLOCK_SIZE).min(cols);
            for bi in 0..rows_in_block {
                let a_row = a.row_slice(i_start + bi);
                let out_row = &mut out_rows[bi];
                for j in jj..j_end {
                    out_row[j] = utils::unchecked_dot(a_row, b_t.row_slice(j));
                }
            }
        }
    });
}

/// Returns None in case the dimensions mismatch.
pub fn mul_views_into<T, U, V>(a: MatrixView<T>, b: MatrixView<U>, out: MatrixViewMut<V>) -> Option<()>
where
    T: Scalar + Mul<U, Output = V>,
    U: Scalar,
    V: Scalar
{
    if a.cols() != b.rows() || out.rows() != a.rows() || out.cols() != b.cols() { return None; }
    if a.rows() == 0 || a.cols() == 0 || b.cols() == 0 { return Some(()); }
    let b_t = b.transpose();
    let bt = b_t.view();
    if a.rows().max(a.cols()).max(b.cols()) >= TILING_THRESHOLD {
        mul_tiled_parallel(a, bt, out);
    } else {
        mul_simple_parallel(a, bt, out);
    }
    Some(())
}
/// Returns None in case the dimensions mismatch.
impl<'a, 'b, T, U, V> Mul<MatrixView<'a, U>> for MatrixView<'b, T>
where
    T: Scalar + Mul<U, Output = V>,
    U: Scalar,
    V: Scalar
{
    type Output = Option<Matrix<V>>;
    fn mul(self, rhs: MatrixView<U>) -> Self::Output {
        if self.cols() != rhs.rows() { return None; }
        if self.rows() == 0 || self.cols() == 0 || rhs.cols() == 0 { return Some(Matrix::zeros(self.rows(), rhs.cols())); }
        let rhs_t = rhs.transpose();
        let rhs_t_view = rhs_t.view();
        let mut res = Matrix::zeros(self.rows(), rhs.cols());
        let out = res.view_mut();
        if self.rows().max(self.cols()).max(rhs.cols()) >= TILING_THRESHOLD {
            mul_tiled_parallel(self, rhs_t_view, out);
        } else {
            mul_simple_parallel(self, rhs_t_view, out);
        }
        Some(res)
    }
}


impl<T: Scalar> Matrix<T> {
    /// Returns `self^n`.
    pub fn pow(&self, n: u64) -> Option<Matrix<T>> {
        if self.m != self.n {return None;}
        let mut result = Matrix::identity(self.m);
        let mut base = self.clone();
        let mut remaining = n;
        while remaining > 0 {
            if remaining % 2 == 1 {
                result = (&result).mul(&base).unwrap(); // `unwrap` is safe since all matrices are quadratic and of the same size
            }
            remaining /= 2;
            if remaining > 0 {
                base = (&base).mul(&base).unwrap();
            }
        }
        Some(result)
    }

    /// Returns `self^t * self`.
    pub fn gram_matrix(&self) -> Matrix<T> {
        let mut values = vec![T::zero(); self.n * self.n];
        for k in 0..self.m {
            for i in 0..self.n {
                for j in 0..=i {
                    let value = self.get(k, i).mul(self.get(k, j));
                    values[i * self.n + j].add_assign(value);
                    if i != j {
                        values[j * self.n + i].add_assign(value);
                    }
                }
            }
        }
        Matrix { m: self.n, n: self.n, values }
    }
    
    /// Multiplies `self` with `rhs^t`. Returns `None` if the dimensions don't match.
    pub fn mul_with_transposed<U, V>(&self, rhs: &Matrix<U>) -> Option<Matrix<V>>
    where
        T: Copy + Mul<U, Output=V>,
        U: Copy,
        V: std::iter::Sum<V>
    {
        if self.n != rhs.n {
            None
        } else {
            let mut values = Vec::<V>::with_capacity(self.m * rhs.m);
            for i in 0..self.m {
                for j in 0..rhs.m {
                    values.push(utils::unchecked_dot(self.row_slice(i), rhs.row_slice(j)))
                }
            }
            Some(Matrix {
                m: self.m,
                n: rhs.m,
                values,
            })
        }
    }
}


impl<T, U, V> Mul<&Matrix<U>> for &Matrix<T>
where
    T: Scalar + Mul<U, Output=V>,
    U: Scalar,
    V: Scalar
{
    type Output = Option<Matrix<V>>;
    fn mul(self, rhs: &Matrix<U>) -> Self::Output {
        let (m, n) = (self.m, rhs.n);
        if self.n != rhs.m { return None; }
        let mut out = Matrix::zeros(m, n);
        mul_views_into(self.view(), rhs.view(), out.view_mut())?;
        Some(out)
    }
}
impl<T, U, V> Mul<&Matrix<U>> for Matrix<T>
where
    T: Scalar + Mul<U, Output=V>,
    U: Scalar,
    V: Scalar
{
    type Output = Option<Matrix<V>>;
    fn mul(self, rhs: &Matrix<U>) -> Self::Output {
        let (m, n) = (self.m, rhs.n);
        if self.n != rhs.m { return None; }
        let mut out = Matrix::zeros(m, n);
        mul_views_into(self.view(), rhs.view(), out.view_mut())?;
        Some(out)
    }
}
impl<T, U, V> Mul<Matrix<U>> for &Matrix<T>
where
    T: Scalar + Mul<U, Output=V>,
    U: Scalar,
    V: Scalar
{
    type Output = Option<Matrix<V>>;
    fn mul(self, rhs: Matrix<U>) -> Self::Output {
        let (m, n) = (self.m, rhs.n);
        if self.n != rhs.m { return None; }
        let mut out = Matrix::zeros(m, n);
        mul_views_into(self.view(), rhs.view(), out.view_mut())?;
        Some(out)
    }
}
impl<T, U, V> Mul<Matrix<U>> for Matrix<T>
where
    T: Scalar + Mul<U, Output=V>,
    U: Scalar,
    V: Scalar
{
    type Output = Option<Matrix<V>>;
    fn mul(self, rhs: Matrix<U>) -> Self::Output {
        let (m, n) = (self.m, rhs.n);
        if self.n != rhs.m { return None; }
        let mut out = Matrix::zeros(m, n);
        mul_views_into(self.view(), rhs.view(), out.view_mut())?;
        Some(out)
    }
}