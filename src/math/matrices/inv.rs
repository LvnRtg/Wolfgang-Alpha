//! Implements functions related to the inverse of a matrix.

use super::*;

impl<T: Scalar> Matrix<T> {
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
}