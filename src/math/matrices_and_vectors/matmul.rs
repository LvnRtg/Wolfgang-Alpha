//! Implements matrix multiplication and adjacent operations (e.g. raising a matrix to a power `n \in \N`)
//! using tiling and parallelization. As for transpositions, we provide a simpler method for small matrices.

use std::ops;
use rayon::prelude::*;

use crate::math::BLOCK_SIZE;
use crate::math::{Matrix, Vector};

/// Tiling will only be used if the dimension exceeds this threshold, i.e. `max(m, n) >= TILING_THRESHOLD`.
const TILING_THRESHOLD: usize = 256;

/// Returns None in case the dimensions mismatch.
impl ops::Mul<&Matrix> for &Matrix {
    type Output = Option<Matrix>;
    fn mul(self, rhs: &Matrix) -> Self::Output {
        if self.n != rhs.m {
            return None;
        }
        let rhs_t = rhs.transpose(); // Improves cache locality and is only O(n²)
        let m = self.m;
        let n = rhs.n;
        let mut values = vec![0.0_f64; m * n];
        if m.max(n).max(self.n) >= TILING_THRESHOLD {
            self.mul_tiled_parallel(&rhs_t, &mut values);
        } else {
            self.mul_simple_parallel(&rhs_t, &mut values);
        }
        Some(Matrix { m, n, values })
    }
}

impl Matrix {
    /// Returns `self^n`.
    pub fn pow(&self, n: u64) -> Option<Matrix> {
        if self.m != self.n {return None;}
        let mut result = Matrix::identity(self.m);
        let mut base = self.clone();
        let mut remaining = n;
        while remaining > 0 {
            if remaining % 2 == 1 {
                result = (&result * &base).unwrap(); // `unwrap` is safe since all matrices are quadratic and of the same size
            }
            remaining /= 2;
            if remaining > 0 {
                base = (&base * &base).unwrap();
            }
        }
        Some(result)
    }

    /// Returns `self^t * self`.
    pub fn gram_matrix(&self) -> Matrix {
        let mut values = vec![0.0; self.n * self.n];
        for k in 0..self.m {
            for i in 0..self.n {
                for j in 0..=i {
                    let value = self.get(k, i) * self.get(k, j);
                    values[i * self.n + j] += value;
                    if i != j {
                        values[j * self.n + i] += value;
                    }
                }
            }
        }
        Matrix { m: self.n, n: self.n, values }
    }

    /// Multiplies `self` with `rhs^t`. Returns `None` if the dimensions don't match.
    pub fn mul_with_transposed(&self, rhs: &Matrix) -> Option<Matrix> {
        if self.n != rhs.n {
            None
        } else {
            let mut values = Vec::<f64>::with_capacity(self.m * rhs.m);
            for i in 0..self.m {
                for j in 0..rhs.m {
                    values.push(Vector::unchecked_dot(self.row_slice(i), rhs.row_slice(j)))
                }
            }
            Some(Matrix {
                m: self.m,
                n: rhs.m,
                values,
            })
        }
    }

    /// Computes `self * rhs` in the natural way using parallelization.
    /// 
    /// This method is used when matrices are small enough that pairs `self.row(i), rhs.row(i)`
    /// fit comfortably in the cache without explicit blocking.
    fn mul_simple_parallel(
        &self,
        rhs_t: &Matrix,
        out: &mut [f64]
    ) {
        out.par_chunks_mut(rhs_t.m).enumerate().for_each(|(i, out_row)| {
            let ith_row = self.row_slice(i);
            out_row.iter_mut().enumerate().for_each(
                |(j, out_elem)|
                *out_elem = Vector::unchecked_dot(ith_row, rhs_t.row_slice(j))
            );
        });
    }

    /// Computes `self * rhs` using parallelization and tiling: the output is
    /// processed in tiles so that the working set of `self` rows and `rhs_t`
    /// rows involved in a tile stays resident in cache across the inner iterations,
    /// cutting down on repeated DRAM traffic for `rhs_t`.
    /// 
    /// This method is used for large matrices.
    fn mul_tiled_parallel(
        &self,
        rhs_t: &Matrix,
        out: &mut [f64],
    ) {
        let l = rhs_t.m;
        out.par_chunks_mut(l * BLOCK_SIZE.min(self.m).max(1))
            .enumerate()
            .for_each(|(block_idx, out_block)| {
                let i_start = block_idx * BLOCK_SIZE.min(self.m).max(1);
                let rows_in_block = out_block.len() / l;
                for jj in (0..l).step_by(BLOCK_SIZE) {
                    let j_end = (jj + BLOCK_SIZE).min(l);
                    for bi in 0..rows_in_block {
                        let i = i_start + bi;
                        let a_row = self.row_slice(i);
                        let out_row = &mut out_block[bi * l..(bi + 1) * l];
                        out_row[jj..j_end].iter_mut().enumerate().for_each(
                            |(j, out_elem)|
                            *out_elem = Vector::unchecked_dot(a_row, rhs_t.row_slice(j))
                        );
                    }
                }
            });
    }
}