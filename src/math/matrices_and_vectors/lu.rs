//! Implements LU decomposition, LU decomposition with pivoting and full-pivot LU decomposition.

use crate::math::{Matrix, utils, Vector};

impl Matrix {
    /// Computes the LU decomposition of `self`.
    /// 
    /// Note: this does not necessarily exist just because a matrix is invertible.
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
                    // Compute self[i, k] - dot(u.col(k), l.row(i))
                    self.get(i, k) - Vector::unchecked_dot_iter(utils::col(&u, k, i, self.m), utils::row(&l, i, self.m))
                } else { self.get(i, k) };
                u.push(newval); // Can't condense into single line because of ownership of u
            }
            for k in i..self.m {
                if utils::approx_eq(u[i*self.m + i], 0.0) {
                    return None; // No LU decomposition
                }
                l[k*self.m + i] = (if i > 0 {
                    self.get(k, i) - Vector::unchecked_dot_iter(utils::col(&u, i, i, self.m), &utils::row(&l, k, self.m)[0..i])
                    //                  = (0..i).map(|j| l[k*self.m + j] * u[j*self.m + i]).sum::<f64>()
                } else { self.get(k, i) }) / u[i*self.m + i];
            }
        }
        Some((Matrix{m: self.m, n: self.m, values: l}, Matrix{m: self.m, n: self.m, values: u}))
    }

    /// Computes the PLU decomposition of `self` (i.e. with partial pivoting).
    /// The returned vector `p` encodes `P` via `P[i][p[i]] = 1`.
    /// 
    /// Note: this exists iff the matrix is invertible.
    /// 
    /// Time complexity: O(n^3).
    pub fn plu_decomposition(&self) -> Option<(Vec<usize>, Matrix, Matrix)> {
        if self.m != self.n { return None; } // Ensure the matrix is square

        let n = self.m;
        let mut a = self.clone();
        let mut l = Matrix::zeros(n, n);
        let mut u = Matrix::zeros(n, n);
        let mut perm: Vec<usize> = (0..n).collect();

        for i in 0..n {
            // Reduced values in column i for rows i..n
            let reduced = |r: usize| -> f64 {
                a.get(r, i) - Vector::unchecked_dot_iter(utils::col(&u.values, i, i, u.n), l.row_slice(r))
                //               = (0..i).map(|j| l.get(r, j) * u.get(j, i)).sum::<f64>()
            };
            // The `unwrap_or` below is there to avoid panicking if either compared value is NaN
            let pivot_row = (i..n)
                .max_by(|&r1, &r2| reduced(r1).abs().partial_cmp(&reduced(r2).abs()).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap();
            if pivot_row != i {
                perm.swap(i, pivot_row);
                for col in 0..n {
                    a.values.swap(i * n + col, pivot_row * n + col);
                }
                for col in 0..i {
                    l.values.swap(i * n + col, pivot_row * n + col);
                }
            }
            // Compute `i`-th row of `u` and `i`-th column of `l`
            for k in i..n {
                u.set(i, k, a.get(i, k) - Vector::unchecked_dot_iter(utils::col(&u.values, k, i, u.n), l.row_slice(i)));
                //                                 = (0..i).map(|j| l.get(i, j) * u.get(j, k)).sum::<f64>()
            }
            if utils::approx_eq(u.get(i, i), 0.0) {
                return None; // Matrix is singular
            }
            for k in i..n {
                l.set(k, i, (a.get(k, i) - Vector::unchecked_dot_iter(utils::col(&u.values, i, i, u.n), l.row_slice(k))) / u.get(i, i));
                //                                       = (0..i).map(|j| l.get(k, j) * u.get(j, i)).sum::<f64>()
            }
        }

        Some((utils::transpose_permutation(&perm), l, u))
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
}