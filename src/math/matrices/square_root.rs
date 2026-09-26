//! Implements methods to compute the square root of a matrix (whenever it exists).
//! 
//! These methods are based on the paper
//! [Blocked Schur Algorithms for Computing the Matrix Square Root](<https://link.springer.com/chapter/10.1007/978-3-642-36803-5_12>)
//! by E. Deadman, N. Highham and R. Ralha.

use crate::math::Complex;
use super::*;

const RECURSION_THRESHOLD: usize = 32;


impl<'a, T: Scalar> MatrixView<'a, T> {
    /// Computes the square root of an upper triangular matrix using the point method from Deadman, Highham and Ralha's paper,
    /// that is, we aim to solve the following equations for upper triangular `U` where `T := self`:
    /// 
    /// 2. `U_{ii}^2 = T_{ii}`
    /// 3. `U_{ii} U_{ij} + U_{ij} U_{jj} = T_{ij} - \sum_{k=i+1}^{j-1} U_{ik} U_{kj}`
    /// 
    /// These equations are solved column-wise and not superdiagonal-wise for better cache locality.
    /// 
    /// Since this function is private, it does not check whether `self` is actually a square upper triangular matrix.
    /// 
    /// Returns `None` iff `T` doesn't possess a square root of type `Matrix<T>`.
    /// This in turn can only happen for real `T`; then, it happens iff a diagonal
    /// entry of `T` is strictly negative.
    fn sqrt_of_upper_triangular_point_method(&self, mut out: MatrixViewMut<T>) -> Option<()> {
        let n = self.cols();
        // We'll need the diagonal values to solve (3)
        for i in 0..n {
            out.set(i, i, self.get(i, i).checked_sqrt()?);
        }
        for j in 1..n {
            for i in (0..j).rev() {
                let denom = out.get(i, i).add(out.get(j, j));
                out.set(
                    i, j,
                    if denom.is_zero() {
                        T::zero()
                    } else {
                        self.get(i, j)
                        .sub(
                            ((i+1)..j)
                            .map(|k| out.get(i, k).mul(out.get(k, j)))
                            .sum()
                        )
                        .div(denom)
                    }
                );
            }
        }
        Some(())
    }

    /// Computes the square root of an upper triangular matrix using the recursive method from Deadman, Highham and Ralha's paper,
    /// that is, we split all matrices into four blocks and aim to solve the
    /// following equations for upper triangular `U` where `T := self`:
    /// 
    /// 1. `U_{11}^2 = T_{11}` (done recursively)
    /// 2. `U_{22}^2 = T_{22}` (done recursively)
    /// 3. `U_{11} U_{12} + U_{12} U_{22} = T_{12}` (Sylvester equation)
    fn sqrt_of_upper_triangular_blocking_method(self, mut out: MatrixViewMut<T>) -> Option<()> {
        let n = self.rows();
        if n <= RECURSION_THRESHOLD {
            return self.sqrt_of_upper_triangular_point_method(out);
        }

        let split_at = (n+1)/2;
        let remaining = n - split_at;
        self.submatrix(0, 0, split_at, split_at)
        .sqrt_of_upper_triangular_blocking_method(
            out.submatrix_mut(0, 0, split_at, split_at)
        );
        self.submatrix(split_at, split_at, remaining, remaining)
        .sqrt_of_upper_triangular_blocking_method(
            out.submatrix_mut(split_at, split_at, remaining, remaining)
        );
        // `top` is `split_at`x`n` and `bottom` is `remaining`x`n`.
        let [tl, tr, _, br] = out.split_quadrants_mut(split_at, split_at);
        sylvester::sylvester_upper_triangular_rec(
            tl.as_view(),
            br.as_view(),
            self.submatrix(0, split_at, split_at, remaining),
            tr
        )
    }
}

impl<T: Real> Matrix<T> {
    pub fn sqrt(&self) -> Result<Matrix<Complex<T>>, String> {
        // Schur decomposition exists iff the matrix is square
        let (q, t) = self.complex_schur_decomposition()
            .ok_or("Matrix must be square.".to_string())?;
        let mut t_sqrt = Matrix::zeros(self.m, self.n); // Quadratic, same size as `self`
        t.view()
            .sqrt_of_upper_triangular_blocking_method(t_sqrt.view_mut())
            .ok_or("Matrix does not possess a square root (singularity detected).")?;
        let res = (&q)
            .mul(t_sqrt).unwrap()
            .mul(q.transpose().transform(|x| x.conjugate())).unwrap();
        Ok(res)
    }
}