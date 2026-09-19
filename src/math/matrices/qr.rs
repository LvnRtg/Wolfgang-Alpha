//! Implements the QR algorithm and QR-adjacent functions such as `eig` to obtain the eigenvalues of a matrix.

use crate::math::{Complex, utils};
use crate::math::traits::*;
use super::Matrix;

/// A Givens rotation acting on rows/columns `i` and `j`:
/// G = I everywhere except G[i][i] = c, G[j][j] = c, G[i][j] = s, G[j][i] = -s.
#[derive(Debug, Clone, Copy)]
struct GivensRotation<T: Scalar> {
    i: usize,
    j: usize,
    c: T,
    s: T,
}
 
impl<T: Scalar> GivensRotation<T> {
    #[inline]
    fn is_identity(&self) -> bool {
        self.c == T::one() && self.s == T::zero()
    }
}

impl<T: Scalar> Matrix<T> {
    /// Computes the matrix `G` such that `G * self` rotates the plane spanned by the rows `i` and `j`
    /// precisely such that `A[j][col]` becomes zero. Returns `GivensRotation` instead of `Matrix`
    /// to avoid materializing the full matrix.
    /// 
    /// Assumes that `self` is quadratic (if this function becomes public, this may be changed).
    fn givens_rotation(&self, i: usize, j: usize, col: usize) -> GivensRotation<T> {
        let x = self.get(i, col).powi(2)
            .add(self.get(j, col).powi(2))
            .sqrt();
        if utils::approx_eq(x, T::zero()) {
            GivensRotation { i, j, c: T::one(), s: T::zero() }
        } else {
            GivensRotation {
                i,
                j,
                c: self.get(i, col).div(x),
                s: self.get(j, col).div(x),
            }
        }
    }

    /// Performs `self = G * self` in place.
    /// 
    /// Assume that all columns before `col_start` are already zero in rows `i` and `j`.
    /// This is true for Hessenberg matrices processed left-to-right as we do in the QR algorithm.
    fn apply_givens_left(&mut self, rot: &GivensRotation<T>, col_start: usize) {
        if rot.is_identity() {
            return;
        }
        let (i, j, c, s) = (rot.i, rot.j, rot.c, rot.s);
        for k in col_start..self.n {
            let ai = self.get(i, k);
            let aj = self.get(j, k);
            self.set(i, k, c.mul(ai).add(s.mul(aj)));
            self.set(j, k, c.mul(aj).sub(s.mul(ai)));
        }
    }
 
    /// Performs `self = self * G^T` in place.
    fn apply_givens_right(&mut self, rot: &GivensRotation<T>, row_start: usize) {
        if rot.is_identity() {
            return;
        }
        let (i, j, c, s) = (rot.i, rot.j, rot.c, rot.s);
        for k in row_start..self.m {
            let ai = self.get(k, i);
            let aj = self.get(k, j);
            self.set(k, i, c.mul(ai).add(s.mul(aj)));
            self.set(k, j, c.mul(aj).sub(s.mul(ai)));
        }
    }

    /// Computes the eigenvalues of `self` in O(n^3) using a QR algorithm, Hessenberg matrices and Givens rotations.
    /// 
    /// Returns `None` if `self` is not quadratic. 
    pub fn eigenvalues(&self) -> Option<Vec<Complex<T::UnderlyingFloat>>> {
        if self.m != self.n {
            return None;
        }
        let n = self.m;
        if n == 0 {
            return Some(Vec::new());
        }
 
        let (mut h, pg) = self.upper_hessenberg();
        let mut u = pg.transpose();
 
        let mut eigenvalues = vec![Complex::<T::UnderlyingFloat>::zero(); n];
        let eps = T::UnderlyingFloat::from_f64(1e-13);
        const MAX_ITERS_PER_BLOCK: usize = 200;
 
        // `p` = size of the active (not-yet-deflated) leading block [0, p).
        let mut p = n;
 
        while p > 0 {
            if p == 1 {
                eigenvalues[0] = h.get(0, 0);
                break;
            }
 
            let mut iters = 0usize;
            loop {
                // Look for a subdiagonal entry we can treat as zero, scanning from the bottom of the active block.
                let mut split_at: Option<usize> = None;
                for i in (1..p).rev() {
                    let scale = h.get(i - 1, i - 1).abs().add(h.get(i, i).abs());
                    let threshold = eps.mul(utils::max(scale, T::UnderlyingFloat::min_positive()));
                    if h.get(i, i - 1).abs() <= threshold {
                        h.set(i, i - 1, Complex::<T::UnderlyingFloat>::zero());
                        split_at = Some(i);
                        break;
                    }
                }
 
                if let Some(i) = split_at {
                    // Interior split: the trailing sub-block [i, p) is
                    // independent of [0, i). We keep the same active size p
                    // and re-scan; the next pass finds the (now closer to
                    // the edge) splits first since we scan bottom-up.
                    if i == p - 1 {
                        eigenvalues[p - 1] = h.get(p - 1, p - 1);
                        p -= 1;
                        break;
                    } else if i == p - 2 {
                        let (e1, e2) = h.eigenvalues_of_2x2_block(p - 2).unwrap();
                        eigenvalues[p - 2] = e1;
                        eigenvalues[p - 1] = e2;
                        p -= 2;
                        break;
                    }
                }
 
                iters += 1;
                if iters > MAX_ITERS_PER_BLOCK {
                    // Give up deflating further rather than looping forever; take the trailing block as final.
                    if p >= 2 {
                        let (e1, e2) = h.eigenvalues_of_2x2_block(p - 2).unwrap();
                        eigenvalues[p - 2] = e1;
                        eigenvalues[p - 1] = e2;
                        p = p.saturating_sub(2);
                    } else {
                        eigenvalues[p - 1] = h.get(p - 1, p - 1);
                        p -= 1;
                    }
                    break;
                }
 
                // Wilkinson shift from the trailing 2x2 of the active block.
                let shift = Matrix::<T>::wilkinson_shift(
                    h.get(p - 2, p - 2),
                    h.get(p - 2, p - 1),
                    h.get(p - 1, p - 2),
                    h.get(p - 1, p - 1)
                );
 
                for k in 0..p {
                    h.set(k, k, h.get(k, k).sub(shift));
                }
 
                // R, plus the rotations used to build it. Q is never formed
                // densely: apply each rotation directly to h (forming R*Q)
                // and to u (accumulating U*Q), one O(n) update at a time.
                let (r, rotations) = h.qr_decomposition_for_hessenberg_matrix(p);
                h = r;
                for rot in &rotations {
                    h.apply_givens_right(rot, 0);
                    u.apply_givens_right(rot, 0);
                }
 
                for k in 0..p {
                    h.set(k, k, h.get(k, k).add(shift));
                }
            }
        }
 
        Some(eigenvalues)
    }

    /// Computes the QR decomposition of the leading `size x size` block of `self`.
    /// Only works if that leading block is a Hessenberg matrix.
    fn qr_decomposition_for_hessenberg_matrix(&self, size: usize) -> (Matrix<T>, Vec<GivensRotation<T>>) {
        let mut r = self.clone();
        let mut rotations = Vec::with_capacity(size.saturating_sub(1));
        for i in 0..size.saturating_sub(1) {
            let rot = r.givens_rotation(i, i + 1, i);
            // Columns < i are already zero in rows i, i+1 (Hessenberg structure maintained inductively), so restrict to i..n.
            r.apply_givens_left(&rot, i);
            rotations.push(rot);
        }
        (r, rotations)
    }

    /// Computes the matrix `G` such that `G * self` rotates the plane spanned by the rows `i` and `j`
    /// precisely such that `A[j][col]` becomes zero.
    /// 
    /// Returns it as complex matrix (even though for real input matrices, it is real) for the sake of simplicity.
    /// 
    /// Assumes that `self` is quadratic (if this function becomes public, this may be changed).
    fn upper_hessenberg(&self) -> (Matrix<Complex<T::UnderlyingFloat>>, Matrix<Complex<T::UnderlyingFloat>>) {
        let mut h = self.transform(|x| x.to_complex());
        let mut pg = Matrix::identity(self.m);
        if self.m > 1 {
            for col in 0..self.m - 2 {
                for row in col + 2..self.m {
                    let rot = h.givens_rotation(col + 1, row, col);
                    if rot.is_identity() {
                        continue;
                    }
                    // Compute the orthogonal similarity transform `self = G * self * G^T`
                    h.apply_givens_left(&rot, col);
                    // The right-multiply changes columns col+1 and row across potentially all rows, so no range restriction here.
                    h.apply_givens_right(&rot, 0);
                    // `pg` is a dense accumulator, needs the full column range.
                    pg.apply_givens_left(&rot, 0);
                }
            }
        }
        (h, pg)
    }

    /// Performs a wilkinson shift.
    /// 
    /// Idea: shift the given 2x2 block towards the eigenvalue closest to `d` to accelerate convergence.
    fn wilkinson_shift(a: Complex<T::UnderlyingFloat>, b: Complex<T::UnderlyingFloat>, c: Complex<T::UnderlyingFloat>, d: Complex<T::UnderlyingFloat>) -> Complex<T::UnderlyingFloat> {
        let tr = a.add(d);
        let det = a.mul(d).sub(b.mul(c));
        let disc_sqrt = tr.mul(tr).sub(det.mul(T::UnderlyingFloat::from_usize(4))).sqrt();
        let l1 = tr.add(disc_sqrt).div(T::UnderlyingFloat::from_usize(2));
        let l2 = tr.sub(disc_sqrt).div(T::UnderlyingFloat::from_usize(2));
        if l1.sub(d).modulus() <= l2.sub(d).modulus() {
            l1
        } else {
            l2
        }
    }

    /// Returns the eigenvalues of the 2x2-block of `self` situated at `(i, i)`. Only returns `Some` this block exists.
    fn eigenvalues_of_2x2_block(&self, i: usize) -> Option<(Complex<T::UnderlyingFloat>, Complex<T::UnderlyingFloat>)> {
        if self.m <= i + 1 || self.n <= i + 1 {return None;}
        let a = self.get(i, i);
        let b = self.get(i, i + 1);
        let c = self.get(i + 1, i);
        let d = self.get(i + 1, i + 1);
        let tr = a.add(d);
        let det = a.mul(d).sub(b.mul(c));
        let disc_sqrt = tr.mul(tr).sub(det.mul(T::UnderlyingFloat::from_usize(4))).to_complex().sqrt();
        Some((
            tr.to_complex().add(disc_sqrt).div(T::UnderlyingFloat::from_usize(2)),
            tr.to_complex().sub(disc_sqrt).div(T::UnderlyingFloat::from_usize(2))
        ))
    }
}