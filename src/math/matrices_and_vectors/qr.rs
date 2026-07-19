//! Implements the QR algorithm and QR-adjacent functions such as `eig` to obtain the eigenvalues of a matrix.

use crate::math::{Complex, Matrix, Object, utils};

/// A Givens rotation acting on rows/columns `i` and `j`:
/// G = I everywhere except G[i][i] = c, G[j][j] = c, G[i][j] = s, G[j][i] = -s.
#[derive(Debug, Clone, Copy)]
struct GivensRotation {
    i: usize,
    j: usize,
    c: f64,
    s: f64,
}
 
impl GivensRotation {
    #[inline]
    fn is_identity(&self) -> bool {
        self.c == 1.0 && self.s == 0.0
    }
}

impl Matrix {
    /// Computes the matrix `G` such that `G * self` rotates the plane spanned by the rows `i` and `j`
    /// precisely such that `A[j][col]` becomes zero. Returns `GivensRotation` instead of `Matrix`
    /// to avoid materializing the full matrix.
    /// 
    /// Assumes that `self` is quadratic (if this function becomes public, this may be changed).
    fn givens_rotation(&self, i: usize, j: usize, col: usize) -> GivensRotation {
        let x = (self.get(i, col).powi(2) + self.get(j, col).powi(2)).sqrt();
        if utils::approx_eq(x, 0.0) {
            return GivensRotation { i, j, c: 1.0, s: 0.0 };
        }
        GivensRotation {
            i,
            j,
            c: self.get(i, col) / x,
            s: self.get(j, col) / x,
        }
    }

    /// Performs `self = self * G` in place.
    /// 
    /// Assume that all columns before `col_start` are already zero in rows `i` and `j`.
    /// This is true for Hessenberg matrices processed left-to-right as we do in the QR algorithm.
    fn apply_givens_left(&mut self, rot: &GivensRotation, col_start: usize) {
        if rot.is_identity() {
            return;
        }
        let (i, j, c, s) = (rot.i, rot.j, rot.c, rot.s);
        for k in col_start..self.n {
            let ai = self.get(i, k);
            let aj = self.get(j, k);
            self.set(i, k, c * ai + s * aj);
            self.set(j, k, -s * ai + c * aj);
        }
    }
 
    /// Performs `self = self * G^T` in place.
    fn apply_givens_right(&mut self, rot: &GivensRotation, row_start: usize) {
        if rot.is_identity() {
            return;
        }
        let (i, j, c, s) = (rot.i, rot.j, rot.c, rot.s);
        for k in row_start..self.m {
            let ai = self.get(k, i);
            let aj = self.get(k, j);
            self.set(k, i, c * ai + s * aj);
            self.set(k, j, -s * ai + c * aj);
        }
    }

    /// Computes the eigenvalues and QR decomposition of `self` in O(n^3) using Hessenberg matrices and Givens rotations.
    /// 
    /// Returns `None` if `self` is not quadratic. Otherwise, returns `(eigenvalues, q, r)`.
    pub fn qr_decomposition(&self) -> Option<(Vec<Object>, Matrix, Matrix)> {
        if self.m != self.n {
            return None;
        }
        let n = self.m;
        if n == 0 {
            return Some((Vec::new(), self.clone(), Matrix::identity(0)));
        }
 
        let (mut h, pg) = self.upper_hessenberg();
        let mut u = pg.transpose();
 
        let mut eigenvalues = vec![Object::Real(0.0); n];
        const EPS: f64 = 1e-13;
        const MAX_ITERS_PER_BLOCK: usize = 200;
 
        // `p` = size of the active (not-yet-deflated) leading block [0, p).
        let mut p = n;
 
        while p > 0 {
            if p == 1 {
                eigenvalues[0] = Object::Real(h.get(0, 0));
                break;
            }
 
            let mut iters = 0usize;
            loop {
                // Look for a subdiagonal entry we can treat as zero, scanning
                // from the bottom of the active block.
                let mut split_at: Option<usize> = None;
                for i in (1..p).rev() {
                    let scale = h.get(i - 1, i - 1).abs() + h.get(i, i).abs();
                    let threshold = EPS * scale.max(f64::MIN_POSITIVE);
                    if h.get(i, i - 1).abs() <= threshold {
                        h.set(i, i - 1, 0.0);
                        split_at = Some(i);
                        break;
                    }
                }
 
                if let Some(i) = split_at {
                    if i == p - 1 {
                        eigenvalues[p - 1] = Object::Real(h.get(p - 1, p - 1));
                        p -= 1;
                    } else if i == p - 2 {
                        let (e1, e2) = h.eigenvalues_of_2x2_block(p - 2).unwrap();
                        eigenvalues[p - 2] = e1;
                        eigenvalues[p - 1] = e2;
                        p -= 2;
                    }
                    // Interior split: the trailing sub-block [i, p) is
                    // independent of [0, i). We keep the same active size p
                    // and re-scan; the next pass finds the (now closer to
                    // the edge) splits first since we scan bottom-up.
                    break;
                }
 
                iters += 1;
                if iters > MAX_ITERS_PER_BLOCK {
                    // Give up deflating further rather than looping forever;
                    // take the trailing block as final.
                    if p >= 2 {
                        let (e1, e2) = h.eigenvalues_of_2x2_block(p - 2).unwrap();
                        eigenvalues[p - 2] = e1;
                        eigenvalues[p - 1] = e2;
                        p = p.saturating_sub(2);
                    } else {
                        eigenvalues[p - 1] = Object::Real(h.get(p - 1, p - 1));
                        p -= 1;
                    }
                    break;
                }
 
                // Wilkinson shift from the trailing 2x2 of the active block.
                let a = h.get(p - 2, p - 2);
                let b = h.get(p - 2, p - 1);
                let c = h.get(p - 1, p - 2);
                let d = h.get(p - 1, p - 1);
                let shift = Matrix::wilkinson_shift(a, b, c, d);
 
                for k in 0..n {
                    h.set(k, k, h.get(k, k) - shift);
                }
 
                // R, plus the rotations used to build it. Q is never formed
                // densely: apply each rotation directly to h (forming R*Q)
                // and to u (accumulating U*Q), one O(n) update at a time.
                let (r, rotations) = h.qr_decomposition_for_hessenberg_matrix();
                h = r;
                for rot in &rotations {
                    h.apply_givens_right(rot, 0);
                    u.apply_givens_right(rot, 0);
                }
 
                for k in 0..n {
                    h.set(k, k, h.get(k, k) + shift);
                }
            }
        }
 
        Some((eigenvalues, h, u))
    }

    /// Computes the QR decomposition of `self`. Only works if `self` is a Hessenberg matrix.
    fn qr_decomposition_for_hessenberg_matrix(&self) -> (Matrix, Vec<GivensRotation>) {
        let mut r = self.clone();
        let mut rotations = Vec::with_capacity(self.m.saturating_sub(1));
        for i in 0..self.m.saturating_sub(1) {
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
    /// Assumes that `self` is quadratic (if this function becomes public, this may be changed).
    fn upper_hessenberg(&self) -> (Matrix, Matrix) {
        let mut h = self.clone();
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
    fn wilkinson_shift(a: f64, b: f64, c: f64, d: f64) -> f64 {
        let tr = a + d;
        let det = a * d - b * c;
        let disc = tr * tr - 4.0 * det;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            let l1 = (tr + sq) / 2.0;
            let l2 = (tr - sq) / 2.0;
            if (l1 - d).abs() < (l2 - d).abs() {
                l1
            } else {
                l2
            }
        } else {
            d
        }
    }

    /// Returns the eigenvalues of the 2x2-block of `self` situated at `(i, i)`. Only returns `Some` this block exists.
    fn eigenvalues_of_2x2_block(&self, i: usize) -> Option<(Object, Object)> {
        if self.m <= i + 1 || self.n <= i+1 {return None;}
        let a = self.get(i, i);
        let b = self.get(i, i + 1);
        let c = self.get(i + 1, i);
        let d = self.get(i + 1, i + 1);
        let tr = a + d;
        let det = a * d - b * c;
        let disc = tr * tr - 4.0 * det;
        if disc >= 0.0 {
            let sq = disc.sqrt();
            Some((
                Object::Real((tr + sq) / 2.0),
                Object::Real((tr - sq) / 2.0),
            ))
        } else {
            let re = tr / 2.0;
            let im = (-disc).sqrt() / 2.0;
            Some((Object::Complex(Complex { real: re, imag: im }), Object::Complex(Complex { real: re, imag: -im })))
        }
    }
}