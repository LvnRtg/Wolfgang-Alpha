use crate::math::Complex;
use crate::math::traits::*;
use crate::math::utils;
use super::Matrix;

/// Returns one eigenvalue of the 2x2 matrix `[a, b \ c, d]`. Its "partner" is `tr - lambda`
/// by trace preservation under similarity transforms.
/// 
/// This is real if and only if the discriminant is non-negative.
fn dominant_eigenvalue_2x2<T: Scalar>(a: T, b: T, c: T, d: T) -> Complex<T::UnderlyingReal> {
    let tr = a.add(d);
    let det = a.mul(d).sub(b.mul(c));
    let disc_sqrt = tr.mul(tr).sub(det.mul(T::from_usize(4))).to_complex().sqrt();
    tr.to_complex().add(disc_sqrt).div(T::UnderlyingReal::from_usize(2))
}

impl<T: Real> Matrix<Complex<T>> {
    /// Computes `self = M * self`, where `M` is the 2x2 block `[m11, m12 \ m21, m22]` acting
    /// on rows `i`/`j` (identity elsewhere). Only columns `>= col_start` are touched,
    /// which is valid whenever rows `i`/`j` are already known to be zero before that.
    fn apply_block_left(
        &mut self,
        i: usize,
        j: usize,
        m11: Complex<T>,
        m12: Complex<T>,
        m21: Complex<T>,
        m22: Complex<T>,
        col_start: usize,
    ) {
        for k in col_start..self.n {
            let ai = self.get(i, k);
            let aj = self.get(j, k);
            self.set(i, k, m11.mul(ai).add(m12.mul(aj)));
            self.set(j, k, m21.mul(ai).add(m22.mul(aj)));
        }
    }

    /// Computes `self = self * M`, where `M` is the 2x2 block `[m11, m12 \ m21, m22]` acting
    /// on columns `i`/`j` (identity elsewhere).
    fn apply_block_right(
        &mut self,
        i: usize,
        j: usize,
        m11: Complex<T>,
        m12: Complex<T>,
        m21: Complex<T>,
        m22: Complex<T>,
        row_start: usize,
    ) {
        for k in row_start..self.m {
            let ai = self.get(k, i);
            let aj = self.get(k, j);
            self.set(k, i, m11.mul(ai).add(m21.mul(aj)));
            self.set(k, j, m12.mul(ai).add(m22.mul(aj)));
        }
    }
}

impl<T: Real> Matrix<T> {
    /// Performs a wilkinson shift on real values.
    /// 
    /// Idea: shift the given 2x2 block towards the eigenvalue closest to `d` to accelerate convergence.
    /// If eigenvalues are complex, do not shift at all.
    pub fn real_wilkinson_shift(a: T, b: T, c: T, d: T) -> T {
        let tr = a.add(d);
        let det = a.mul(d).sub(b.mul(c));
        let disc = tr.mul(tr).sub(det.mul(T::from_usize(4)));
        if disc >= T::zero() {
            let sq = disc.sqrt();
            let l1 = tr.add(sq).div(T::from_usize(2));
            let l2 = tr.sub(sq).div(T::from_usize(2));
            if l1.sub(d).abs() < l2.sub(d).abs() {
                l1
            } else {
                l2
            }
        } else {
            d
        }
    }

    /// Returns `(U, H)` where `U` is a real orthogonal matrix and `H` is quasi-upper-triangular
    /// such that `self = U * H * U^T`.
    /// 
    /// *Quasi-upper-triangular* means upper triangular except for some 2x2 blocks straddling
    /// the diagonal (each corresponding to a pair of complex-conjugate eigenvalues, or, rarely,
    /// an undeflated pair of real ones).
    /// 
    /// Returns `None` iff the matrix isn't square.
    pub fn real_schur_decomposition(&self) -> Option<(Matrix<T>, Matrix<T>)> {
        /* This is structurally identical to `eigenvalues`'s Hessenberg + shifted-QR
        iteration (cf. `qr.rs`); the only difference is that instead of collapsing each
        deflated block into an eigenvalue, it keeps `H` and its accumulated orthogonal
        transform `U` around, and it explicitly maintains `H`'s zero pattern even in
        the non-convergence fallback path (which `eigenvalues` doesn't need to care
        about, since it only reads the trailing block's numbers). */
        if self.m != self.n {return None;}

        let n = self.m;
        if n == 0 {
            return Some((Matrix::identity(0), Matrix::identity(0)));
        }

        let (mut h, pg) = self.upper_hessenberg();
        let mut u = pg.transpose();

        let eps: T = T::from_f64(1e-13);
        const MAX_ITERS_PER_BLOCK: usize = 200;

        let mut p = n;

        while p > 0 {
            if p == 1 {
                break;
            }

            let mut iters = 0usize;
            loop {
                let mut split_at: Option<usize> = None;
                for i in (1..p).rev() {
                    let scale = h.get(i - 1, i - 1).abs().add(h.get(i, i).abs());
                    let threshold = eps.mul(utils::max(scale, T::min_positive()));
                    if h.get(i, i - 1).abs() <= threshold {
                        h.set(i, i - 1, T::zero());
                        split_at = Some(i);
                        break;
                    }
                }

                if let Some(i) = split_at {
                    if i == p - 1 {
                        p -= 1;
                        break;
                    } else if i == p - 2 {
                        p -= 2;
                        break;
                    }
                    // Interior split: `h` was already zeroed at (i, i-1); `p` is
                    // unchanged, and the next scan will pick up the separated pieces.
                }

                iters += 1;
                if iters > MAX_ITERS_PER_BLOCK {
                    // Give up deflating further. Unlike `eigenvalues`, we must keep
                    // `h` genuinely quasi-triangular, so explicitly sever the
                    // trailing block from whatever remains above it.
                    if p >= 2 {
                        if p >= 3 {
                            h.set(p - 2, p - 3, T::zero());
                        }
                        p -= 2;
                    } else {
                        p -= 1;
                    }
                    break;
                }

                let a = h.get(p - 2, p - 2);
                let b = h.get(p - 2, p - 1);
                let c = h.get(p - 1, p - 2);
                let d = h.get(p - 1, p - 1);
                let shift = Matrix::real_wilkinson_shift(a, b, c, d);

                for k in 0..p {
                    h.set(k, k, h.get(k, k).sub(shift));
                }

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

        Some((u, h))
    }

    /// Returns `(Q, T)` where `Q` is a complex-valued unitary matrix and `T` is a complex-valued
    /// upper triangular matrix such that `self = QUQ^{-1}`.
    /// 
    /// Returns `None` iff `self` isn't square.
    pub fn complex_schur_decomposition(&self) -> Option<(Matrix<Complex<T>>, Matrix<Complex<T>>)> {
        if self.m != self.n {return None;}

        let (u_real, h_real) = self.real_schur_decomposition()?;
        let n = h_real.m();

        let mut u = u_real.transform(|x| x.to_complex());
        let mut t = h_real.transform(|x| x.to_complex());

        // `h_real` is quasi-upper-triangular: every subdiagonal entry is either
        // exactly zero (a converged 1x1 block) or belongs to a genuine 2x2 block.
        // Walk the diagonal; for each remaining 2x2 block, apply a single complex
        // unitary similarity transform that zeroes its lower-left entry, turning it
        // into a proper (complex) upper-triangular 2x2 block.
        let mut i = 0;
        while i + 1 < n {
            let c = h_real.get(i + 1, i);
            let scale = h_real.get(i, i).abs().add(h_real.get(i+1, i+1).abs());
            let threshold = T::EPSILON.mul(if scale.is_zero() { T::one() } else { scale });
            if c.abs() <= threshold {
                i += 1;
                continue;
            }

            let a = h_real.get(i, i);
            let b = h_real.get(i, i + 1);
            let d = h_real.get(i + 1, i + 1);
            let lambda = dominant_eigenvalue_2x2(a, b, c, d);

            // Eigenvector of the block `[a, b \ c, d]` for `lambda`: from row 2 of
            // (M - lambda*I)v = 0, with c != 0 (guaranteed, since that's exactly the
            // condition that put us in this branch): v = (lambda - d, c).
            let v1 = lambda.sub(d);
            let v2 = c.to_complex();
            let norm = v1.real.mul(v1.real)
                .add(v1.imag.mul(v1.imag))
                .add(v2.real.mul(v2.real))
                .add(v2.imag.mul(v2.imag))
                .sqrt();
            let v1 = v1.div(norm);
            let v2 = v2.div(norm);

            // V = [v1, -conj(v2) \ v2, conj(v1)] is unitary with first column v,
            // so V^H * M * V has zero below its diagonal, with `lambda` in the
            // top-left corner (and, by trace preservation, its conjugate/partner
            // eigenvalue in the bottom-right one).
            let conj_v1 = v1.conjugate();
            let neg_conj_v2 = Complex { real: v2.real.neg(), imag: v2.imag };

            // Rows i,i+1 of T are zero for columns < i (quasi-triangular structure),
            // so the left-multiply only needs columns >= i.
            t.apply_block_left(i, i + 1, conj_v1, v2.conjugate(), v2.neg(), v1, i);
            // Columns i,i+1 of T (and of the dense U) may be nonzero anywhere above
            // the block, so the right-multiply needs the full row range.
            t.apply_block_right(i, i + 1, v1, neg_conj_v2, v2, conj_v1, 0);
            u.apply_block_right(i, i + 1, v1, neg_conj_v2, v2, conj_v1, 0);

            // Clean up rounding error; this entry is analytically zero.
            t.set(i + 1, i, Complex::zero());

            i += 2;
        }

        Some((u, t))
    }
}