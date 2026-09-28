use super::*;

mod svd_tests;

/// A singular value decomposition. For an `m`x`n` matrix `A`,
/// this should satisfy `A = U * diag(s) * V^*` where `U` is
/// `m`x`k` and `V` is `n`x`k` for `k := min(m, n)`.
/// 
/// The singular values in `s` are sorted in descending order.
pub struct Svd<T: Scalar> {
    pub u: Matrix<T>,
    pub s: Vec<T::UnderlyingReal>,
    pub v: Matrix<T>,
}


/// Returns `(c, s, r)` such that `c*a + s*b = r` and `-s*a + c*b = 0`.
fn givens<R: Real>(a: R, b: R) -> (R, R, R) {
    if b.is_zero() {
        (R::one(), R::zero(), a)
    } else {
        let r = a.powi(2).add(b.powi(2)).sqrt();
        (a.div(r), b.div(r), r)
    }
}
/// Acts like LAPACK's' `larfg`.
/// 
/// Returns `(tau, beta)` for real `beta` such that `(I - tau v v^*)^* x = beta * e1`.
/// 
/// On entry, `x` the vector `x` appearing above`. On exit, `x` holds the vector `v` instead (with `v[0] = 1`).
fn householder<T: Scalar>(x: &mut [T]) -> (T, T::UnderlyingReal) {
    let alpha = x[0];
    let mut tail = T::UnderlyingReal::zero();
    for xi in &x[1..] {
        tail.add_assign(xi.abs().squared());
    }
    if tail.is_zero() && let Some(beta) = alpha.to_real() {
        x[0] = T::one();
        return (T::zero(), beta);
    }
    let norm = (alpha.abs().squared().add(tail)).sqrt();
    let beta = if alpha.real() >= T::UnderlyingReal::zero() { norm.neg() } else { norm };
    let b = T::from_real(beta);
    let tau = b.sub(alpha).div(b);
    let scale = T::one().div(alpha.sub(b));
    for xi in x[1..].iter_mut() {
        xi.mul_assign(scale);
    }
    x[0] = T::one();
    (tau, beta)
}



impl<T: Scalar> Matrix<T> {
    /// Returns the singular value decomposition of `self`.
    /// 
    /// Returns `None` iff the iteration fails to converge.
    pub fn svd(&self) -> Option<Svd<T>> {
        if self.m >= self.n {
            self.svd_tall()
        } else {
            // Transpose `self` to make it a tall matrix
            self.transpose().conjugate()
            .svd_tall()
            .map(|r| Svd { u: r.v, s: r.s, v: r.u })
        }
    }

    /// Redefines `A[r0.., c0..c1]` as `(I - tau v v^*) * A[r0.., c0..c1]`.
    fn apply_left(&mut self, r0: usize, c0: usize, c1: usize, v: &[T], tau: T) {
        for j in c0..c1 {
            let mut w = T::zero();
            for (i, &vi) in v.iter().enumerate() {
                w.add_assign(vi.conjugate().mul(self.get(r0 + i, j)));
            }
            for (i, &vi) in v.iter().enumerate() {
                let x = self.get(r0 + i, j).sub(vi.mul(tau.mul(w)));
                self.set(r0 + i, j, x);
            }
        }
    }
    /// Redefines `A[r0..r1, c0..]` as `A[r0..r1, c0..] * (I - tau v v^*)`.
    fn apply_right(&mut self, r0: usize, r1: usize, c0: usize, v: &[T], tau: T) {
        for i in r0..r1 {
            let mut w = T::zero();
            for (j, &vj) in v.iter().enumerate() {
                w.add_assign(self.get(i, c0 + j).mul(vj));
            }
            for (j, &vj) in v.iter().enumerate() {
                let x = self.get(i, c0 + j).sub(tau.mul(w).mul(vj.conjugate()));
                self.set(i, c0 + j, x);
            }
        }
    }
    /// Real plane rotation on columns `p`, `q`: `(Mp, Mq)` becomes `(c*Mp + s*Mq, -s*Mp + c*Mq)`.
    fn rot_cols(&mut self, p: usize, q: usize, c: T::UnderlyingReal, s: T::UnderlyingReal) {
        let (cc, ss) = (T::from_real(c), T::from_real(s));
        for i in 0..self.m {
            let (x, y) = (self.get(i, p), self.get(i, q));
            self.set(i, p, cc.mul(x).add(ss.mul(y)));
            self.set(i, q, cc.mul(y).sub(ss.mul(x)));
        }
    }

    /// Bidiagonalizes the `m`x`n` matrix `self`, that is, returns unitary matrices `U` (`m`x`n`)
    /// and `V` (`n`x`n`) as well as two vectors `d`, `e` such that `A = U * upper_bidiag(d, e) * V^*`.
    /// 
    /// An upper bidiagonal matrix is a matrix that only has non-zero entries on its main diagonal
    /// and on the diagonal above the main diagonal.
    /// 
    /// ## Returns `None` if `m < n`.
    /// 
    /// This function requires about `4mn² - (4/3)n³` flops.
    pub fn bidiagonalize(mut self) -> Option<(Matrix<T>, Vec<T::UnderlyingReal>, Vec<T::UnderlyingReal>, Matrix<T>)> {
        let (m, n) = (self.m, self.n);
        if m < n {return None;}
        let mut d = vec![T::UnderlyingReal::zero(); n];
        let mut e = vec![T::UnderlyingReal::zero(); n.saturating_sub(1)];
        let mut left: Vec<(Vec<T>, T)> = Vec::with_capacity(n);
        let mut right: Vec<(Vec<T>, T)> = Vec::with_capacity(n.saturating_sub(1));
    
        for k in 0..n {
            // Left reflector: zero A[k+1.., k]
            let mut v: Vec<T> = (k..m).map(|i| self.get(i, k)).collect();
            let (tau, beta) = householder(&mut v);
            d[k] = beta;
            self.apply_left(k, k + 1, n, &v, tau.conjugate()); // H^* A
            left.push((v, tau));
    
            // Right reflector: zero A[k, k+2..]
            if k + 1 < n {
                let mut w: Vec<T> = (k + 1..n).map(|j| self.get(k, j).conjugate()).collect();
                let (tau, beta) = householder(&mut w);
                e[k] = beta;
                self.apply_right(k + 1, m, k + 1, &w, tau); // A H
                right.push((w, tau));
            }
        }
    
        // Accumulate U1 = H_0 H_1 ... H_{n-1} (first n columns), backward.
        let mut u = Matrix::zeros(m, n);
        for i in 0..n {
            u.set(i, i, T::one());
        }
        for k in (0..n).rev() {
            let (v, tau) = &left[k];
            u.apply_left(k, 0, n, v, *tau);
        }
    
        // Accumulate V1 = G_0 G_1 ... G_{n-2}, backward.
        let mut vm = Matrix::zeros(n, n);
        for i in 0..n {
            vm.set(i, i, T::one());
        }
        for k in (0..right.len()).rev() {
            let (w, tau) = &right[k];
            vm.apply_left(k + 1, 0, n, w, *tau);
        }
    
        Some((u, d, e, vm))
    }

    /// On entry, `d`, `e`, `u` and `v` are such that `A = U * upper_bidiag(d, e) * V^*`.
    /// 
    /// On exit, `d` holds the (unsorted, possibly negative) singular values of `A`
    /// and `U`, `V` have been updated so that we now have `A = U diag(d) V^*`.
    /// 
    /// Returns `None` if the iteration fails to converge.
    fn bidiagonal_qr(
        d: &mut [T::UnderlyingReal],
        e: &mut [T::UnderlyingReal],
        u: &mut Matrix<T>,
        v: &mut Matrix<T>,
    ) -> Option<()> {
        let n = d.len();
        if n <= 1 {
            return Some(());
        }
        let zero = T::UnderlyingReal::zero();
        let eps = T::UnderlyingReal::EPSILON;
    
        let bnorm = utils::max_abs_of(d.iter().chain(e.iter()));
        let tol = eps.mul(bnorm);
        let max_iter = 75 * n;
        let mut iter = 0;
        let mut hi = n - 1;
    
        loop {
            // Deflation: negligible off-diagonals are set to zero.
            for i in 0..n-1 {
                if e[i].abs() <= eps.mul(d[i].abs().add(d[i + 1].abs())) {
                    e[i] = zero;
                }
            }
            while hi > 0 && e[hi - 1].is_zero() {
                hi -= 1;
            }
            if hi == 0 {
                return Some(());
            }
            let mut lo = hi;
            while lo > 0 && !e[lo - 1].is_zero() {
                lo -= 1;
            }

            for i in lo..=hi {
                if d[i].abs() <= tol {
                    d[i] = zero;
                }
            }

            // Case 1: if `d[hi]` is zero, chase `e[hi-1]` out along the last column using right rotations
            if d[hi].is_zero() {
                let mut f = e[hi - 1];
                e[hi - 1] = zero;
                for j in (lo..hi).rev() {
                    let (c, s, r) = givens(d[j], f);
                    d[j] = r;
                    if j > lo {
                        f = s.neg().mul(e[j - 1]);
                        e[j - 1] = c.mul(e[j - 1]);
                    }
                    v.rot_cols(j, hi, c, s);
                }
                continue;
            }

            // Case 2: if `d[i]` for some `i < hi`, chase `e[i]` out along row `i` using _left_ rotations
            if let Some(i) = (lo..hi).find(|&i| d[i].is_zero()) {
                let mut f = e[i];
                e[i] = zero;
                for j in i + 1..=hi {
                    let (c, s, r) = givens(d[j], f);
                    d[j] = r;
                    if j < hi {
                        f = s.neg().mul(e[j]);
                        e[j] = c.mul(e[j]);
                    }
                    u.rot_cols(j, i, c, s);
                }
                continue;
            }

            // Case 3: one Golub-Kahan step with a Wilkinson shift on the block `lo..=hi`
            iter += 1;
            if iter > max_iter {
                return None;
            }

            // Similarly to the other QR algorithm (see `qr.rs`), we shift the 2x2 block towards its largest eigenvalue.
            let (dm, dn, fm) = (d[hi - 1], d[hi], e[hi - 1]);
            let em = if hi - 1 > lo { e[hi - 2] } else { zero };
            let ta = dm.squared().add(em.squared());
            let tb = dm.mul(fm);
            let tc = dn.squared().add(fm.squared());
            let delta = ta.sub(tc).div(T::UnderlyingReal::from_usize(2));
            let root = delta.squared().add(tb.squared()).sqrt();
            let denom = if delta >= zero { delta.add(root) } else { delta.sub(root) };
            let mu = if denom.is_zero() { tc } else { tc.sub(tb.squared().div(denom)) };

            let mut y = d[lo].squared().sub(mu);
            let mut z = d[lo].mul(e[lo]);

            for k in lo..hi {
                // The right rotation on columns (k, k+1); creates "bulge" at (k+1, k), i.e. a larger value
                let (c, s, r) = givens(y, z);
                if k > lo {
                    e[k - 1] = r;
                }
                let f = c.mul(d[k]).add(s.mul(e[k]));
                e[k] = c.mul(e[k]).sub(s.mul(d[k]));
                let g = s.mul(d[k + 1]);
                d[k + 1] = c.mul(d[k + 1]);
                v.rot_cols(k, k + 1, c, s);
    
                // A left rotation on rows (k, k+1) removes bulge but creates a new one at (k, k+2) instead.
                // Thus, we loop to push it away further and further.
                let (c, s, r) = givens(f, g);
                d[k] = r;
                let t = c.mul(e[k]).add(s.mul(d[k + 1]));
                d[k + 1] = c.mul(d[k + 1]).sub(s.mul(e[k]));
                e[k] = t;
                if k + 1 < hi {
                    let ek1 = e[k + 1];
                    z = s.mul(ek1);
                    e[k + 1] = c.mul(ek1);
                    y = e[k];
                }
                u.rot_cols(k, k + 1, c, s);
            }
        }
    }

    /// Computes the SVD of an `m`x`n` matrix for which `m >= n`.
    /// 
    /// Returns `None` if the iteration fails to converge.
    fn svd_tall(&self) -> Option<Svd<T>> {
        let n = self.n;
        if n == 0 {
            return Some(Svd { u: Matrix::zeros(self.m, 0), s: vec![], v: Matrix::zeros(0, 0) });
        }
        let (mut u, mut d, mut e, mut v) = self.clone().bidiagonalize().unwrap();
        Self::bidiagonal_qr(&mut d, &mut e, &mut u, &mut v)?;
    
        // Make all singular values nonnegative by moving the sign into `V`.
        let minus_one = T::one().neg();
        for j in 0..n {
            if d[j] < T::UnderlyingReal::zero() {
                d[j] = d[j].neg();
                for i in 0..v.m {
                    v.set(i, j, v.get(i, j).mul(minus_one));
                }
            }
        }
    
        // Sort the singular values in descending order.
        // The defining equation of the SVD is preserved by permuting columns of `U` and `V`.
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&i, &j| d[j].partial_cmp(&d[i]).unwrap_or(std::cmp::Ordering::Equal));
        let mut u2 = Matrix::zeros(u.m, n);
        let mut v2 = Matrix::zeros(v.m, n);
        let mut s = Vec::with_capacity(n);
        for (new, &old) in order.iter().enumerate() {
            s.push(d[old]);
            for i in 0..u.m {
                u2.set(i, new, u.get(i, old));
            }
            for i in 0..v.m {
                v2.set(i, new, v.get(i, old));
            }
        }
        Some(Svd { u: u2, s, v: v2 })
    }

}