#[allow(dead_code)]
mod support {
    use super::super::*;

    pub struct Rng(pub u64);
    impl Rng {
        pub fn next(&mut self) -> f64 {
            self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        }
    }
    
    pub fn rand_matrix(m: usize, n: usize, rng: &mut Rng) -> Matrix<f64> {
        let values = (0..m * n)
            .map(|_| {
                rng.next()
            })
            .collect();
        Matrix::from(m, n, values)
    }
    pub fn rand_matrix_c(m: usize, n: usize, rng: &mut Rng) -> Matrix<Complex<f64>> {
        let values = (0..m * n)
            .map(|_| {
                let real = rng.next();
                let imag = rng.next();
                Complex{real, imag}
            })
            .collect();
        Matrix::from(m, n, values)
    }
    
    /// Check XᴴX = I for a matrix with orthonormal columns.
    pub fn assert_orthonormal_cols<T: Scalar>(x: &Matrix<T>) {
        assert!(x.transpose_conjugate().mul(x).unwrap().approx_eq(&Matrix::identity(x.n())));
    }

    #[inline]
    pub fn max_diff<T: Scalar>(a: &Matrix<T>, b: &Matrix<T>) -> T::UnderlyingReal {
        utils::max_of(a.iter().zip(b.iter()).map(|(x, y)| x.sub(*y).abs())).unwrap_or(T::UnderlyingReal::zero())
    }
    
    pub fn assert_close_vec(got: &[f64], want: &[f64], tol: f64) {
        assert_eq!(got.len(), want.len());
        for (i, (g, w)) in got.iter().zip(want).enumerate() {
            assert!((g - w).abs() <= tol, "index {i}: got {g:?}, want {w:?}");
        }
    }
}

#[cfg(test)]
mod givens_tests {
    use super::super::*;

    #[test]
    fn givens_zeros_second_component() {
        let cases: [(f64, f64); 6] = [(3.0, 4.0), (-3.0, 4.0), (3.0, -4.0), (0.0, 2.0), (1e-8, 1.0), (7.0, 1e-9)];
        for &(a, b) in &cases {
            let (c, s, r) = givens(a, b);
            assert!((c * c + s * s - 1.0).abs() < 1e-14, "c²+s²≠1 for ({a},{b})");
            assert!((c * a + s * b - r).abs() < 1e-14 * r.abs().max(1.0));
            assert!((-s * a + c * b).abs() < 1e-14 * r.abs().max(1.0));
            assert!((r.abs() - a.hypot(b)).abs() < 1e-14 * r.abs().max(1.0));
        }
    }
    
    #[test]
    fn givens_known_values() {
        let (c, s, r) = givens(3.0, 4.0);
        assert!(f64::abs(c - 0.6) < 1e-15 && (s - 0.8).abs() < 1e-15 && (r - 5.0).abs() < 1e-15);
    }
    
    #[test]
    fn givens_degenerate_inputs() {
        assert_eq!(givens(5.0, 0.0), (1.0, 0.0, 5.0));
        assert_eq!(givens(-5.0, 0.0), (1.0, 0.0, -5.0));
        assert_eq!(givens(0.0, 0.0), (1.0, 0.0, 0.0));
    }
}

#[cfg(test)]
mod householder_tests {
    use super::super::*;
    use super::support::*;

    /// Verifies Hᴴx = β e₁ (β real), |β| = ‖x‖, v₀ = 1, and H unitary.
    fn check_householder<T: Scalar>(x: Vec<T>, tol: T::UnderlyingReal) {
        let n = x.len();
        let mut v = x.clone();
        let (tau, beta) = householder(&mut v);
        assert!(v[0] == T::one(), "v[0] must be 1");
    
        // dense H = I − τ v vᴴ
        let mut hv = vec![T::zero(); n * n];
        for i in 0..n {
            for j in 0..n {
                let id = if i == j { T::one() } else { T::zero() };
                hv[i * n + j] = id.sub(tau.mul(v[i]).mul(v[j].conjugate()));
            }
        }
        let h = Matrix::from(n, n, hv);
    
        let unitary_err = max_diff(&h.transpose_conjugate().mul(&h).unwrap(), &Matrix::identity(n));
        assert!(unitary_err <= tol, "H not unitary: {unitary_err:?}");
    
        let xm = Matrix::from(n, 1, x.clone());
        let y = h.transpose_conjugate().mul(&xm).unwrap();
        let b = T::from_real(beta);
        assert!(y.values[0].sub(b).abs() <= tol, "Hᴴx first entry ≠ β");
        for i in 1..n {
            assert!(y.values[i].abs() <= tol, "Hᴴx entry {i} not zero");
        }
        let norm = x.iter().map(|z| z.abs().powi(2)).sum::<T::UnderlyingReal>().sqrt();
        assert!(beta.abs().sub(norm).abs() <= tol.mul(utils::max(norm, T::UnderlyingReal::one())), "|β| ≠ ‖x‖");
    }
    
    #[test]
    fn householder_real_known_values() {
        // x = (3, 4): β = −5
        let mut v: Vec<f64> = vec![3.0, 4.0];
        let (tau, beta) = householder(&mut v);
        assert!((beta + 5.0).abs() < 1e-14);
        assert!((tau - (beta - 3.0) / beta).abs() < 1e-14);
        assert_eq!(v[0], 1.0);
        assert!((v[1] - 4.0 / (3.0 - beta)).abs() < 1e-14);
    
        // x = (−3, 4): β = +5 (sign opposite to x₀ avoids cancellation)
        let mut v: Vec<f64> = vec![-3.0, 4.0];
        let (_, beta) = householder(&mut v);
        assert!((beta - 5.0).abs() < 1e-14);
    }
    
    #[test]
    fn householder_real_random() {
        let mut rng = Rng(1);
        for n in 1..=8 {
            for _ in 0..5 {
                let x: Vec<f64> = (0..n).map(|_| rng.next()).collect();
                check_householder(x, 1e-13);
            }
        }
    }
    
    #[test]
    fn householder_real_degenerate() {
        // Tail already zero and α real: identity reflector.
        for x in [vec![3.0], vec![-3.0, 0.0, 0.0], vec![0.0, 0.0], vec![0.0]] {
            let mut v = x.clone();
            let (tau, beta) = householder(&mut v);
            assert_eq!(tau, 0.0);
            assert_eq!(beta, x[0]);
            assert_eq!(v[0], 1.0);
        }
        // Still passes the generic check
        check_householder(vec![0.0, 0.0, 0.0], 1e-14);
        check_householder(vec![-2.0, 0.0], 1e-14);
    }
    
    #[test]
    fn householder_complex_random() {
        let mut rng = Rng(2);
        for n in 1..=8 {
            for _ in 0..5 {
                let x: Vec<Complex<f64>> = (0..n).map(|_| Complex { real: rng.next(), imag: rng.next() }).collect();
                check_householder(x, 1e-13);
            }
        }
    }
    
    #[test]
    fn householder_complex_single_entry_phase() {
        // 1×1 case with complex α: reflector is a phase making β real.
        let x = vec![Complex { real: 0.0, imag: 1.0 }];
        check_householder(x.clone(), 1e-14);
        let mut v = x;
        let (tau, beta) = householder(&mut v);
        assert!(tau.abs() > 0.0);
        assert!((f64::abs(beta) - 1.0).abs() < 1e-14);
    
        check_householder(vec![Complex { real: 0.6, imag: -0.8 }], 1e-14);
    }
    
    #[test]
    fn householder_complex_tail_zero_but_alpha_complex() {
        check_householder(vec![Complex { real: 1.0, imag: 2.0 }, Complex::<f64>::zero(), Complex::<f64>::zero()], 1e-14);
    }
}

#[cfg(test)]
mod apply_left_right_tests {
    use super::super::*;
    use super::support::*;

    /// Embed the k×k matrix `h` into an n×n identity at offset `off`.
    fn embed<T: Scalar>(n: usize, off: usize, h: &Matrix<T>) -> Matrix<T> {
        let mut e = Matrix::<T>::identity(n);
        for i in 0..h.m {
            for j in 0..h.n {
                e.values[(off + i) * n + off + j] = h.values[i * h.n + j];
            }
        }
        e
    }
    
    fn dense_reflector<T: Scalar>(v: &[T], tau: T) -> Matrix<T> {
        let k = v.len();
        let mut vals = vec![T::zero(); k * k];
        for i in 0..k {
            for j in 0..k {
                let id = if i == j { T::one() } else { T::zero() };
                vals[i * k + j] = id.sub(tau.mul(v[i]).mul(v[j].conjugate()));
            }
        }
        Matrix::from(k, k, vals)
    }
    
    #[test]
    fn apply_left_matches_dense_product() {
        let mut rng = Rng(3);
        let (m, n, r0, c0, c1) = (6, 5, 2, 1, 4);
        let a: Matrix<Complex<f64>> = rand_matrix_c(m, n, &mut rng);
        let v: Vec<Complex<f64>> = (0..m - r0).map(|_| Complex::<f64> { real: rng.next(), imag: rng.next() }).collect();
        let tau = Complex { real: 0.7, imag: -0.3 };
    
        let mut got = Matrix::from(m, n, a.values.clone());
        got.apply_left(r0, c0, c1, &v, tau);
    
        let full = embed(m, r0, &dense_reflector(&v, tau)).mul(&a).unwrap();
        for i in 0..m {
            for j in 0..n {
                let want = if (c0..c1).contains(&j) { full.values[i * n + j] } else { a.values[i * n + j] };
                assert!((got.values[i * n + j].sub(want)).abs() < 1e-13, "mismatch at ({i},{j})");
            }
        }
    }
    
    #[test]
    fn apply_right_matches_dense_product() {
        let mut rng = Rng(4);
        let (m, n, r0, r1, c0) = (6, 5, 1, 5, 2);
        let a: Matrix<Complex<f64>> = rand_matrix_c(m, n, &mut rng);
        let v: Vec<Complex<f64>> = (0..n - c0).map(|_| Complex::<f64> { real: rng.next(), imag: rng.next() }).collect();
        let tau = Complex { real: -0.4, imag: 0.9 };
    
        let mut got = Matrix::from(m, n, a.values.clone());
        got.apply_right(r0, r1, c0, &v, tau);
    
        let full = (&a).mul(embed(n, c0, &dense_reflector(&v, tau))).unwrap();
        for i in 0..m {
            for j in 0..n {
                let want = if (r0..r1).contains(&i) { full.values[i * n + j] } else { a.values[i * n + j] };
                assert!((got.values[i * n + j].sub(want)).abs() < 1e-13, "mismatch at ({i},{j})");
            }
        }
    }
}

#[cfg(test)]
mod rot_cols_tests {
    use super::super::*;

    #[test]
    fn rot_cols_is_orthogonal_and_correct() {
        let a = Matrix::from(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let mut b = Matrix::from(2, 3, a.values.clone());
        let (c, s) = (0.6, 0.8);
        b.rot_cols(0, 2, c, s);
        // column 1 untouched
        assert_eq!(b.values[1], 2.0);
        assert_eq!(b.values[4], 5.0);
        // new col0 = c*col0 + s*col2, new col2 = -s*col0 + c*col2
        assert!(f64::abs(b.values[0] - (0.6 * 1.0 + 0.8 * 3.0)) < 1e-15);
        assert!((b.values[2] - (-0.8 * 1.0 + 0.6 * 3.0)).abs() < 1e-15);
        assert!((b.values[3] - (0.6 * 4.0 + 0.8 * 6.0)).abs() < 1e-15);
        assert!((b.values[5] - (-0.8 * 4.0 + 0.6 * 6.0)).abs() < 1e-15);
        // Frobenius norm preserved
        let na: f64 = a.values.iter().map(|x| x * x).sum();
        let nb: f64 = b.values.iter().map(|x| x * x).sum();
        assert!((na - nb).abs() < 1e-13);
    }
}

#[cfg(test)]
mod bidiagonalization_tests {
    use super::super::*;
    use super::support::*;

    fn check_bidiagonalize<T: Scalar>(a: &Matrix<T>, tol: T::UnderlyingReal) {
        let (m, n) = (a.m, a.n);
        let (u, d, e, v) = a.clone().bidiagonalize().unwrap();
        assert_eq!((u.m, u.n), (m, n));
        assert_eq!((v.m, v.n), (n, n));
        assert_eq!(d.len(), n);
        assert_eq!(e.len(), n.saturating_sub(1));
    
        let mut bv = vec![T::zero(); n * n];
        for i in 0..n {
            bv[i * n + i] = T::from_real(d[i]);
            if i + 1 < n {
                bv[i * n + i + 1] = T::from_real(e[i]);
            }
        }
        let b = Matrix::from(n, n, bv);
        let recon = (&u).mul(b).unwrap().mul(v.transpose_conjugate()).unwrap();
        let scale = utils::max(utils::max_abs_of(a.iter()), T::UnderlyingReal::one());
        let err = max_diff(&recon, a);
        assert!(err <= tol.mul(scale), "U1·B·V1ᴴ ≠ A: {err:?}");
        assert_orthonormal_cols(&u);
        assert_orthonormal_cols(&v);
        // V1 is square, so also V1 V1ᴴ = I
        let vvh = (&v).mul(v.transpose_conjugate()).unwrap();
        assert!(max_diff(&vvh, &Matrix::identity(n)) <= tol);
    }
    
    #[test]
    fn bidiagonalize_real_shapes() {
        let mut rng = Rng(5);
        for &(m, n) in &[(1, 1), (2, 1), (5, 1), (2, 2), (3, 3), (6, 4), (5, 5), (9, 3), (12, 12)] {
            let a: Matrix<f64> = rand_matrix(m, n, &mut rng);
            check_bidiagonalize(&a, 1e-13);
        }
    }
    
    #[test]
    fn bidiagonalize_complex_shapes() {
        let mut rng = Rng(6);
        for &(m, n) in &[(1, 1), (3, 1), (2, 2), (4, 3), (6, 6), (8, 5)] {
            let a: Matrix<Complex<f64>> = rand_matrix_c(m, n, &mut rng);
            check_bidiagonalize(&a, 1e-13);
        }
    }
    
    #[test]
    fn bidiagonalize_preserves_frobenius_norm() {
        let mut rng = Rng(7);
        let a: Matrix<f64> = rand_matrix(7, 4, &mut rng);
        let (_, d, e, _) = a.clone().bidiagonalize().unwrap();
        let fa: f64 = a.values.iter().map(|x| x * x).sum();
        let fb: f64 = d.iter().chain(e.iter()).map(|x| x * x).sum();
        assert!((fa - fb).abs() < 1e-12 * fa);
    }
    
    #[test]
    fn bidiagonalize_already_bidiagonal() {
        let a = Matrix::from(3, 3, vec![2.0, 1.0, 0.0, 0.0, 3.0, 4.0, 0.0, 0.0, 5.0]);
        check_bidiagonalize(&a, 1e-14);
    }
}

#[cfg(test)]
mod bidiagonal_qr_tests {
    use super::super::*;
    use super::support::*;

    fn check_bidiagonal_qr(d0: Vec<f64>, e0: Vec<f64>) {
        let n = d0.len();
        let mut bv = vec![0.0; n * n];
        for i in 0..n {
            bv[i * n + i] = d0[i];
            if i + 1 < n {
                bv[i * n + i + 1] = e0[i];
            }
        }
        let b = Matrix::from(n, n, bv);
    
        let (mut d, mut e) = (d0.clone(), e0.clone());
        let (mut u, mut v) = (Matrix::<f64>::identity(n), Matrix::<f64>::identity(n));
        Matrix::<f64>::bidiagonal_qr(&mut d, &mut e, &mut u, &mut v).expect("converge");
    
        // B = U diag(d) Vᵀ
        let recon = (&u).mul(Matrix::diag(&d)).unwrap().mul(v.transpose_conjugate()).unwrap();
        let scale = utils::max_abs_of(b.iter()).max(1.0);
        assert!(max_diff(&recon, &b) < 1e-13 * scale, "reconstruction failed");
        assert_orthonormal_cols(&u);
        assert_orthonormal_cols(&v);
    
        // singular values agree with the (independent) full driver applied to B
        let mut got: Vec<f64> = d.iter().map(|x| x.abs()).collect();
        got.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let want = b.svd().unwrap().s;
        assert_close_vec(&got, &want, 1e-12 * scale);
    }
    
    #[test]
    fn bidiagonal_qr_random() {
        let mut rng = Rng(8);
        for &n in &[2usize, 3, 4, 6, 10, 25] {
            let d: Vec<f64> = (0..n).map(|_| rng.next() * 3.0).collect();
            let e: Vec<f64> = (0..n - 1).map(|_| rng.next() * 3.0).collect();
            check_bidiagonal_qr(d, e);
        }
    }
    
    #[test]
    fn bidiagonal_qr_single_element() {
        check_bidiagonal_qr(vec![-5.0], vec![]);
    }
    
    #[test]
    fn bidiagonal_qr_already_diagonal() {
        check_bidiagonal_qr(vec![3.0, -1.0, 2.0], vec![0.0, 0.0]);
    }
    
    #[test]
    fn bidiagonal_qr_zero_diagonal_first() {
        check_bidiagonal_qr(vec![0.0, 2.0, 3.0], vec![1.0, 1.0]);
    }
    
    #[test]
    fn bidiagonal_qr_zero_diagonal_interior() {
        check_bidiagonal_qr(vec![1.0, 0.0, 2.0, 3.0], vec![1.0, 1.0, 1.0]);
        check_bidiagonal_qr(vec![2.0, 1.0, 0.0, 0.0, 4.0], vec![1.0, 2.0, 3.0, 1.0]);
    }
    
    #[test]
    fn bidiagonal_qr_zero_diagonal_last() {
        check_bidiagonal_qr(vec![2.0, 3.0, 1.0, 0.0], vec![1.0, 1.0, 1.0]);
    }
    
    #[test]
    fn bidiagonal_qr_all_zero_diagonal() {
        check_bidiagonal_qr(vec![0.0, 0.0, 0.0], vec![1.0, 1.0]);
    }
    
    #[test]
    fn bidiagonal_qr_split_blocks() {
        // e[2] = 0 splits the problem in two independent blocks
        check_bidiagonal_qr(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![1.0, 1.0, 0.0, 1.0]);
    }
    
    #[test]
    fn bidiagonal_qr_graded() {
        check_bidiagonal_qr(vec![1.0, 1e-3, 1e-6, 1e-9], vec![1e-1, 1e-4, 1e-7]);
    }
}

#[cfg(test)]
mod svd_tests {
    use super::super::*;
    use super::support::*;

    /// Full validation of a computed SVD of `a`. `tol` is relative to max(1, ‖A‖max).
    fn check_svd<T: Scalar>(a: &Matrix<T>, tol: T::UnderlyingReal) -> Svd<T> {
        let r = a.svd().expect("SVD should converge");
        let k = a.m.min(a.n);
    
        // shapes
        assert_eq!((r.u.m, r.u.n), (a.m, k), "U shape");
        assert_eq!((r.v.m, r.v.n), (a.n, k), "V shape");
        assert_eq!(r.s.len(), k, "s length");
    
        // nonnegative and sorted descending
        for i in 0..k {
            assert!(r.s[i] >= T::UnderlyingReal::zero(), "s[{i}] is negative");
            if i + 1 < k {
                assert!(r.s[i] >= r.s[i + 1], "s not sorted at {i}");
            }
        }
    
        let scale = utils::max(utils::max_abs_of(a.iter()), T::UnderlyingReal::one());
        let atol = tol.mul(scale);
    
        // reconstruction A ≈ U diag(s) Vᴴ
        let recon = (&r.u).mul(Matrix::diag(&r.s)).unwrap().mul(r.v.transpose_conjugate()).unwrap();
        let err = max_diff(&recon, a);
        assert!(err <= atol, "reconstruction error {err:?} > {atol:?}");
    
        // orthonormality
        assert_orthonormal_cols(&r.u);
        assert_orthonormal_cols(&r.v);
        r
    }
    
    /// Householder reflector I − 2 w wᵀ / wᵀw as a dense real n×n matrix.
    fn reflector(w: &[f64]) -> Matrix<f64> {
        let n = w.len();
        let ww: f64 = w.iter().map(|x| x * x).sum();
        let mut values = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                values[i * n + j] = (if i == j { 1.0 } else { 0.0 }) - 2.0 * w[i] * w[j] / ww;
            }
        }
        Matrix::from(n, n, values)
    }
    
    // ===========================================================================
    // Known answers
    // ===========================================================================
    
    #[test]
    fn svd_known_2x2() {
        // AᵀA = [[25,20],[20,25]] has eigenvalues 45 and 5.
        let a = Matrix::from(2, 2, vec![3.0, 0.0, 4.0, 5.0]);
        let r = check_svd(&a, 1e-13);
        assert_close_vec(&r.s, &[45f64.sqrt(), 5f64.sqrt()], 1e-13);
    }
    
    #[test]
    fn svd_diagonal_with_negative_entry() {
        let a = Matrix::from(2, 2, vec![3.0, 0.0, 0.0, -2.0]);
        let r = check_svd(&a, 1e-14);
        assert_close_vec(&r.s, &[3.0, 2.0], 1e-14);
    }
    
    #[test]
    fn svd_identity() {
        let a = Matrix::<f64>::identity(5);
        let r = check_svd(&a, 1e-14);
        assert_close_vec(&r.s, &[1.0; 5], 1e-14);
    }
    
    #[test]
    fn svd_scaled_identity_repeated_values() {
        let mut a = Matrix::<f64>::identity(4);
        for x in a.values.iter_mut() {
            *x *= 2.0;
        }
        let r = check_svd(&a, 1e-14);
        assert_close_vec(&r.s, &[2.0; 4], 1e-14);
    }
    
    #[test]
    fn svd_1x1() {
        let r = check_svd(&Matrix::from(1, 1, vec![-7.5]), 1e-15);
        assert_close_vec(&r.s, &[7.5], 1e-15);
    }
    
    #[test]
    fn svd_column_vector() {
        // n = 1: single singular value = ‖x‖
        let a = Matrix::from(4, 1, vec![1.0, 2.0, 3.0, 4.0]);
        let r = check_svd(&a, 1e-14);
        assert_close_vec(&r.s, &[30f64.sqrt()], 1e-14);
    }
    
    #[test]
    fn svd_row_vector() {
        let a = Matrix::from(1, 4, vec![1.0, 2.0, 3.0, 4.0]);
        let r = check_svd(&a, 1e-14);
        assert_close_vec(&r.s, &[30f64.sqrt()], 1e-14);
    }
    
    #[test]
    fn svd_zero_matrix() {
        let a = Matrix::from(4, 3, vec![0.0; 12]);
        let r = check_svd(&a, 1e-14);
        assert_close_vec(&r.s, &[0.0; 3], 0.0);
    }
    
    #[test]
    fn svd_zero_columns() {
        let a = Matrix::from(0usize.max(3), 0, Vec::<f64>::new());
        let r = a.svd().unwrap();
        assert_eq!((r.u.m, r.u.n), (3, 0));
        assert!(r.s.is_empty());
        assert_eq!((r.v.m, r.v.n), (0, 0));
    }
    
    #[test]
    fn svd_rank_one() {
        // ones(4,4): σ = (4, 0, 0, 0)
        let a = Matrix::from(4, 4, vec![1.0; 16]);
        let r = check_svd(&a, 1e-13);
        assert_close_vec(&r.s, &[4.0, 0.0, 0.0, 0.0], 1e-13);
    }
    
    #[test]
    fn svd_rank_deficient_duplicate_columns() {
        let mut rng = Rng(9);
        let base: Matrix<f64> = rand_matrix(7, 3, &mut rng);
        // columns: c0, c1, c0 + c1  -> rank 2
        let mut vals = vec![0.0; 7 * 3];
        for i in 0..7 {
            let (x, y) = (base.values[i * 3], base.values[i * 3 + 1]);
            vals[i * 3] = x;
            vals[i * 3 + 1] = y;
            vals[i * 3 + 2] = x + y;
        }
        let r = check_svd(&Matrix::from(7, 3, vals), 1e-13);
        assert!(r.s[2] < 1e-13, "smallest singular value should vanish");
        assert!(r.s[1] > 1e-3);
    }
    
    #[test]
    fn svd_upper_triangular_with_zero_diagonal_entries() {
        // exercises the zero-diagonal chase in both positions
        check_svd(&Matrix::from(3, 3, vec![1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 2.0]), 1e-13);
        check_svd(&Matrix::from(3, 3, vec![2.0, 1.0, 0.0, 0.0, 3.0, 1.0, 0.0, 0.0, 0.0]), 1e-13);
        check_svd(&Matrix::from(3, 3, vec![0.0, 1.0, 0.0, 0.0, 2.0, 1.0, 0.0, 0.0, 3.0]), 1e-13);
    }
    
    #[test]
    fn svd_recovers_prescribed_singular_values() {
        // A = U0 · diag(s0) · V0ᵀ with U0, V0 products of Householder reflectors.
        let cases: Vec<(usize, usize, Vec<f64>)> = vec![
            (6, 4, vec![10.0, 5.0, 1.0, 1e-3]),
            (4, 6, vec![7.0, 3.0, 2.0, 0.5]),
            (5, 5, vec![2.0, 2.0, 2.0, 1.0, 1.0]),
            (8, 3, vec![1.0, 1e-4, 1e-8]),
        ];
        for (m, n, s0) in cases {
            let k = m.min(n);
            let u0 = reflector(&(0..m).map(|i| 1.0 + i as f64).collect::<Vec<_>>())
                .mul(reflector(&(0..m).map(|i| (i as f64 * 1.3).sin() + 0.2).collect::<Vec<_>>()))
                .unwrap();
            let v0 = reflector(&(0..n).map(|i| (i as f64 * 0.7).cos() + 1.5).collect::<Vec<_>>())
                .mul(reflector(&(0..n).map(|i| 2.0 - i as f64 * 0.3).collect::<Vec<_>>()))
                .unwrap();
            // A = U0[:, :k] diag(s0) V0[:, :k]ᵀ
            let mut a = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    a[i * n + j] = (0..k).map(|l| u0.values[i * m + l] * s0[l] * v0.values[j * n + l]).sum();
                }
            }
            let a = Matrix::from(m, n, a);
            let r = check_svd(&a, 1e-12);
            assert_close_vec(&r.s, &s0, 1e-12 * s0[0]);
        }
    }
    
    // ===========================================================================
    // Random matrices, all shapes
    // ===========================================================================
    
    #[test]
    fn svd_random_real_many_shapes() {
        let mut rng = Rng(10);
        for &(m, n) in &[
            (2, 2), (3, 2), (2, 3), (4, 4), (7, 4), (4, 7), (10, 10), (20, 5), (5, 20), (30, 20), (20, 30), (40, 40),
        ] {
            let a: Matrix<f64> = rand_matrix(m, n, &mut rng);
            check_svd(&a, 1e-12);
        }
    }
    
    #[test]
    fn svd_random_complex_many_shapes() {
        let mut rng = Rng(11);
        for &(m, n) in &[(1, 1), (2, 2), (5, 3), (3, 5), (8, 8), (15, 6), (6, 15), (20, 20)] {
            let a: Matrix<Complex<f64>> = rand_matrix_c(m, n, &mut rng);
            check_svd(&a, 1e-12);
        }
    }
    
    #[test]
    fn svd_many_small_random_stress() {
        let mut rng = Rng(13);
        for trial in 0..200 {
            let m = 1 + (trial % 7);
            let n = 1 + ((trial / 7) % 7);
            let a: Matrix<f64> = rand_matrix(m, n, &mut rng);
            check_svd(&a, 1e-12);
        }
    }
    
    #[test]
    fn svd_random_low_rank_and_structured_zeros() {
        let mut rng = Rng(14);
        // rank-2 product of thin factors
        let l: Matrix<f64> = rand_matrix(9, 2, &mut rng);
        let rr: Matrix<f64> = rand_matrix(2, 6, &mut rng);
        let a = l.mul(rr).unwrap();
        let r = check_svd(&a, 1e-12);
        let s = r.s;
        assert!(s[1] > 1e-3);
        for &x in &s[2..] {
            assert!(x < 1e-12, "expected numerically zero singular value, got {x:?}");
        }
    
        // matrix with a zero row and zero column
        let mut b: Matrix<f64> = rand_matrix(5, 5, &mut rng);
        for j in 0..5 {
            b.values[2 * 5 + j] = 0.0;
        }
        for i in 0..5 {
            b.values[i * 5 + 3] = 0.0;
        }
        check_svd(&b, 1e-12);
    }
    
    // ===========================================================================
    // svd: complex-specific
    // ===========================================================================
    
    #[test]
    fn svd_complex_diagonal_known() {
        let a = Matrix::from(2, 2, vec![Complex { real: 0.0, imag: 1.0 }, Complex::<f64>::zero(), Complex::<f64>::zero(), Complex { real: 2.0, imag: 0.0 }]);
        let r = check_svd(&a, 1e-14);
        assert_close_vec(&r.s, &[2.0, 1.0], 1e-14);
    }
    
    #[test]
    fn svd_complex_unitary_has_unit_singular_values() {
        let phases = [0.3f64, 1.1, -2.0];
        let mut vals = vec![Complex::<f64>::zero(); 9];
        for (i, &p) in phases.iter().enumerate() {
            vals[i * 3 + i] = Complex { real: p.cos(), imag: p.sin() };
        }
        let r = check_svd(&Matrix::from(3, 3, vals), 1e-14);
        assert_close_vec(&r.s, &[1.0; 3], 1e-14);
    }
    
    #[test]
    fn svd_complex_scalar_phase_times_real_matrix_keeps_singular_values() {
        let mut rng = Rng(15);
        let a: Matrix<f64> = rand_matrix(6, 4, &mut rng);
        let phase = Complex { real: (0.8f64).cos(), imag: (0.8f64).sin() };
        let ac = Matrix::from(6, 4, a.values.iter().map(|&x| Complex::<f64>::from_real(x).mul(phase)).collect());
        let s_real = check_svd(&a, 1e-12).s;
        let s_cplx = check_svd(&ac, 1e-12).s;
        assert_close_vec(&s_cplx, &s_real, 1e-12);
    }
    
    // ===========================================================================
    // svd: relations between problems
    // ===========================================================================
    
    #[test]
    fn svd_singular_values_of_transpose_and_conjugate_transpose_agree() {
        let mut rng = Rng(16);
        let a: Matrix<Complex<f64>> = rand_matrix_c(7, 4, &mut rng);
        let s1 = check_svd(&a, 1e-12).s;
        let s2 = check_svd(&a.transpose_conjugate(), 1e-12).s;
        assert_close_vec(&s1, &s2, 1e-12);
    }
    
    #[test]
    fn svd_singular_values_scale_linearly() {
        let mut rng = Rng(17);
        let a: Matrix<f64> = rand_matrix(6, 5, &mut rng);
        let b = Matrix::from(6, 5, a.values.iter().map(|x| x * 3.5).collect());
        let s1 = a.svd().unwrap().s;
        let s2 = b.svd().unwrap().s;
        let scaled: Vec<f64> = s1.iter().map(|x| x * 3.5).collect();
        assert_close_vec(&s2, &scaled, 1e-12);
    }
    
    #[test]
    fn svd_invariant_under_orthogonal_transformations() {
        let mut rng = Rng(18);
        let a: Matrix<f64> = rand_matrix(6, 4, &mut rng);
        let q = reflector(&[1.0, -2.0, 0.5, 3.0, 1.0, -1.0]);
        let z = reflector(&[0.3, 1.0, -1.0, 2.0]);
        let b = q.mul(&a).unwrap().mul(z).unwrap();
        assert_close_vec(&b.svd().unwrap().s, &a.svd().unwrap().s, 1e-12);
    }
    
    #[test]
    fn svd_frobenius_norm_equals_root_sum_squares_of_singular_values() {
        let mut rng = Rng(19);
        let a: Matrix<Complex<f64>> = rand_matrix_c(9, 7, &mut rng);
        let fro: f64 = a.values.iter().map(|x| x.abs().powi(2)).sum::<f64>().sqrt();
        let s = a.svd().unwrap().s;
        let ss: f64 = s.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((fro - ss).abs() < 1e-12 * fro);
    }
    
    #[test]
    fn svd_largest_singular_value_is_spectral_norm_bound() {
        // ‖Ax‖ ≤ σ₁‖x‖ for random x, and σ₁ ≥ ‖A e_j‖ for every column
        let mut rng = Rng(20);
        let a: Matrix<f64> = rand_matrix(8, 5, &mut rng);
        let s1 = a.svd().unwrap().s[0];
        for _ in 0..20 {
            let x: Vec<f64> = (0..5).map(|_| rng.next()).collect();
            let xn: f64 = x.iter().map(|v| v * v).sum::<f64>().sqrt();
            let ax = (&a).mul(Matrix::from(5, 1, x)).unwrap();
            let axn: f64 = ax.values.iter().map(|v| v * v).sum::<f64>().sqrt();
            assert!(axn <= s1 * xn * (1.0 + 1e-12));
        }
        for j in 0..5 {
            let cn: f64 = (0..8).map(|i| a.values[i * 5 + j].powi(2)).sum::<f64>().sqrt();
            assert!(s1 >= cn * (1.0 - 1e-12));
        }
    }
    
    #[test]
    fn svd_does_not_modify_input() {
        let mut rng = Rng(21);
        let a: Matrix<f64> = rand_matrix(5, 3, &mut rng);
        let copy = a.values.clone();
        let _ = a.svd().unwrap();
        assert_eq!(a.values, copy);
    }
    
    #[test]
    fn svd_small_singular_values_not_destroyed() {
        // Diagonal-dominant graded matrix: singular values span 12 orders of magnitude.
        // Absolute accuracy O(ε σ₁) is guaranteed; check that.
        let d = [1.0, 1e-4, 1e-8, 1e-12];
        let mut vals = vec![0.0; 16];
        for i in 0..4 {
            vals[i * 4 + i] = d[i];
            if i + 1 < 4 {
                vals[i * 4 + i + 1] = d[i] * 0.1;
            }
        }
        let a = Matrix::from(4, 4, vals);
        let r = check_svd(&a, 1e-13);
        let s = r.s;
        assert!(s[0] > 0.9 && s[0] < 1.1);
        assert!(s[3] < 1e-10);
    }
}