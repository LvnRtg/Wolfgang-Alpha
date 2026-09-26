//! These tests focus on properties rather than on exact decompositions since
//! Schur decompositions are generally not unique.

#[cfg(test)]
mod support {
    use crate::math::{Complex, Matrix};
    use crate::math::traits::*;

    pub const TOL: f64 = 1e-6;

    pub fn is_orthogonal(u: &Matrix<f64>) -> bool {
        let n = u.m();
        u.n() == n && (&u).mul(u.transpose()).unwrap().approx_eq(&Matrix::identity(n))
    }

    pub fn is_unitary(q: &Matrix<Complex<f64>>) -> bool {
        let n = q.m();
        q.n() == n && (&q).mul(q.transpose().conjugate()).unwrap().approx_eq(&Matrix::identity(n))
    }

    /// Upper Hessenberg, with no two consecutive nonzero sub-diagonal entries
    /// (i.e. no unreduced block bigger than 2x2).
    pub fn is_quasi_upper_triangular(h: &Matrix<f64>) -> bool {
        let n = h.m();
        if h.n() != n {
            return false;
        }
        for i in 2..n {
            for j in 0..(i - 1) {
                if h.get(i, j).abs() > TOL {
                    return false;
                }
            }
        }
        let mut i = 0;
        while i + 1 < n {
            if h.get(i+1, i).abs() > TOL {
                if i + 2 < n && h.get(i+2, i+1).abs() > TOL {
                    return false; // adjacent/overlapping blocks: not a valid quasi-triangular form
                }
                i += 2;
            } else {
                i += 1;
            }
        }
        true
    }

    pub fn is_upper_triangular_c(t: &Matrix<Complex<f64>>) -> bool {
        let n = t.m();
        if t.n() != n {
            return false;
        }
        (1..n).all(|i| {
            (0..i).all(|j| {
                let z = t.get(i, j);
                (z.real * z.real + z.imag * z.imag).sqrt() <= TOL
            })
        })
    }

    /// Extracts the (possibly complex) eigenvalues sitting on the diagonal /
    /// 2x2 blocks of a quasi-upper-triangular real matrix.
    pub fn extract_eigs_from_h(h: &Matrix<f64>) -> Vec<Complex<f64>> {
        let n = h.m();
        let mut eigs = Vec::with_capacity(n);
        let mut i = 0;
        while i < n {
            if i + 1 < n && h.get(i+1, i).abs() > TOL {
                let a = h.get(i, i);
                let b = h.get(i, i+1);
                let c = h.get(i+1, i);
                let d = h.get(i+1, i+1);
                let tr = a + d;
                let det = a * d - b * c;
                let disc = tr * tr - 4.0 * det;
                if disc >= 0.0 {
                    let s = disc.sqrt();
                    eigs.push(Complex {real: (tr + s) / 2.0, imag: 0.0});
                    eigs.push(Complex {real: (tr - s) / 2.0, imag: 0.0});
                } else {
                    let s = (-disc).sqrt();
                    eigs.push(Complex {real: tr / 2.0, imag: s / 2.0});
                    eigs.push(Complex {real: tr / 2.0, imag: -s / 2.0});
                }
                i += 2;
            } else {
                eigs.push(Complex {real: h.get(i, i), imag: 0.0});
                i += 1;
            }
        }
        eigs
    }

    pub fn diag_c(t: &Matrix<Complex<f64>>) -> Vec<Complex<f64>> {
        (0..t.m()).map(|i| t.get(i, i)).collect()
    }

    /// Order-independent comparison of two multisets of (possibly complex) eigenvalues.
    pub fn eig_sets_match(a: &[Complex<f64>], b: &[Complex<f64>]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut remaining: Vec<Complex<f64>> = b.to_vec();
        for x in a {
            let pos = remaining.iter().position(|y| {
                let d = (*x).sub(*y);
                (d.real * d.real + d.imag * d.imag).sqrt() <= TOL
            });
            match pos {
                Some(idx) => {
                    remaining.remove(idx);
                }
                None => return false,
            }
        }
        true
    }
}

#[cfg(test)]
mod real_schur_decomposition_tests {
    use super::support::*;
    use crate::math::{Complex, Matrix};
    use crate::math::traits::*;

    #[test]
    fn non_square_returns_none() {
        let m = Matrix::from(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(m.real_schur_decomposition().is_none());

        let m2 = Matrix::from(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(m2.real_schur_decomposition().is_none());
    }

    #[test]
    fn one_by_one_matrix() {
        let m = Matrix::from(1, 1, vec![7.0]);
        let (u, h) = m.real_schur_decomposition().unwrap();
        assert!(is_orthogonal(&u));
        assert!(is_quasi_upper_triangular(&h));
        let reconstructed = (&u).mul(&h).unwrap().mul(u.transpose()).unwrap();
        assert!(reconstructed.to_complex().approx_eq(&m.to_complex()));
        assert!((h.get(0, 0) - 7.0).abs() < TOL);
    }

    #[test]
    fn identity_matrix_reconstructs() {
        let m = Matrix::from(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let (u, h) = m.real_schur_decomposition().unwrap();
        assert!(is_orthogonal(&u));
        assert!(is_quasi_upper_triangular(&h));
        let reconstructed = (&u).mul(&h).unwrap().mul(u.transpose()).unwrap();
        assert!(reconstructed.to_complex().approx_eq(&m.to_complex()));
    }

    #[test]
    fn symmetric_matrix_has_no_2x2_blocks() {
        // Symmetric real matrices have strictly real eigenvalues, so a valid
        // (if not required) Schur form here is fully upper triangular.
        let m = Matrix::from(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (u, h) = m.real_schur_decomposition().unwrap();
        assert!(is_orthogonal(&u));
        assert!(is_quasi_upper_triangular(&h));
        let reconstructed = (&u).mul(&h).unwrap().mul(u.transpose()).unwrap();
        assert!(reconstructed.to_complex().approx_eq(&m.to_complex()));

        let mut eigs = extract_eigs_from_h(&h);
        eigs.sort_by(|a, b| a.real.partial_cmp(&b.real).unwrap());
        assert!((eigs[0].real - 1.0).abs() < TOL && eigs[0].imag.abs() < TOL);
        assert!((eigs[1].real - 3.0).abs() < TOL && eigs[1].imag.abs() < TOL);
    }

    #[test]
    fn rotation_matrix_keeps_undeflated_2x2_block() {
        // eigenvalues of [[0,-1],[1,0]] are +/- i: cannot be made triangular over the reals.
        let m = Matrix::from(2, 2, vec![0.0, -1.0, 1.0, 0.0]);
        let (u, h) = m.real_schur_decomposition().unwrap();
        assert!(is_orthogonal(&u));
        assert!(is_quasi_upper_triangular(&h));
        assert!(h.get(1, 0).abs() > TOL, "expected an undeflated 2x2 block");
        let reconstructed = (&u).mul(&h).unwrap().mul(u.transpose()).unwrap();
        assert!(reconstructed.to_complex().approx_eq(&m.to_complex()));

        let eigs = extract_eigs_from_h(&h);
        assert!(eig_sets_match(
            &eigs,
            &[Complex {real: 0.0, imag: 1.0}, Complex {real: 0.0, imag: -1.0}]
        ));
    }

    #[test]
    fn jordan_block_reconstructs() {
        let m = Matrix::from(2, 2, vec![2.0, 1.0, 0.0, 2.0]);
        let (u, h) = m.real_schur_decomposition().unwrap();
        assert!(is_orthogonal(&u));
        assert!(is_quasi_upper_triangular(&h));
        let reconstructed = (&u).mul(&h).unwrap().mul(u.transpose()).unwrap();
        assert!(reconstructed.to_complex().approx_eq(&m.to_complex()));
        assert!((h.get(0, 0) + h.get(1, 1) - 4.0).abs() < TOL); // trace invariant
    }

    #[test]
    fn general_3x3_matrix_reconstructs() {
        let m = Matrix::from(
            3,
            3,
            vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 6.0],
        );
        let (u, h) = m.real_schur_decomposition().unwrap();
        assert!(is_orthogonal(&u));
        assert!(is_quasi_upper_triangular(&h));
        let reconstructed = (&u).mul(&h).unwrap().mul(u.transpose()).unwrap();
        assert!(reconstructed.to_complex().approx_eq(&m.to_complex()));

        let trace_h: f64 = (0..3).map(|i| h.get(i, i)).sum();
        assert!((trace_h - 10.0).abs() < TOL); // 1 + 3 + 6
    }

    #[test]
    fn two_conjugate_pairs_4x4_produces_two_blocks() {
        let m = Matrix::from(
            4,
            4,
            vec![
                0.0, -4.0, 0.0, 0.0, //
                1.0, 0.0, 0.0, 0.0, //
                0.0, 0.0, 0.0, -9.0, //
                0.0, 0.0, 1.0, 0.0, //
            ],
        );
        let (u, h) = m.real_schur_decomposition().unwrap();
        assert!(is_orthogonal(&u));
        assert!(is_quasi_upper_triangular(&h));
        let reconstructed = (&u).mul(&h).unwrap().mul(u.transpose()).unwrap();
        assert!(reconstructed.to_complex().approx_eq(&m.to_complex()));

        let mut block_count = 0;
        let mut i = 0;
        while i + 1 < 4 {
            if h.get(i+1, i).abs() > TOL {
                block_count += 1;
                i += 2;
            } else {
                i += 1;
            }
        }
        assert_eq!(block_count, 2, "expected exactly two undeflated 2x2 blocks");

        let eigs = extract_eigs_from_h(&h);
        assert!(eig_sets_match(
            &eigs,
            &[
                Complex {real: 0.0, imag: 2.0},
                Complex {real: 0.0, imag: -2.0},
                Complex {real: 0.0, imag: 3.0},
                Complex {real: 0.0, imag: -3.0},
            ]
        ));
    }
}

#[cfg(test)]
mod complex_schur_decomposition_tests {
    use super::support::*;
    use crate::math::{Complex, Matrix};
    use crate::math::traits::*;

    #[test]
    fn non_square_returns_none() {
        let m = Matrix::from(2, 3, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(m.complex_schur_decomposition().is_none());

        let m2 = Matrix::from(3, 2, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert!(m2.complex_schur_decomposition().is_none());
    }

    #[test]
    fn one_by_one_matrix() {
        let m = Matrix::from(1, 1, vec![7.0]);
        let (q, t) = m.complex_schur_decomposition().unwrap();
        assert!(is_unitary(&q));
        assert!(is_upper_triangular_c(&t));
        let reconstructed = (&q).mul(&t).unwrap().mul(q.transpose().transform(|z| z.conjugate())).unwrap();
        assert!(reconstructed.approx_eq(&m.to_complex()));
        assert!((t.get(0, 0).real - 7.0).abs() < TOL && t.get(0, 0).imag.abs() < TOL);
    }

    #[test]
    fn identity_matrix_reconstructs() {
        let m = Matrix::from(3, 3, vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
        let (q, t) = m.complex_schur_decomposition().unwrap();
        assert!(is_unitary(&q));
        assert!(is_upper_triangular_c(&t));
        let reconstructed = (&q).mul(&t).unwrap().mul(q.transpose().transform(|z| z.conjugate())).unwrap();
        assert!(reconstructed.approx_eq(&m.to_complex()));
    }

    #[test]
    fn symmetric_matrix_has_real_diagonal() {
        let m = Matrix::from(2, 2, vec![2.0, 1.0, 1.0, 2.0]);
        let (q, t) = m.complex_schur_decomposition().unwrap();
        assert!(is_unitary(&q));
        assert!(is_upper_triangular_c(&t));
        let reconstructed = (&q).mul(&t).unwrap().mul(q.transpose().transform(|z| z.conjugate())).unwrap();
        assert!(reconstructed.approx_eq(&m.to_complex()));

        let mut eigs = diag_c(&t);
        for e in &eigs {
            assert!(e.imag.abs() < TOL);
        }
        eigs.sort_by(|a, b| a.real.partial_cmp(&b.real).unwrap());
        assert!((eigs[0].real - 1.0).abs() < TOL);
        assert!((eigs[1].real - 3.0).abs() < TOL);
    }

    #[test]
    fn rotation_matrix_gives_imaginary_eigenvalues() {
        let m = Matrix::from(2, 2, vec![0.0, -1.0, 1.0, 0.0]);
        let (q, t) = m.complex_schur_decomposition().unwrap();
        assert!(is_unitary(&q));
        assert!(is_upper_triangular_c(&t));
        let reconstructed = (&q).mul(&t).unwrap().mul(q.transpose().transform(|z| z.conjugate())).unwrap();
        assert!(reconstructed.approx_eq(&m.to_complex()));

        let eigs = diag_c(&t);
        assert!(eig_sets_match(
            &eigs,
            &[Complex {real: 0.0, imag: 1.0}, Complex {real: 0.0, imag: -1.0}]
        ));
    }

    #[test]
    fn general_3x3_matrix_reconstructs_and_preserves_trace() {
        let m = Matrix::from(
            3,
            3,
            vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 6.0],
        );
        let (q, t) = m.complex_schur_decomposition().unwrap();
        assert!(is_unitary(&q));
        assert!(is_upper_triangular_c(&t));
        let reconstructed = (&q).mul(&t).unwrap().mul(q.transpose().transform(|z| z.conjugate())).unwrap();
        assert!(reconstructed.approx_eq(&m.to_complex()));

        let trace_t: f64 = diag_c(&t).iter().map(|z| z.real).sum();
        assert!((trace_t - 10.0).abs() < TOL); // 1 + 3 + 6
        let trace_im: f64 = diag_c(&t).iter().map(|z| z.imag).sum();
        assert!(trace_im.abs() < TOL); // real matrix -> eigenvalues sum to a real number
    }

    #[test]
    fn two_conjugate_pairs_4x4() {
        let m = Matrix::from(
            4,
            4,
            vec![
                0.0, -4.0, 0.0, 0.0,
                1.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, -9.0,
                0.0, 0.0, 1.0, 0.0
            ],
        );
        let (q, t) = m.complex_schur_decomposition().unwrap();
        assert!(is_unitary(&q));
        assert!(is_upper_triangular_c(&t));
        let reconstructed = (&q).mul(&t).unwrap().mul(q.transpose().transform(|z| z.conjugate())).unwrap();
        assert!(reconstructed.approx_eq(&m.to_complex()));

        let eigs = diag_c(&t);
        assert!(eig_sets_match(
            &eigs,
            &[
                Complex {real: 0.0, imag: 2.0},
                Complex {real: 0.0, imag: -2.0},
                Complex {real: 0.0, imag: 3.0},
                Complex {real: 0.0, imag: -3.0},
            ]
        ));
    }
}

/// Cross-checks between the two decompositions: the eigenvalues implied by the
/// quasi-triangular real Schur form of a real matrix must match the diagonal of
/// its complex Schur form.
#[cfg(test)]
mod cross_decomposition_consistency_tests {
    use super::support::*;
    use crate::math::{Complex, Matrix};

    #[test]
    fn eigenvalues_agree_between_real_and_complex_schur_forms() {
        let matrices: Vec<(usize, Vec<f64>)> = vec![
            (2, vec![0.0, -1.0, 1.0, 0.0]),
            (2, vec![2.0, 1.0, 1.0, 2.0]),
            (3, vec![1.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0, 6.0]),
            (
                4,
                vec![
                    0.0, -4.0, 0.0, 0.0,
                    1.0, 0.0, 0.0, 0.0,
                    0.0, 0.0, 0.0, -9.0,
                    0.0, 0.0, 1.0, 0.0
                ],
            ),
        ];

        for (n, vals) in matrices {
            let m = Matrix::from(n, n, vals.clone());

            let (_, h) = m.real_schur_decomposition().unwrap();
            let eigs_from_real: Vec<Complex<f64>> = extract_eigs_from_h(&h);

            let (_, t) = m.complex_schur_decomposition().unwrap();
            let eigs_from_complex: Vec<Complex<f64>> = diag_c(&t);

            assert!(
                eig_sets_match(&eigs_from_real, &eigs_from_complex),
                "eigenvalue sets disagree for {}x{} matrix {:?}",
                n,
                n,
                vals
            );
        }
    }
}