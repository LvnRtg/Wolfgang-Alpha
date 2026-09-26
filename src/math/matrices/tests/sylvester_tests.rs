#[cfg(test)]
mod sylvester_tests {
    use crate::math::Matrix;
    use crate::math::matrices::sylvester::*;
    use crate::math::traits::*;

    /// Computes `max_{i,j} |(AX + XB - C)_{i,j}|`.
    fn residual(a: &Matrix<f64>, b: &Matrix<f64>, c: &Matrix<f64>, x: &Matrix<f64>) -> f64 {
        let (m, n) = (a.m, b.m);
        let mut worst = 0.0_f64;
        for i in 0..m {
            for j in 0..n {
                let mut s = -c.get(i, j);
                for l in 0..m {
                    s += a.get(i, l) * x.get(l, j);
                }
                for l in 0..n {
                    s += x.get(i, l) * b.get(l, j);
                }
                worst = worst.max(s.abs());
            }
        }
        worst
    }


    /// Returns the matrix `C` such that `X` solves the Sylvester equation induced by `(A, B, C)`.
    fn make_c(a: &Matrix<f64>, b: &Matrix<f64>, x: &Matrix<f64>) -> Matrix<f64> {
        a.mul(x).unwrap().add(x.mul(b).unwrap()).unwrap()
    }

    /// Builds a pseudo-random upper triangular `n`x`n` matrix with diagonal shifted by `shift`.
    fn random_upper_triangular(state: &mut u64, n: usize, shift: f64) -> Matrix<f64> {
        let mut mat = Matrix::zeros(n, n);
        for i in 0..n {
            for j in i..n {
                *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let v = ((*state >> 33) as f64) / (1u64 << 31) as f64 - 0.5;
                mat.set(i, j, v);
            }
            let d = mat.get(i, i);
            mat.set(i, i, d + shift);
        }
        mat
    }

    #[test]
    fn scalar_case() {
        // 2x + 3x = 10  =>  x = 2
        let a = Matrix::from(1, 1, vec![2.0]);
        let b = Matrix::from(1, 1, vec![3.0]);
        let c = Matrix::from(1, 1, vec![10.0]);
        let x = sylvester(&a, &b, &c).unwrap();
        assert!(x.approx_eq(&Matrix::from(1, 1, vec![2.0])))
    }

    #[test]
    fn recovers_known_solution_non_square() {
        // n != m, so any mix-up between row/column indexing is caught.
        let a = Matrix::from(3, 3, vec![
            4.0, 1.0, 0.0,
            0.0, 5.0, 1.0,
            1.0, 0.0, 6.0
        ]);
        let b = Matrix::from(2, 2, vec![
            3.0, 1.0,
            0.0, 4.0
        ]);
        let x_true = Matrix::from(3, 2, vec![
            1.0,  2.0,
            3.0, -1.0,
            0.5,  4.0
        ]);
        let c = make_c(&a, &b, &x_true);
        let x = sylvester(&a, &b, &c).unwrap();
        assert!(x.approx_eq(&x_true));
        assert!(residual(&a, &b, &c, &x) < 1e-10);
    }

    #[test]
    fn recovers_known_solution_wide() {
        // The opposite shape: n < m.
        let a = Matrix::from(2, 2, vec![
            5.0, 2.0,
            1.0, 4.0
        ]);
        let b = Matrix::from(3, 3, vec![
            3.0, 0.0, 1.0,
            1.0, 6.0, 0.0,
            0.0, 2.0, 5.0
        ]);
        let x_true = Matrix::from(2, 3, vec![
            1.0, -2.0, 0.0,
            2.5,  1.0, 3.0
        ]);
        let c = make_c(&a, &b, &x_true);
        let x = sylvester(&a, &b, &c).unwrap();
        assert!(x.approx_eq(&x_true));
    }

    #[test]
    fn small_residual_on_pseudo_random_input() {
        // Deterministic LCG, so the test is reproducible without a rand dependency.
        let mut state: u64 = 0x2545F4914F6CDD1D;
        let mut next = || {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as f64) / (1u64 << 31) as f64 - 0.5 // in [-0.5, 0.5)
        };

        let (n, m) = (5, 4);
        let mut a = Matrix::zeros(n, n);
        let mut b = Matrix::zeros(m, m);
        let mut c = Matrix::zeros(n, m);
        for i in 0..n {
            for j in 0..n {
                a.set(i, j, next());
            }
            a.values[i * a.n + i] += 10.0; // spectrum of A near +10
        }
        for i in 0..m {
            for j in 0..m {
                b.set(i, j, next());
            }
            b.values[i * b.n + i] += 10.0; // spectrum of -B near -10, so it is disjoint from A's
        }
        for i in 0..n {
            for j in 0..m {
                c.set(i, j, next());
            }
        }

        let x = sylvester(&a, &b, &c).unwrap();
        assert!(residual(&a, &b, &c, &x) < 1e-10);
    }

    #[test]
    fn zero_rhs_gives_zero_solution() {
        let a = Matrix::from(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = Matrix::from(2, 2, vec![5.0, 1.0, 0.0, 6.0]);
        let c = Matrix::zeros(2, 2);
        let x = sylvester(&a, &b, &c).unwrap();
        assert!(x.approx_eq(&Matrix::zeros(2, 2)));
    }

    #[test]
    fn shared_eigenvalue_is_reported_singular() {
        // eig(A) = {1, 2}, eig(-B) = {2, -5}: 2 is shared, so no unique solution.
        let a = Matrix::from(2, 2, vec![1.0, 0.0, 0.0, 2.0]);
        let b = Matrix::from(2, 2, vec![-2.0, 0.0, 0.0, 5.0]);
        let c = Matrix::from(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
        assert!(matches!(sylvester(&a, &b, &c), Err(SylvesterError::Singular)));
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let a = Matrix::<f64>::zeros(2, 2);
        let b = Matrix::<f64>::zeros(3, 3);
        // C should be 2x3 but is 3x2.
        let c_bad = Matrix::<f64>::zeros(3, 2);
        assert!(matches!(
            sylvester(&a, &b, &c_bad),
            Err(SylvesterError::DimensionMismatch)
        ));
        // A is not square.
        let a_bad = Matrix::zeros(2, 3);
        let c = Matrix::zeros(2, 3);
        assert!(matches!(
            sylvester(&a_bad, &b, &c),
            Err(SylvesterError::DimensionMismatch)
        ));
    }


    #[test]
    fn triangular_scalar_case() {
        // 2x + 3x = 10  =>  x = 2
        let a = Matrix::from(1, 1, vec![2.0]);
        let b = Matrix::from(1, 1, vec![3.0]);
        let c = Matrix::from(1, 1, vec![10.0]);
        let x = sylvester_upper_triangular(&a, &b, &c).unwrap();
        assert!(x.approx_eq(&Matrix::from(1, 1, vec![2.0])));
    }

    #[test]
    fn triangular_recovers_known_solution_2x2() {
        // All four blocks are 1x1, so every sign in the block recurrence is exercised.
        let a = Matrix::from(2, 2, vec![
            4.0, 1.0,
            0.0, 5.0
        ]);
        let b = Matrix::from(2, 2, vec![
            3.0, 2.0,
            0.0, 6.0
        ]);
        let x_true = Matrix::from(2, 2, vec![
            1.0, -2.0,
            3.0,  0.5
        ]);
        let c = make_c(&a, &b, &x_true);
        let x = sylvester_upper_triangular(&a, &b, &c).unwrap();
        assert!(x.approx_eq(&x_true));
        assert!(residual(&a, &b, &c, &x) < 1e-10);
    }

    #[test]
    fn triangular_odd_size_three() {
        // n = 3 splits into 2 + 1, so sub-problems are rectangular.
        let a = Matrix::from(3, 3, vec![
            4.0, 1.0, -1.0,
            0.0, 5.0,  2.0,
            0.0, 0.0,  6.0
        ]);
        let b = Matrix::from(3, 3, vec![
            3.0, 1.0,  2.0,
            0.0, 4.0, -1.0,
            0.0, 0.0,  7.0
        ]);
        let x_true = Matrix::from(3, 3, vec![
            1.0,  2.0, -1.0,
            3.0, -1.0,  0.5,
            0.5,  4.0,  2.0
        ]);
        let c = make_c(&a, &b, &x_true);
        let x = sylvester_upper_triangular(&a, &b, &c).unwrap();
        assert!(x.approx_eq(&x_true));
        assert!(residual(&a, &b, &c, &x) < 1e-10);
    }

    #[test]
    fn triangular_small_residual_on_pseudo_random_input() {
        // Deterministic LCG, so the test is reproducible without a rand dependency.
        // Sizes cover even, odd, powers of two and non-powers of two.
        let mut state: u64 = 0x2545F4914F6CDD1D;
        for &n in &[1usize, 2, 3, 4, 5, 6, 7, 8, 9, 13, 16, 17] {
            let a = random_upper_triangular(&mut state, n, 10.0); // spectrum of A near +10
            let b = random_upper_triangular(&mut state, n, 10.0); // spectrum of -B near -10
            let mut c = Matrix::zeros(n, n);
            for i in 0..n {
                for j in 0..n {
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    c.set(i, j, ((state >> 33) as f64) / (1u64 << 31) as f64 - 0.5);
                }
            }
            let x = sylvester_upper_triangular(&a, &b, &c).unwrap();
            assert!(residual(&a, &b, &c, &x) < 1e-10, "residual too large for n = {n}");
        }
    }

    #[test]
    fn triangular_agrees_with_general_solver() {
        let mut state: u64 = 0x9E3779B97F4A7C15;
        for &n in &[2usize, 3, 5, 6] {
            let a = random_upper_triangular(&mut state, n, 8.0);
            let b = random_upper_triangular(&mut state, n, 8.0);
            let x_true = random_upper_triangular(&mut state, n, 0.0); // arbitrary values
            let c = make_c(&a, &b, &x_true);
            let fast = sylvester_upper_triangular(&a, &b, &c).unwrap();
            let slow = sylvester(&a, &b, &c).unwrap();
            assert!(fast.approx_eq(&slow), "solvers disagree for n = {n}");
        }
    }

    #[test]
    fn triangular_zero_rhs_gives_zero_solution() {
        let a = Matrix::from(3, 3, vec![1.0, 2.0, 3.0, 0.0, 4.0, 5.0, 0.0, 0.0, 6.0]);
        let b = Matrix::from(3, 3, vec![5.0, 1.0, 2.0, 0.0, 6.0, 1.0, 0.0, 0.0, 7.0]);
        let c = Matrix::zeros(3, 3);
        let x = sylvester_upper_triangular(&a, &b, &c).unwrap();
        assert!(x.approx_eq(&Matrix::zeros(3, 3)));
    }

    #[test]
    fn triangular_shared_eigenvalue_is_reported_singular() {
        // eig(A) = {1, 2}, eig(-B) = {2, -5}: 2 is shared, so no unique solution.
        let a = Matrix::from(2, 2, vec![1.0, 1.0, 0.0, 2.0]);
        let b = Matrix::from(2, 2, vec![-2.0, 1.0, 0.0, 5.0]);
        let c = Matrix::from(2, 2, vec![1.0, 1.0, 1.0, 1.0]);
        assert!(matches!(
            sylvester_upper_triangular(&a, &b, &c),
            Err(SylvesterError::Singular)
        ));
    }

    #[test]
    fn triangular_singular_deep_in_recursion_is_reported() {
        // The shared eigenvalue (3 on A's last diagonal entry, -3 on B's first) only
        // shows up in a deep sub-problem; it must not be swallowed.
        let a = Matrix::from(3, 3, vec![
            10.0, 1.0, 1.0,
            0.0, 11.0, 1.0,
            0.0, 0.0,  3.0
        ]);
        let b = Matrix::from(3, 3, vec![
            -3.0, 1.0, 1.0,
            0.0, 12.0, 1.0,
            0.0, 0.0, 13.0
        ]);
        let c = Matrix::from(3, 3, vec![1.0; 9]);
        assert!(matches!(
            sylvester_upper_triangular(&a, &b, &c),
            Err(SylvesterError::Singular)
        ));
    }

    #[test]
    fn triangular_dimension_mismatch_is_rejected() {
        let a = Matrix::<f64>::zeros(2, 2);
        let b = Matrix::<f64>::zeros(3, 3);
        let c = Matrix::<f64>::zeros(2, 2);
        // B is 3x3 but A is 2x2: must be the same n.
        assert!(matches!(
            sylvester_upper_triangular(&a, &b, &c),
            Err(SylvesterError::DimensionMismatch)
        ));
        // A is not square.
        let a_bad = Matrix::<f64>::zeros(2, 3);
        let b2 = Matrix::<f64>::zeros(2, 2);
        assert!(matches!(
            sylvester_upper_triangular(&a_bad, &b2, &c),
            Err(SylvesterError::DimensionMismatch)
        ));
    }
}