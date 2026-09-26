use crate::math::utils;
use super::*;

#[derive(Debug)]
#[allow(unused)]
pub enum SylvesterError {
    DimensionMismatch,
    /// When `A` and `-B` share an eigenvalue, the solution is not unique (and may not even exist).
    Singular
}

/// Solves the sylvester equation `AX + XB = C` for `X`.
/// 
/// Required dimensions:
/// - `A` must be `m`x`m` for some `m`
/// - `B` must be `n`x`n` for some `n`
/// - `C` must be `m`x`n` for these `m`, `n`
/// Then, the result `X` is `m`x`n`.
#[allow(unused)]
pub fn sylvester<T: Scalar>(
    a: &Matrix<T>,
    b: &Matrix<T>,
    c: &Matrix<T>
) -> Result<Matrix<T>, SylvesterError> {
    let m = a.m;
    let n = b.m;
    if a.n != m || b.n != n || c.m != m || c.n != n {
        return Err(SylvesterError::DimensionMismatch);
    }
    let size = m * n;

    // Build augmented system `K * X.values = -C.values`, `K = I ⊗ A + B^T ⊗ I`
    let mut k = vec![vec![T::zero(); size + 1]; size];
    for i in 0..m {
        for j in 0..n {
            let row = i * n + j;
            // (AX)[i,j] = sum_l A[i,l] X[l,j]
            for l in 0..m {
                k[row][l * n + j].add_assign(a.get(i, l));
            }
            // (XB)[i,j] = sum_l X[i,l] B[l,j]
            for l in 0..n {
                k[row][i * n + l].add_assign(b.get(l, j));
            }
            k[row][size] = c.get(i, j);
        }
    }

    // Gaussian elimination with partial pivoting.
    // Tolerance is relative to the largest entry of K.
    let scale = utils::max_abs_of(
        k
        .iter()
        .flat_map(|r| r[..size].iter())
    );
    let tol = scale.mul(<T as Scalar>::UnderlyingReal::from_usize(size)).mul(<T as Scalar>::UnderlyingReal::EPSILON);

    for col in 0..size {
        let mut piv = col;
        let mut best = k[col][col].abs();
        for r in (col + 1)..size {
            let v = k[r][col].abs();
            if v > best {
                best = v;
                piv = r;
            }
        }
        if best <= tol {
            return Err(SylvesterError::Singular);
        }
        k.swap(col, piv);

        let p = k[col][col];
        for r in (col + 1)..size {
            let f = k[r][col].div(p);
            if f.abs().is_zero() {
                continue;
            }
            for cc in col..=size {
                let t = f.mul(k[col][cc]);
                k[r][cc] = k[r][cc].sub(t);
            }
        }
    }

    // Back substitution.
    let mut x = vec![T::zero(); size];
    for r in (0..size).rev() {
        let mut s = k[r][size];
        for cc in (r + 1)..size {
            s = s.sub(k[r][cc].mul(x[cc]));
        }
        x[r] = s.div(k[r][r]);
    }

    Ok(Matrix::from(m, n, x))
}

/// Solves the Sylvester equation `AX + XB = C` where all present matrices must be `n`x`n` for the same `n`.
/// 
/// The algorithm is taken from part I of the article [Recursive blocked algorithms for solving triangular
/// systems](<https://dl.acm.org/doi/10.1145/592843.592845>) by I. Jonsson and B. Kågström.
/// 
/// This function does not verify that the matrices are upper triangular, only that the dimensions match
/// (to avoid panicking). For non-upper triangular matrices, it will return garbage.
#[allow(unused)]
pub fn sylvester_upper_triangular<T: Scalar>(
    a: &Matrix<T>,
    b: &Matrix<T>,
    c: &Matrix<T>
) -> Result<Matrix<T>, SylvesterError> {
    let n = a.m;
    if a.m != n || a.n != n || b.m != n || b.n != n || c.m != n || c.n != n {
        return Err(SylvesterError::DimensionMismatch);
    }
    let mut x = Matrix::zeros(n, n);
    sylvester_upper_triangular_rec(a.view(), b.view(), c.view(), x.view_mut()).ok_or(SylvesterError::Singular)?;
    Ok(x)
}
/// Does not check if dimensions match anymore, but requires `A` to be `m`x`m`,
/// `B` to be `n`x`n` and `C` and `out` to be `m`x`n`.
pub fn sylvester_upper_triangular_rec<T: Scalar>(
    a: MatrixView<T>,
    b: MatrixView<T>,
    c: MatrixView<T>,
    mut out: MatrixViewMut<T>
) -> Option<()>
{
    let m = a.rows();
    let n = b.rows();
    if m == 1 && n == 1 {
        // `ax + xb = c`, so `x = c/(a+b)`
        let denom = a.get(0, 0).add(b.get(0, 0));
        if utils::approx_eq(denom, T::zero()) {
            return None;
        }
        out.set(0, 0, c.get(0, 0).div(denom));
        return Some(());
    }

    if m >= n {
        // Split rows: X = [X1; X2], C = [C1; C2].
        //   A22 X2 + X2 B = C2
        //   A11 X1 + X1 B = C1 - A12 X2
        let m1 = (m + 1) / 2;
        let m2 = m - m1;
        sylvester_upper_triangular_rec(
            a.submatrix(m1, m1, m2, m2),
            b,
            c.submatrix(m1, 0, m2, n),
            out.submatrix_mut(m1, 0, m2, n)
        )?;
        let c1 = c.submatrix(0, 0, m1, n)
            .sub(
                a.submatrix(0, m1, m1, m2)
                    .mul(out.submatrix_immut(m1, 0, m2, n))
                    .unwrap() // Safe, dimensions match
                    .view()
            )
            .unwrap();
        sylvester_upper_triangular_rec(
            a.submatrix(0, 0, m1, m1),
            b,
            c1.view(),
            out.submatrix_mut(0, 0, m1, n)
        )
    } else {
        // Split columns: X = [X1 X2], C = [C1 C2].
        //   A X1 + X1 B11 = C1
        //   A X2 + X2 B22 = C2 - X1 B12
        let n1 = (n + 1) / 2;
        let n2 = n - n1;
        sylvester_upper_triangular_rec(
            a,
            b.submatrix(0, 0, n1, n1),
            c.submatrix(0, 0, m, n1),
            out.submatrix_mut(0, 0, m, n1)
        )?;
        let c2 = c.submatrix(0, n1, m, n2)
            .sub(
                out.submatrix_immut(0, 0, m, n1)
                    .mul(b.submatrix(0, n1, n1, n2))
                    .unwrap() // Safe, dimensions match
                    .view()
            )
            .unwrap();
        sylvester_upper_triangular_rec(
            a,
            b.submatrix(n1, n1, n2, n2),
            c2.view(),
            out.submatrix_mut(0, n1, m, n2)
        )
    }
}