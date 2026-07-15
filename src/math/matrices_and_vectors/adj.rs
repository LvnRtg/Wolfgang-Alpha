//! Implements functions to compute adjugate matrices in O(n³).

use crate::math::{Matrix, utils};

impl Matrix {
    /// Returns the adjugate matrix of `self`. Returns `None` if `self` is not square.
    /// 
    /// We decompose `A := self` into `PAQ = LDU` via the full-pivot LU decomposition where `L` and `U` only
    /// have `1`s on their respective diagonal. Then, `A = XDY` for `X = P^t L` and `Y = U Q^t`.
    /// 
    /// Then, `adj(A) = adj(Y) adj(D) adj(X) = det(P) det(Q) Q adj(U) adj(D) adj(L) P`, which can be computed in O(n³).
    pub fn adj(&self) -> Option<Matrix> {
        if self.m != self.n {return None;}
        if self.m == 0 {return Some(Matrix{m: 0, n: 0, values: vec![]});}
        if self.m == 1 {return Some(Matrix{m: 1, n: 1, values: vec![1.0]});} // Adjugate of 1x1 matrix is always [1]
        let (l, mut u, pt, qt) = self.lu_decomposition_full_pivot()?;
        let n = self.n;
        // Currently, U doesn't necessarily have diagonal entries (1, ..., 1). We still have to extract D.
        let d = Matrix::diag(&(0..n).map(|i| u.get(i, i)).collect::<Vec<f64>>());
        for i in 0..n {
            if !utils::approx_eq(d.get(i, i), 0.0) {
                for j in i..n {
                    u.values[i*n+j] /= d.get(i, i);
                }
            }
        }
        // Now, apply the formula from the documentation.
        let det_p = if utils::permutation_parity(&pt) {1.0} else {-1.0};
        let det_q = if utils::permutation_parity(&qt) {1.0} else {-1.0};
        let adj_d = d.adj_for_diagonal_matrix()?;
        &(
            det_p
            * det_q
            * &(&u.adj_for_upper_triangular_matrix().permute_rows(&utils::transpose_permutation(&qt))?
            * &adj_d)?
        ) * &l.transpose().adj_for_upper_triangular_matrix()
            .permute_rows(&utils::transpose_permutation(&pt))?
            .transpose()
    }

    /// Returns the adjugate matrix of `self` assuming that `self` is a diagonal matrix `D`. Then, `adj(D)` is the diagonal matrix with
    /// entries `adj(D)_{i,i} = \prod_{j \neq i} D_{j,j}`. Returns `None` if the matrix isn't square.
    /// 
    /// Interestingly enough, this function runs in only O(n) of time (see implementation).
    /// 
    /// This forces us not to check whether the matrix is indeed diagonal (this would be O(n²)).
    /// If the matrix is square but not diagonal, returns a matrix that probably isn't the correct adjugate.
    pub fn adj_for_diagonal_matrix(&self) -> Option<Matrix> {
        if self.m != self.n {return None;}
        if self.m == 0 {return Some(Matrix{m: 0, n: 0, values: vec![]});}
        if self.m == 1 {return Some(Matrix{m: 1, n: 1, values: vec![1.0]});} // Adjugate of 1x1 matrix is always [1]
        // Compute the prefix products p_i := \prod_{k=0}^i D_{k,k} and the suffix products s_i := \prod_{k=i}^{n-1} D_{k,k}.
        let mut p = Vec::<f64>::with_capacity(self.n);
        p.push(self.get(0, 0)); // p_0 = D_{0,0}
        for i in 1..self.n {
            p.push(p[i-1] * self.get(i, i));
        }
        let mut s_reverse = Vec::<f64>::with_capacity(self.n);
        s_reverse.push(self.get(self.n-1, self.n-1)); // s_{n-1} = D_{n-1,n-1}
        for i in 1..self.n {
            s_reverse.push(s_reverse[i-1] * self.get(self.n - i - 1, self.n - i - 1));
        }
        // Now, adj(D)_{i,i} = \prod_{j \neq i} D_{j,j} = p_{i-1} * s_{i+1} = p_{i-1} * s_reverse_{n-i}
        let mut diag = Vec::<f64>::with_capacity(self.n);
        diag.push(s_reverse[self.n-2]);
        for i in 1..self.n-1 {
            diag.push(p[i-1] * s_reverse[self.n - i - 2]);
        }
        diag.push(p[self.n-2]);
        Some(Matrix::diag(&diag))
    }
    
    /// Returns the adjugate matrix of `self` assuming that `self` is a square upper triangular matrix `U`.
    /// 
    /// Then, `adj(U)` is also an upper triangular matrix with non-trivial entries
    /// `adj(U)_{i,j} = (-1)^{i+j} (\prod_{p<i} U_{p,p}) (\prod_{q>j} U_{q,q}) det(H^{(i,j)})`
    /// where `H` is the `(j-i)x(j-i)` Hessenberg matrix given by (indexing from zero)
    /// `H^{(i,j)}_{k,l} = u_{i+k,i+1+l}`.
    /// 
    /// This function runs in `O(n³)`.
    pub fn adj_for_upper_triangular_matrix(&self) -> Matrix {
        let n = self.n;
        // Compute the prefix products p_i := \prod_{k=0}^i U_{k,k} and the suffix products s_i := \prod_{k=i}^{n-1} U_{k,k}.
        let mut p = Vec::<f64>::with_capacity(n);
        p.push(self.get(0, 0)); // p_0 = D_{0,0}
        for i in 1..n {
            p.push(p[i-1] * self.get(i, i));
        }
        let mut s_reverse = Vec::<f64>::with_capacity(n);
        s_reverse.push(self.get(n-1, n-1)); // s_{n-1} = D_{n-1,n-1}
        for i in 1..n {
            s_reverse.push(s_reverse[i-1] * self.get(n - i - 1, n - i - 1));
        }

        // Notice `adj(U)` is an upper triangular matrix too
        let mut adj = Matrix::zeros(n, n);

        for i in 0..n {
            // Fix i.
            // To compute the determinant of H := H_{i,j}, we generally proceed as in `det_for_hessenberg_matrix`.
            // However, we can share the subdeterminants to reduce time complexity from O(n⁴) to O(n³).
            // Specifically, when indexing from 1, we'd have with d_r := det H_{i, i+r} (r=0,...,n-i) that
            // d_0 = 1, d_k = \sum_{r=1}^k (-1)^{k-r} U_{i+r-1,i+k} (\prod_{t=r}^{k-1} U_{i+t,i+t}) d_{r-1}.
            // Since indexing actually starts at zero, the code will have some indices shifted.
            // We first compute all D_i in order (total complexity: O(n²)).
            let mut d = vec![0.0; n-i+1];
            d[0] = 1.0;
            for k in 1..n-i {
                let mut chain = 1.0;
                for r in (1..=k).rev() {
                    d[k] += (if (k-r) % 2 == 0 {1.0} else {-1.0}) * self.get(i+r-1, i+k) * chain * d[r-1];
                    if r > 1 {
                        chain *= self.get(i+r-1, i+r-1);
                    }
                }
            }
            for j in i..n {
                adj.set(i, j,
                    (if (i+j) % 2 == 0 {1.0} else {-1.0})
                    * (if i > 0 {p[i-1]} else {1.0})
                    * (if j < n-1 {s_reverse[n-j-2]} else {1.0})
                    * d[j-i]
                )
            }
        }
        adj
    }
}