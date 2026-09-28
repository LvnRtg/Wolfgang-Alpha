//! Implements functions closely related to the determinant of a matrix.

use super::*;

impl<T: Scalar> Matrix<T> {
    /// Returns the product of all diagonal entries of `self`.
    fn diag_product(&self) -> T {
        (0..self.m).fold(T::one(), |acc, i| acc.mul(self.get(i, i)))
    }

    /// Returns the determinant of `self` via a full-pivot LU-decomposition.
    /// 
    /// Runs in 2/3 * n^3 + O(n^2).
    pub fn det(&self) -> Option<T> {
        self.lu_decomposition_full_pivot().map(
            |(l, u, p, q)|
            // We now have `self = p^T * L * U * q^T` so
            // det(self) = det(p) det(L) det(U) det(q)
            if p.parity() {T::one()} else {T::one().neg()}
            .mul(if q.parity() {T::one()} else {T::one().neg()})
            .mul(l.diag_product())
            .mul(u.diag_product())
        )
    }
}

impl<T> Matrix<T>
where T:
    Copy
    + One
    + Zero
    + Neg<Output=T>
    + AddAssign<T>
    + MulAssign<T>
{
    /// Returns the determinant of `self` requiring that `self` is an `nxn` Hessenberg matrix (`None` if `self` isn't square).
    /// 
    /// This algorithm is based on the fact that (indexing from 1) with `d_k := det(H_{1:k, 1:k}` and `d_0 := 1`,
    /// we have the recursive formula `d_k = \sum_{r=0}^k (-1)^{k-r} H_{r,k} (\prod_{t=r}^{k-1} H_{t+1,t}) d_{r-1}`.
    /// 
    /// Runs in O(n^2).
    pub fn det_for_hessenberg_matrix(&self) -> Option<T> {
        if self.m != self.n { return None; }
        let mut d = vec![T::zero(); self.n+1];
        d[0] = T::one();
        for k in 1..=self.n {
            let mut chain = T::one();
            for r in (1..=k).rev() {
                let local_storage = d[r-1];
                d[k].add_assign(
                    if (k-r) % 2 == 0 {T::one()} else {T::one().neg()}
                    .mul(self.get(r-1, k-1))
                    .mul(chain)
                    .mul(local_storage)
                );
                if r > 1 {
                    chain.mul_assign(self.get(r-1, r-2));
                }
            }
        }
        Some(d[self.n])
    }
}