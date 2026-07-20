//! Implements matrix- and vector norms as well as a function to compute the dual vector (which is closely tied to vector norms).

use std::f64::consts::PI;

use crate::lang::eval;
use crate::math::utils;
use crate::math::{Env, Expression, Matrix, Object, VarStack, Vector};

pub enum VectorNorm {
    P(f64)
}
impl VectorNorm {
    /// If `opt` is `None`, use the euclidian 2-norm. If `opt` is "inf" or "infty", use the supremum norm.
    /// Otherwise, evaluate `opt` and use the corresponding p-norm.
    pub fn from_expr(opt: &Option<Box<Expression>>, extra_vars: &VarStack, env: &mut Env) -> Result<VectorNorm, String> {
        if let Some(inner) = opt {match &**inner {
            Expression::Identifier(ident) if ident == "inf" || ident == "infty"
                => Ok(VectorNorm::P(f64::INFINITY)),
            other => {
                if let Object::Real(z) = eval(other, extra_vars, env)? {
                    Ok(VectorNorm::P(z))
                }
                else {
                    Err(format!("Couldn't evaluate {other} to float."))
                }
            }
        }} else {Ok(VectorNorm::P(2.0))}
    }
}

pub enum MatrixNorm {
    P(f64),
    Frobenius,
}
impl MatrixNorm {
    /// If `opt` is `None`, use the spectral norm. If `opt` is "inf" or "infty", use the supremum norm.
    /// If it is a string starting with f, use the Frobenius norm.
    /// Otherwise, evaluate `opt` and use the corresponding p-norm.
    pub fn from_expr(opt: &Option<Box<Expression>>, extra_vars: &VarStack, env: &mut Env) -> Result<MatrixNorm, String> {
        if let Some(inner) = opt {match &**inner {
            Expression::Identifier(ident) if ident == "inf" || ident == "infty"
                => Ok(MatrixNorm::P(f64::INFINITY)),
            Expression::Identifier(ident) if ident.starts_with('f')
                => Ok(MatrixNorm::Frobenius),
            other => {
                if let Object::Real(z) = eval(other, extra_vars, env)? {
                    Ok(MatrixNorm::P(z))
                }
                else {
                    Err(format!("Couldn't evaluate {other} to float."))
                }
            }
        }} else {Ok(MatrixNorm::P(2.0))}
    }
}


impl Vector {
    /// Returns the norm of this vector (w.r.t. the given norm).
    pub fn norm(&self, norm_type: &VectorNorm) -> f64 {
        match norm_type {
            VectorNorm::P(f64::INFINITY) => utils::max_abs(self.values.iter()),
            VectorNorm::P(p) => self.values.iter().map(|x| x.abs().powf(*p)).sum::<f64>().powf(1.0 / *p),
        }
    }

    /// Returns the dual of `self` w.r.t. the p-norm. Since (l^p)^* and l^q are isometrically isomorphic for q s.t. 1/p+1/q=1
    /// (standard result from functional analysis), the dual of `self` can be identified by a vector `v*` s.t.
    /// `<v*, self> = ||self||_p`.
    /// 
    /// In this function, we return that `v` with the additional constraint `||v*||_q = 1`.
    /// 
    /// Returns `Err` if p is strictly less than one.
    pub fn dual(&self, p: f64) -> Result<Vector, String> {
        if p < 1.0 {
            return Err(format!("p must be at least 1, got p={p} instead."));
        }
    
        let n = self.len();
        let supnorm = self.norm(&VectorNorm::P(f64::INFINITY));
        if supnorm == 0.0 {
            return Ok(self.clone());
        }
    
        if p == 1.0 {
            // Then, `q = \infty`, so the dual is simply `self.values.map(sign)`.
            Ok(Vector { values: self.values.iter().map(|x| x.signum()).collect() })
        } else if p == f64::INFINITY {
            // Then, `q = 1`, so the dual is simply the unit vector pointing in direction `argmax_i |self[i]|`.
            let mut i: usize = 0; let mut highest_abs = 0.0;
            for (j, x) in self.values.iter().enumerate() {
                let abs = x.abs();
                if abs > highest_abs {
                    highest_abs = abs;
                    i = j;
                }
            }
            let mut dual = Vector::zeros(n);
            dual[i] = self[i].signum();
            Ok(dual)
        } else {
            let q = 1.0 / (1.0 - 1.0 / p);
            let mut dual = Vector {values: self.values.iter().map(|x| x.signum() * (x / supnorm).abs().powf(p - 1.0)).collect()};
            dual /= dual.norm(&VectorNorm::P(q));
            Ok(dual)
        }
    }
}


impl Matrix {
    /// Approximates the operator norm of `self` induced by the `p`-norm up to an error of at most `tolerance`.
    /// 
    /// This implementation is based on the [article](https://link.springer.com/article/10.1007/BF01396242)
    /// "Estimating the matrix p-norm" by Nicholas Highham. For explanations, see the article.
    /// 
    /// `p` must be at least one and may be `f64::INFINITY`. The matrix `self` does not have any constraints. `tolerance` is recommended to be `1e-10`.
    /// 
    /// This method does not treat the cases `p=1.0`, `p=2.0` and `p=f64::INFINITY` separately. For those, call `self.norm`.
    fn pnorm(&self, p: f64, tolerance: f64) -> Result<f64, String> {
        // All `unwrap`s below are safe because the dimensions of the operands fit.
        let q = if p == 1.0 {
            f64::INFINITY
        } else {
            1.0 / (1.0 - 1.0 / p)
        };
        let samples = 9; // Could theoretically be increased or reduced (until 2), but the default is 9.
        let mut y = Vector::zeros(self.m);
        let mut x = Vector::zeros(self.n);

        // Initialisation: block power method with angle sampling.
        for k in 0..self.n {
            let col_k = self.col(k);
            let (c, s) = if k == 0 {
                (1.0, 0.0)
            } else {
                let mut best_f = 0.0_f64;
                let mut best_c = 1.0_f64;
                let mut best_s = 0.0_f64;
                for i in 0..samples {
                    let th = PI * i as f64 / (samples - 1) as f64;
                    let mut cs = Vector{values: vec![th.cos(), th.sin()]};
                    cs /= cs.norm(&VectorNorm::P(p));
                    let w_cs = (&(cs[0] * &col_k) + &(cs[1] * &y)).unwrap();
                    let f = w_cs.norm(&VectorNorm::P(p));
                    if f > best_f {
                        best_f = f;
                        best_c = cs[0];
                        best_s = cs[1];
                    }
                }
                (best_c, best_s)
            };
            x[k] = c;
            y = (&(c * &col_k) + &(s * &y)).unwrap();
            if k > 0 {
                for xi in x.values.iter_mut().take(k) {
                    *xi *= s;
                }
            }
        }

        // Refinement: power iteration with dual vectors.
        let mut est = y.norm(&VectorNorm::P(p));
        for iter in 1usize.. {
            y = (self * &x).unwrap();
            let eo = est;
            est = y.norm(&VectorNorm::P(p));
            let dv_y = y.dual(p)?;
            // Slightly hacky; instead of `self^t * dv_y`, we write `dv_y * self`, which I implemented as `(dv_y^t * self)^t` for convenience
            // (the other operation would be undefined anyway), which in turn is mathematically exactly `self^t * dv_y`.
            let z = (&dv_y * self).unwrap();
            let z_q_norm = z.norm(&VectorNorm::P(q));
            // Below: `est` shouldn't be NaN, but this avoids looping forever if somehow this happens
            if iter > 1 && (z_q_norm < (&z * &x).unwrap() || (est - eo).abs() <= tolerance * est) || est.is_nan() {
                break;
            }
            x = z.dual(q)?;
        }
        Ok(est)
    }

    pub fn norm(&self, norm_type: &MatrixNorm) -> Result<f64, String> {
        match norm_type {
            // The sup-norm is simply the highest row sum, i.e. \max_i \sum_{j=1}^n |a_{i,j}|
            MatrixNorm::P(f64::INFINITY) => Ok(utils::max(
                (0..self.m).map(
                    |i| self.row_slice(i).iter().map(|x| x.abs()).sum()
                )
            )),
            // The 1-norm is the highest column sum. We take a different approach than above to improve cache locality.
            MatrixNorm::P(1.0) => {
                let mut sums = vec![0.0; self.n];
                for i in 0..self.m {
                    sums.iter_mut().enumerate().for_each(|(j, x)| *x += self.get(i, j).abs());
                }
                Ok(utils::max(sums.into_iter()))
            }
            MatrixNorm::P(2.0) => {
                match self.gram_matrix().eigenvalues() {
                    Some(eigenvalues) => Ok(utils::max(eigenvalues.into_iter().map(
                        |obj| match obj {
                            Object::Real(x) => x.abs(),
                            Object::Complex(x) => x.modulus(),
                            _ => 0.0 // Will never happen anyway, `qr_decomposition[0]` only consists of floats and complex values
                        }
                    )).sqrt()),
                    None => Err(format!("Matrix must be quadratic (got {}x{}).", self.m, self.n))
                }
            }
            MatrixNorm::P(p) if *p >= 1.0 => {
                self.pnorm(*p, 1e-10)
            }
            MatrixNorm::P(other) => Err(format!("Parameter `p` must be at least 1 (got {other}).")),
            MatrixNorm::Frobenius => Ok(self.values.iter().map(|x| x.powi(2)).sum::<f64>().sqrt())
        }
    }
}