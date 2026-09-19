//! Implements algorithms to approximate matrix norms.

use std::f64::consts::PI;

use crate::ok;
use crate::lang::eval;
use crate::math::traits::*;
use crate::math::utils;
use crate::math::vectors::VectorNorm;
use crate::math::{Env, Expression, Matrix, Object, VarStack, Vector};
use crate::status::Status;


pub enum MatrixNorm {
    P(f64),
    Frobenius,
}
impl MatrixNorm {
    /// If `opt` is `None`, use the spectral norm. If `opt` is "inf" or "infty", use the supremum norm.
    /// If it is a string starting with f, use the Frobenius norm.
    /// Otherwise, evaluate `opt` and use the corresponding p-norm.
    pub fn from_expr(opt: &Option<Box<Expression>>, extra_vars: &VarStack, env: &mut Env) -> Result<Status<MatrixNorm>, String> {
        if let Some(inner) = opt {match &**inner {
            Expression::Identifier(ident) if ident == "inf" || ident == "infty" || ident == "∞"
                => ok!(MatrixNorm::P(f64::INFINITY)),
            Expression::Identifier(ident) if ident == "f" || ident == "F"
                => ok!(MatrixNorm::Frobenius),
            Expression::Identifier(ident) if ident == "spec"
                => ok!(MatrixNorm::P(2.0)),
            other => {
                if let Status{value: Object::Real(z), warnings} = eval(other, extra_vars, env)? {
                    Ok(Status{value: MatrixNorm::P(z), warnings})
                }
                else {
                    Err(format!("Couldn't evaluate {other} to float."))
                }
            }
        }} else {
            ok!(MatrixNorm::P(2.0))
        }
    }
}


impl<T: Scalar> Matrix<T> {
    /// Approximates the operator norm of `self` induced by the `p`-norm up to an error of at most `tolerance`.
    /// 
    /// This implementation is based on the [article](https://link.springer.com/article/10.1007/BF01396242)
    /// "Estimating the matrix p-norm" by Nicholas Highham. For explanations, see the article.
    /// 
    /// `p` must be at least one and may be `f64::INFINITY`. The matrix `self` does not have any constraints. `tolerance` is recommended to be `1e-10`.
    /// 
    /// This method does not treat the cases `p=1.0`, `p=2.0` and `p=f64::INFINITY` separately. For those, call `self.norm`.
    fn pnorm(&self, p: f64, tolerance: f64) -> Result<T::UnderlyingFloat, String> {
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
            let col_k: Vector<T> = self.col(k);
            let (c, s) = if k == 0 {
                (T::one(), T::zero())
            } else {
                let mut best_f = T::UnderlyingFloat::zero();
                let mut best_c = T::one();
                let mut best_s = T::zero();
                for i in 0..samples {
                    let th = PI * i as f64 / (samples - 1) as f64;
                    let mut cs = Vector::<T>{values: vec![T::from_f64(th.cos()), T::from_f64(th.sin())]};
                    let cs_norm = cs.norm(&VectorNorm::P(p));
                    for x in cs.values.iter_mut() {
                        x.div_assign(cs_norm);
                    }
                    let w_cs = (&col_k).mul(cs[0]).add((&y).mul(cs[1])).unwrap();
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
            y = col_k.mul(c).add(y.mul(s)).unwrap();
            if k > 0 {
                for xi in x.values.iter_mut().take(k) {
                    xi.mul_assign(s);
                }
            }
        }

        // Refinement: power iteration with dual vectors.
        let mut est = y.norm(&VectorNorm::P(p));
        for iter in 1usize.. {
            y = self.mul(&x).unwrap();
            let eo = est;
            est = y.norm(&VectorNorm::P(p));
            let dv_y = y.dual(p)?;
            let z = self.transpose().mul(&dv_y).unwrap();
            let z_q_norm = z.norm(&VectorNorm::P(q));
            if iter > 1 && (z_q_norm < (&z).mul(x).unwrap().abs() || est.sub(eo).abs() <= <T::UnderlyingFloat as Scalar>::UnderlyingFloat::from_f64(tolerance).mul(est)) {
                break;
            }
            x = z.dual(q)?;
        }
        Ok(est)
    }
    
    /// Returns the norm of the given matrix. The norm of a 0x0 matrix is set to be zero.
    pub fn norm(&self, norm_type: &MatrixNorm) -> Result<T::UnderlyingFloat, String> {
        match norm_type {
            // The sup-norm is simply the highest row sum, i.e. \max_i \sum_{j=1}^n |a_{i,j}|
            MatrixNorm::P(f64::INFINITY) => Ok(utils::max_of(
                (0..self.m).map(
                    |i| self.row_slice(i).iter().map(|x| x.abs()).sum()
                )
            ).unwrap_or(T::UnderlyingFloat::zero())),
            // The 1-norm is the highest column sum. We take a different approach than above to improve cache locality.
            MatrixNorm::P(1.0) => {
                let mut sums = vec![T::UnderlyingFloat::zero(); self.n];
                for i in 0..self.m {
                    sums.iter_mut().enumerate().for_each(|(j, x)| x.add_assign(self.get(i, j).abs()));
                }
                Ok(utils::max_of(sums.into_iter()).unwrap_or(T::UnderlyingFloat::zero()))
            }
            MatrixNorm::P(2.0) => {
                match self.gram_matrix().eigenvalues() {
                    Some(eigenvalues) => Ok(
                        utils::max_of(
                            eigenvalues.into_iter().map(
                                |x| x.modulus()
                            )
                        )
                        .unwrap_or(T::UnderlyingFloat::zero())
                        .sqrt()
                    ),
                    // `None` means the matrix isn't square.
                    None => self.pnorm(2.0, 1e-10)
                }
            }
            MatrixNorm::P(p) if *p >= 1.0 => {
                self.pnorm(*p, 1e-10)
            }
            MatrixNorm::P(other) => Err(format!("Parameter `p` must be at least 1 (got {:?}).", other)),
            MatrixNorm::Frobenius => Ok(self.values.iter().map(|x| x.abs().pow(T::UnderlyingFloat::from_usize(2))).sum::<T::UnderlyingFloat>().sqrt())
        }
    }
}