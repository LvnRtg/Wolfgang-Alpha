//! Implements vector norms as well as a function to compute the dual vector (which is closely tied to vector norms).

use crate::lang::eval;
use crate::math::traits::*;
use crate::math::utils;
use crate::math::{Env, Expression, Object, VarStack, Vector};
use crate::status::Status;


pub enum VectorNorm {
    P(f64)
}
impl VectorNorm {
    /// If `opt` is `None`, use the euclidian 2-norm. If `opt` is "inf" or "infty", use the supremum norm.
    /// Otherwise, evaluate `opt` and use the corresponding p-norm.
    pub fn from_expr(opt: &Option<Box<Expression>>, extra_vars: &VarStack, env: &mut Env) -> Result<Status<VectorNorm>, String> {
        if let Some(inner) = opt {match &**inner {
            Expression::Identifier(ident) if ident == "inf" || ident == "infty" || ident == "∞"
                => Ok(Status::ok(VectorNorm::P(f64::INFINITY))),
            other => {
                if let Status{value: Object::Real(z), warnings} = eval(other, extra_vars, env)? {
                    Ok(Status{value: VectorNorm::P(z), warnings})
                }
                else {
                    Err(format!("Couldn't evaluate {other} to float."))
                }
            }
        }} else {Ok(Status::ok(VectorNorm::P(2.0)))}
    }
}


impl<T: Scalar> Vector<T> {
    /// Returns the norm of this vector (w.r.t. the given norm).
    pub fn norm(&self, norm_type: &VectorNorm) -> T::UnderlyingFloat {
        match norm_type {
            VectorNorm::P(f64::INFINITY) => utils::max_abs_of(self.values.iter()),
            VectorNorm::P(p) => self.values.iter().map(
                |x| x.abs().pow(T::UnderlyingFloat::from_f64(*p))
            )
            .sum::<T::UnderlyingFloat>()
            .pow(T::UnderlyingFloat::from_f64(1.0 / *p)),
        }
    }

    /// Returns the dual of `self` w.r.t. the p-norm. Since (l^p)^* and l^q are isometrically isomorphic for q s.t. 1/p+1/q=1
    /// (standard result from functional analysis), the dual of `self` can be identified by a vector `v*` s.t.
    /// `<v*, self> = ||self||_p`.
    /// 
    /// In this function, we return that `v` with the additional constraint `||v*||_q = 1`.
    /// 
    /// Returns `Err` if p is strictly less than one.
    pub fn dual(&self, p: f64) -> Result<Vector<T>, String> {
        if p < 1.0 {
            return Err(format!("p must be at least 1, got p={p} instead."));
        }
    
        let n = self.len();
        let supnorm = self.norm(&VectorNorm::P(f64::INFINITY));
        if supnorm == T::UnderlyingFloat::zero() {
            return Ok(self.clone());
        }
    
        if p == 1.0 {
            // Then, `q = \infty`, so the dual in the real case is simply `self.values.map(sign)`.
            // In the complex case, it wouldn't just be `sign(v[i])` but `conj(v[i]) / |v[i]|`.
            // In the most general form, this would simply be `|v[i]| / v[i]`.
            Ok(Vector { values: self.values.iter().map(|x| x.abs_div_by_self()).collect() })
        } else if p == f64::INFINITY {
            // Then, `q = 1`, so the dual is simply the unit vector pointing in direction `argmax_i |self[i]|`.
            // Generally, this "direction" is just `x / |x|`
            let mut i: usize = 0; let mut highest_abs = T::UnderlyingFloat::zero();
            for (j, x) in self.values.iter().enumerate() {
                let abs = x.abs();
                if abs > highest_abs {
                    highest_abs = abs;
                    i = j;
                }
            }
            let mut dual = Vector::zeros(n);
            dual[i] = self[i].normalized();
            Ok(dual)
        } else {
            let q = 1.0 / (1.0 - (1.0 / p));
            let mut dual = Vector {
                values: self.values.iter().map(
                    |x| x.normalized().mul(x.div(supnorm).abs().pow(T::UnderlyingFloat::from_f64(p - 1.0)))
                ).collect()
            };
            let dual_norm = dual.norm(&VectorNorm::P(q));
            for x in dual.values.iter_mut() {
                x.div_assign(dual_norm)
            }
            Ok(dual)
        }
    }
}