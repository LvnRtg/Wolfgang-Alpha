use std::collections::HashMap;
use std::ops::{Add, AddAssign, Div, Mul, Neg};
use num_traits::float::Float;

use crate::lang::eval;
use crate::math::{Env, Expression, Object, VarStack};
use crate::math::objects::try_operation;
use crate::math::operations::{BinaryOperation, FoldedOperation, UnaryOperation};
use crate::{expr_add, expr_div, expr_mul, expr_neg, expr_pow, expr_sub};

/// Approximates the integral `\int_a^b f(x) dx` by splitting `[a, b]` into
/// `n` intervals of equal size and applying the Simpson rule to each one,
/// that is, it returns ```\frac{h}{3} [f(a) + 4f(a+h) + 2f(a+2h) + 4f(a+3h) + ... + f(b)]``` with `h := \frac{b-a}{2n}`.
/// 
/// Time complexity: O(n) * Complexity of f.
/// 
/// Error bound: if f \in C^4, then this function returns the true value of the integral with an error of O(n^{-4}).
/// This function is still generally good even for non-smooth functions: for e.g. an indicator function, for large
/// enough n, the error would be at most O(jump_height * (b-a) / n).
/// 
/// If `b < a`, return `-simpson_rule(f, b, a, n)`.
pub fn simpson_rule<F, T, U>(f: F, a: T, b: T, n: usize) -> U
where F: Fn(T) -> U,
      T: Float + AddAssign<T> + Div<f64, Output=T>,
      U: AddAssign<U> + Mul<f64, Output=U> + Neg<Output=U> + Mul<T, Output=U> + Default {
    if n == 0 {
        // There is no return value that makes sense here, but we don't want to panic and this should never happen anyway (so `Option` would be unnecessary overhead).
        return U::default();
    }
    if b < a {return -simpson_rule(f, b, a, n);}
    let h: T = (b - a) / T::from(2 * n).unwrap(); // Safe: Float (f32/f64) can always represent any usize, possibly with precision loss
    let mut x = a;
    let mut res = f(a);
    for _ in 0..(n-1) {
        x += h;
        res += f(x) * 4.;
        x += h;
        res += f(x) * 2.0;
    }
    x += h;
    res += f(x) * 4.0;
    x += h;
    res += f(x);
    res * (h / 3.0)
}
/// Variant of `simpson_rule` where `f` outputs `Result` which is passed down on error.
pub fn simpson_rule_result_variant<F, T, U>(mut f: F, a: T, b: T, n: usize) -> Result<U, String>
where F: FnMut(T) -> Result<U, String>,
      T: Float + AddAssign<T> + Div<f64, Output=T>,
      U: Add<U, Output=Result<U, String>> + Mul<f64, Output=U> + Neg<Output=Result<U, String>> + Mul<T, Output=U> {
    if b < a {
        return simpson_rule_result_variant(f, b, a, n)?.neg(); // I have no idea why, but `-` doesn't seem to work here.
    }
    let h: T = (b - a) / T::from(2 * n).unwrap(); // Safe: Float (f32/f64) can always represent any usize, possibly with precision loss
    let mut x = a;
    let mut res = f(a)?;
    for _ in 0..(n-1) {
        x += h;
        res = (res + (f(x)? * 4.0))?;
        x += h;
        res = (res + (f(x)? * 2.0))?;
    }
    x += h;
    res = (res + (f(x)? * 4.0))?;
    x += h;
    res = (res + f(x)?)?;
    Ok(res * (h / 3.0))
}


/// Numerically integrates the given expresion numerically from a to b.
/// 
/// Procedure: first, split of the cases where `a` or `b` is infinite. Assuming they are both finite, check if the expression is of a special form
/// (e.g. a sum, a constant, etc.; for a detailed list, cf. implementation),
/// compute the integral accordingly (e.g. via direct calculation for constants or by integrating both summands and then adding the results).
/// If no special form is found, integrate numericaclly using the Simpson rule on a grid of 100 equally distributed points.
/// 
/// If `a = -∞` or `b = ∞`, we use a substitution trick to reduce to a finite interval.
pub fn integrate(expr: &Expression, a: f64, b: f64, wrt: &String, extra_vars: &VarStack, env: &mut Env) -> Result<Object, String> {
    if a == f64::INFINITY || b == -f64::INFINITY {
        return Ok(Object::Real(0.0));
    }
    if a.is_finite() && b == f64::INFINITY {
        // Substitute φ(t) = t/(1-ct) for c:=1 if a!=-1 and c:=2 otherwise, leading to
        // int_a^∞ f(x) dx = int_{a/(1+ac)}^{1/c} f(t/(1-ct)) / (1-ct)² dt
        // Below function `ct` creates `c * t` as expression.
        let ct = || if a != -1.0 {
            Expression::Identifier(wrt.clone())
        } else {
            expr_mul!(Expression::Number(2.0), Expression::Identifier(wrt.clone()))
        };
        let new_arg = expr_div!(
            Expression::Identifier(wrt.clone()),
            expr_sub!(
                Expression::Number(1.0),
                ct()
            )
        );
        return integrate(
            &expr_div!(
                expr.replace_identifiers(wrt, &new_arg),
                expr_pow!(
                    expr_sub!(
                        Expression::Number(1.0),
                        ct()
                    ),
                    Expression::Number(2.0)
                )
            ),
            a / (1.0 + (if a != -1.0 {a} else {2.0*a})), // a / (1 + ac)
            if a != -1.0 {1.0} else {0.5}, // 1/c
            wrt,
            extra_vars,
            env
        );
    } else if a == -f64::INFINITY && b.is_finite() {
        // Substitute φ(t) = t/(ct-1) for c:=1 if b!=1 and c:=2 otherwise, leading to
        // int_{-∞}^b f(x) dx = int_{b/(cb-1)}^{1/c} -f(t/(ct-1)) / (1-ct)² dt
        let ct = || if b != 1.0 {
            Expression::Identifier(wrt.clone())
        } else {
            expr_mul!(Expression::Number(2.0), Expression::Identifier(wrt.clone()))
        };
        let new_arg = expr_div!(
            Expression::Identifier(wrt.clone()),
            expr_sub!(
                ct(),
                Expression::Number(1.0)
            )
        );
        return integrate(
            &expr_neg!(expr_div!(
                expr.replace_identifiers(wrt, &new_arg),
                expr_pow!(
                    expr_sub!(
                        Expression::Number(1.0),
                        ct()
                    ),
                    Expression::Number(2.0)
                )
            )),
            if b != 1.0 {1.0} else {0.5}, // 1/c
            b / ((if b != 1.0 {b} else {2.0*b}) - 1.0), // b / (cb - 1)
            wrt,
            extra_vars,
            env
        );
    } else if a == -f64::INFINITY && b == f64::INFINITY {
        // Substitute φ(t) = t/(1-t²), leading to
        // int_{-∞}^∞ f(x) dx = int_{-1}^1 f(t/(1-t²)) * (1+t²)/((1-t²)²) dt
        let t_square = || expr_pow!(
            Expression::Identifier(wrt.clone()),
            Expression::Number(2.0)
        );
        let new_arg = expr_div!(
            Expression::Identifier(wrt.clone()),
            expr_sub!(
                Expression::Number(1.0),
                t_square()
            )
        );
        return integrate(
            &expr_mul!(
                expr.replace_identifiers(wrt, &new_arg),
                expr_div!(
                    expr_add!(
                        Expression::Number(1.0),
                        t_square()
                    ),
                    expr_pow!(
                        expr_sub!(
                            Expression::Number(1.0),
                            t_square()
                        ),
                        Expression::Number(2.0)
                    )
                )
            ),
            -1.0,
            1.0,
            wrt,
            extra_vars,
            env
        );
    }

    match expr {
        Expression::None => Ok(Object::Undefined),
        Expression::Identifier(ident) => {
            if ident == wrt {
                // Having to compute \int_a^b x dx doesn't tell us what the type of x is supposed to be, so we treat it as a real number.
                Ok(Object::Real((b.powi(2) - a.powi(2)) / 2.0))
            } else {
                Ok((b-a) * (extra_vars.lookup(ident).or_else(|| env.constants.get(ident)).ok_or(format!("No such variable `{}`.", ident))?))
            }
        }
        Expression::Number(x) => Ok(Object::Real((b-a) * x)),
        Expression::Vector(v) => Ok(Object::Vector(crate::math::Vector{
            values: v.iter().map(|e|
                integrate(e, a, b, wrt, extra_vars, env).and_then(|o| o.expect_float())
            ).collect::<Result<Vec<_>, _>>()?
        })),
        Expression::Matrix(m, n, v) => Ok(Object::Matrix(crate::math::Matrix::from(
            *m,
            *n,
            v.iter().map(|e|
                integrate(e, a, b, wrt, extra_vars, env).and_then(|o| o.expect_float())
            ).collect::<Result<Vec<_>, _>>()?
        ))),
        Expression::UnaryOperation(UnaryOperation::Neg, e) => Ok((-&integrate(e, a, b, wrt, extra_vars, env)?)?),
        Expression::BinaryOperation(lhs, op @ (BinaryOperation::Add | BinaryOperation::Sub), rhs)
            => try_operation(&integrate(lhs, a, b, wrt, extra_vars, env)?, &integrate(rhs, a, b, wrt, extra_vars, env)?, op),
        
        // Only consider sums if all bounds do not include the integration variable (i.e. `w.r.t.`).
        Expression::FoldedOperation(FoldedOperation::Sum, index_var, from, conditions, to, inner)
        if !from.contains_identifier(wrt) && !to.contains_identifier(wrt) && conditions.iter().all(|e| !e.contains_identifier(wrt)) => {
            // This code is _borrowed_ from `lang::evaluator::eval` as well.
            let mut i = eval(from, extra_vars, env)?.expect_int()?;
            let initial_to_eval = eval(to, &VarStack::Frame { vars: &HashMap::from([(index_var, &Object::Real(i))]), parent: extra_vars }, env)?.expect_float()?;
            let mut res = FoldedOperation::Sum.if_empty(&inner.get_type(
                &VarStack::Frame { vars: &HashMap::from([(index_var, &Object::Real(initial_to_eval))]), parent: extra_vars },
                env
            )?);
            if i > initial_to_eval {
                return Ok(res);
            }
            'outer: while i <= eval(to, &VarStack::Frame { vars: &HashMap::from([(index_var, &Object::Real(i))]), parent: extra_vars }, env)?.expect_float()? {
                for cond in conditions {
                    match eval(cond, &VarStack::Frame { vars: &HashMap::from([(index_var, &Object::Real(i))]), parent: extra_vars }, env)? {
                        Object::Real(1.0) => {}
                        Object::Real(0.0) => {
                            i += 1.0;
                            continue 'outer;
                        }
                        other => return Err(format!("Expected 1 or 0 when evaluating condition, got {:?}.", other))
                    }
                }
                let next_term = integrate(
                    inner,
                    a, b,
                    wrt,
                    &VarStack::Frame { vars: &HashMap::from([(index_var, &Object::Real(i))]), parent: extra_vars },
                    env
                )?;
                res = try_operation(&res, &next_term, &BinaryOperation::Add)?;
                i += 1.0;
            }
            Ok(res)
        }
        Expression::Tuple(v) => Ok(Object::Tuple(
            v.iter().map(|e|
                integrate(e, a, b, wrt, extra_vars, env)
            ).collect::<Result<Vec<_>, _>>()?
        )),
        // \int_a^b d/dx f(x) dx = f(b) - f(a)
        Expression::PartialDerivative(diff_wrt, e) if diff_wrt == wrt => try_operation(
            &eval(e, &VarStack::Frame { vars: &HashMap::from([(wrt, &Object::Real(b))]), parent: extra_vars }, env)?,
            &eval(e, &VarStack::Frame { vars: &HashMap::from([(wrt, &Object::Real(a))]), parent: extra_vars }, env)?,
            &BinaryOperation::Sub
        ),
        other => simpson_rule_result_variant(
            |x| eval(other, &VarStack::Frame { vars: &HashMap::from([(wrt, &Object::Real(x))]), parent: extra_vars }, env),
            a, b,
            100
        )
    }
}