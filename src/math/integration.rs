use std::collections::HashMap;
use std::ops::{Add, AddAssign, Div, Mul, Neg};
use num_traits::float::Float;

use crate::lang::eval;
use crate::math::{BinaryOperation, Env, Expression, FoldedOperation, Object, UnaryOperation, VarStack};
use crate::math::objects::try_operation;

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
      U: AddAssign<U> + Mul<f64, Output=U> + Neg<Output=U> + Mul<T, Output=U> {
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


/// Integrates the given expresion numerically from a to b.
pub fn integrate(expr: &Expression, a: f64, b: f64, wrt: &String, extra_vars: &VarStack, env: &mut Env) -> Result<Object, String> {
    match expr {
        Expression::None => Ok(Object::Undefined),
        Expression::Identifier(ident) => {
            if ident == wrt {
                // Having to compute \int_a^b x dx doesn't tell us what the type of x is supposed to be, so we treat it as a real number.
                Ok(Object::Float((b.powi(2) - a.powi(2)) / 2.0))
            } else {
                Ok((b-a) * (extra_vars.lookup(ident).unwrap_or(env.constants.get(ident).ok_or(format!("No such variable `{}`.", ident))?)))
            }
        }
        Expression::Number(x) => Ok(Object::Float((b-a) * x)),
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
        Expression::FoldedOperation(FoldedOperation::Sum, varname, from, conditions, to, inner)
        if !from.contains_identifier(wrt) && !to.contains_identifier(wrt) && conditions.iter().all(|e| !e.contains_identifier(wrt)) => {
            // This code is stolen from `lang::evaluator::eval` as well
            let mut i = eval(from, extra_vars, env)?.expect_int()?;
            if i > eval(to, &VarStack::Frame { vars: &HashMap::from([(varname, &Object::Float(i))]), parent: extra_vars }, env)?.expect_float()? {
                // TODO check type once done in `eval`
                return Ok(FoldedOperation::Sum.if_empty());
            }
            let mut res = FoldedOperation::Sum.if_empty(); // TODO also change type here
            'outer: while i <= eval(to, &VarStack::Frame { vars: &HashMap::from([(varname, &Object::Float(i))]), parent: extra_vars }, env)?.expect_float()? {
                for cond in conditions {
                    match eval(cond, &VarStack::Frame { vars: &HashMap::from([(varname, &Object::Float(i))]), parent: extra_vars }, env)? {
                        Object::Float(1.0) => {}
                        Object::Float(0.0) => {
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
                    &VarStack::Frame { vars: &HashMap::from([(varname, &Object::Float(i))]), parent: extra_vars },
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
            &eval(e, &VarStack::Frame { vars: &HashMap::from([(wrt, &Object::Float(b))]), parent: extra_vars }, env)?,
            &eval(e, &VarStack::Frame { vars: &HashMap::from([(wrt, &Object::Float(a))]), parent: extra_vars }, env)?,
            &BinaryOperation::Sub
        ),
        other => simpson_rule_result_variant(
            |x| eval(other, &VarStack::Frame { vars: &HashMap::from([(wrt, &Object::Float(x))]), parent: extra_vars }, env),
            a, b,
            100
        )
    }
}