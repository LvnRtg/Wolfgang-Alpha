use std::iter::zip;
use std::collections::HashMap;

use crate::math::matrices_and_vectors::VectorNorm;
use crate::math::objects::{try_operation};
use crate::math::expressions::*;
use crate::math::utils::approx_eq;
use crate::math::{Env, Object, DirectFunction, FunctionRepr, Vector, Matrix};
use crate::math::operations::{BinaryOperation, FoldedOperation, UnaryOperation};
use crate::{defaults, expr_compare, expr_if_else, expr_neg, lang};
use lang::evaluator::VarStack;

/// Differentiates the given expression w.r.t. the variable `wrt` analytically, that is, by parsing the expression recursively and
/// applying known differentiation rules (e.g. product rule, chain rule).
/// 
/// If an function `f` with representation `FunctionRepr::Direct` is encountered for which the derivative is not provided (as it is for default identifiers),
/// we cannot differentiate it analytically. Then, we use the special syntax `___diff_num_f`; the function `eval` then processes it as the function
/// `(x, y) \mapsto Df(x)[y]` instead of searching within `functions`. Then, this function proceeds as if the derivative of `f` had been provided already
/// and composes the new expression according to the chain rule.
/// 
/// When the expression is valid but not differentiable, this does not return a `Err` but `Ok(Expression::None)`.
/// If however the expression is invalid (e.g. unknown function identifier), then `Err` is returned.
pub fn analytic_partial_derivative(
    expr: &Expression,
    wrt: &String,
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Expression, String> {
    match expr {
        Expression::None => Ok(Expression::None),
        Expression::Identifier(ident) => Ok(Expression::Number(if ident == wrt { 1.0 } else { 0.0 })),
        Expression::Number(_) => Ok(Expression::Number(0.0)),
        Expression::Tuple(entries) => {
            Ok(Expression::Tuple(
                entries.iter()
                .map(|x| analytic_partial_derivative(x, wrt, extra_vars, env))
                .collect::<Result<Vec<_>, _>>()?
            ))
        }
        Expression::Vector(entries) => {
            Ok(Expression::Vector(
                entries.iter()
                .map(|x| analytic_partial_derivative(x, wrt, extra_vars, env))
                .collect::<Result<Vec<_>, _>>()?
            ))
        }
        Expression::Matrix(m, n, entries) => {
            Ok(Expression::Matrix(*m, *n,
                entries.iter()
                .map(|x| analytic_partial_derivative(x, wrt, extra_vars, env))
                .collect::<Result<Vec<_>, _>>()?
            ))
        }
        Expression::UnaryOperation(UnaryOperation::Neg, rhs) => {
            Ok(Expression::UnaryOperation(UnaryOperation::Neg, Box::new(
                analytic_partial_derivative(rhs, wrt, extra_vars, env)?
            )))
        }
        Expression::UnaryOperation(UnaryOperation::Not, _) => Err("Cannot differentiate the operation `Not`.".to_string()),
        Expression::UnaryOperation(UnaryOperation::Factorial, _) => {
            unimplemented!() // TODO when integrals are available via \Gamma'(x) = \int_0^\infty e^{-t} t^{x-1} ln(t) dt.
        }
        Expression::UnaryOperation(UnaryOperation::Abs, rhs) => {
            let diff_r = analytic_partial_derivative(rhs, wrt, extra_vars, env)?;
            Ok(expr_if_else!(
                expr_compare!(*rhs.clone(), Gt, Expression::Number(0.0)),
                diff_r.clone(),
                expr_if_else!(
                    expr_compare!(*rhs.clone(), Lt, Expression::Number(0.0)),
                    expr_neg!(diff_r.clone()),
                    Expression::None
                )
            ))
        }
        Expression::UnaryOperation(UnaryOperation::Norm(opt), rhs) => {
            match &**rhs {
                Expression::Vector(g_exprs) => {
                    // As discussed in the case `Expression::Function`, we need to return `Df(g(x))[Dg(x)[1]]`
                    // where `f(v) = ||v||_{opt}` and `g(x) = [g_exprs[0](x), ...]`.
                    match VectorNorm::from_expr(opt, extra_vars, env)? {
                        VectorNorm::P(f64::INFINITY) => {
                            // Derivative: undefined if there exist i != j s.t. |g_exprs[i](x)| = |g_exprs[j](x)|.
                            // Otherwise, equals sign(g_exprs[m](x)) * diff_g[m](x) with m := argmax_k |x_k|.
                            unimplemented!() // TODO when any() or something similar is available
                        }
                        VectorNorm::P(p) => {
                            // In this case, \partial_j ||y||_p = (|y_j| / ||y||_p)^{p-1} sign(y_j).
                            // Hence, D(f(g(x)))[Dg(x)[1]] = (\partial_j ||y||_p |_{g(x)})_j * (g'_j(x))_j
                            //                             = ((|g_j(x)| / ||g(x)||_p)^{p-1} sign(g_j(x))_j * (g'_j(x))_j
                            //                             = \sum_{j=1}^n (|g_j(x)| / ||g(x)||_p)^{p-1} * sign(y_j) * g'_j(x)
                            unimplemented!() // TODO when sums are available
                        }
                    }
                }
                Expression::Matrix(..) => {unimplemented!()} // TODO when the above is implemented
                rhs => {
                    // In this case, the norm should simply be an absolute value, regardless of `opt`.
                    let diff_r = analytic_partial_derivative(rhs, wrt, extra_vars, env)?;
                    Ok(expr_if_else!(
                        expr_compare!(rhs.clone(), Gt, Expression::Number(0.0)),
                        diff_r.clone(),
                        expr_if_else!(
                            expr_compare!(rhs.clone(), Lt, Expression::Number(0.0)),
                            expr_neg!(diff_r.clone()),
                            Expression::None
                        )
                    ))
                }
            }
        }
        Expression::BinaryOperation(lhs, op, rhs) => {
            let diff_l = analytic_partial_derivative(lhs, wrt, extra_vars, env)?;
            let diff_r = analytic_partial_derivative(rhs, wrt, extra_vars, env)?;
            match op {
                BinaryOperation::Add => Ok(simplify_add(diff_l, diff_r)),
                BinaryOperation::Sub => Ok(simplify_sub(diff_l, diff_r)),
                BinaryOperation::Quo | BinaryOperation::Rem | BinaryOperation::And | BinaryOperation::Or
                    => Err(format!("Cannot differentiate the operation `{op}`.")),
                BinaryOperation::Mul => Ok(simplify_add(
                    simplify_mul(diff_l, *rhs.clone()), // f'(x) * g(x)
                    simplify_mul(*lhs.clone(), diff_r)  // f(x) * g'(x)
                )),
                BinaryOperation::Div => Ok(simplify_div( // d/dx (f(x) / g(x)) = (f'(x)g(x) - f(x)g'(x)) / g(x)²
                    simplify_sub(
                        simplify_mul(diff_l, *rhs.clone()),
                        simplify_mul(*lhs.clone(), diff_r)
                    ),
                    simplify_pow(*rhs.clone(), Expression::Number(2.0))
                )),
                BinaryOperation::Pow => Ok(simplify_mul( // d/dx (f(x) ^ g(x)) = f(x)^(g(x)-1) * (f'(x)g(x) + f(x)g'(x)ln(f(x)))
                    simplify_pow(
                        *lhs.clone(),
                        simplify_sub(*rhs.clone(), Expression::Number(1.0))
                    ),
                    simplify_add(
                        simplify_mul(diff_l, *rhs.clone()),
                        simplify_mul(
                            simplify_mul(*lhs.clone(), diff_r),
                            Expression::Function("ln".to_string(), vec![*lhs.clone()])
                        )
                    )
                )),
                BinaryOperation::Comp(..) => Err(format!("Cannot differentiate comparison {:?}", expr)),
            }
        }
        Expression::FoldedOperation(FoldedOperation::Sum, varname, from, conditions, to, inner) => Ok(Expression::FoldedOperation(
            FoldedOperation::Sum,
            varname.clone(),
            from.clone(),
            conditions.clone(),
            to.clone(),
            Box::new(analytic_partial_derivative(inner, wrt, extra_vars, env)?)
        )),
        Expression::FoldedOperation(FoldedOperation::Product, varname, from, conditions, to, inner) => {
            // TODO
            unimplemented!()
        }
        Expression::Function(function_name, g_exprs) => {
            // Standard trick. To be able to create mutable references of `functions` within the `match` block, we don't call
            // `functions.get` but `functions.remove` and later reinsert the function. The only caveat is that we'll have
            // to clone `function_name` once, but this is fast since `function_name` typically only is `f`, `g`, etc.
            // For simplicity, I'll subsequently write `f` instead of `function_name`.
            // Define `g` such that `f(arg_expressions) = f(g(wrt))`. This explains the above name `g_exprs`
            let f = env.functions.remove(function_name).ok_or(format!("No such function \"{}\".", function_name))?;
            let res = analytic_partial_derivative_for_function(wrt, function_name, &f, g_exprs.clone(), extra_vars, env);
            env.functions.insert(function_name.clone(), f);
            res
        }
        // You can't differentiate expressions like `y := ...`, that makes no sense. If the user wants `y := d/dx ...`, he should have typed that. 
        Expression::Assignment(..) => Err("Assignment cannot be differentiated.".to_string()),
        Expression::PartialDerivative(wrt_other, inner) => {
            // Idea is simple: d/dx (d/dy f(x, y)) -> First evaluate the inner derivative, then differentiate the result.
            let res = analytic_partial_derivative(inner, wrt_other, extra_vars, env)?;
            analytic_partial_derivative( // Outer derivative
                &res, // Inner derivative
                wrt, extra_vars, env
            )
        }
        Expression::DirectionalDerivative(..)
            // The directional derivative is an object, so whatever it actually is, its derivative is zero.
            => Ok(Expression::Number(0.0)),
        Expression::IfElse(x, y, z)
            => Ok(Expression::IfElse(x.clone(), Box::new(analytic_partial_derivative(y, wrt, extra_vars, env)?), Box::new(analytic_partial_derivative(z, wrt, extra_vars, env)?))),
    }
}

fn analytic_partial_derivative_for_function(
    wrt: &String,
    function_name: &String,
    f: &FunctionRepr,
    mut g_exprs: Vec<Expression>,
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Expression, String> {
    match f {
        FunctionRepr::ByExpression(f_argnames, f_expr) => {
            // As discussed in the case `FunctionRepr::Direct`, we aim to return `Df(g(x))[Dg(x)[1]]` as an expression,
            // not as a value.
            if g_exprs.len() == 1 {
                let mut diff_f = analytic_partial_derivative(f_expr, &f_argnames[0], extra_vars, env)?;
                diff_f.replace_identifiers(&f_argnames[0], &g_exprs[0]); // Plug in g(x) into f'
                // If g only outputs one value, we can simply apply the 1d chain rule, (f \circ g)'(x) = g'(x) * f'(g(x)).
                Ok(simplify_mul(
                    analytic_partial_derivative(&g_exprs[0], wrt, extra_vars, env)?,
                    diff_f
                ))
            }
            else {
                // Otherwise, the idea is to resolve Dg(x)[1] and then return an `Expression::DirectionalDerivative`.
                let direction = g_exprs.iter()
                    .map(|g_i| analytic_partial_derivative(g_i, wrt, extra_vars, env))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Expression::DirectionalDerivative(
                    f_argnames.clone(),
                    Box::new(f_expr.clone()),
                    g_exprs,
                    direction
                ))
            }
        }
        FunctionRepr::Direct(_) => {
            // If `function_name` refers to a default function (e.g. `exp`), we can spare ourselves the below code.
            if defaults::FUNCTIONS_WITH_PROVIDED_DERIVATIVE.contains(&function_name.as_str()) {
                // Similar to the chain rule block in `analytic_directional_derivative`, with a little change: with the same f, g as there, we have
                //     D(f \circ g)(x)[1.0] = Df(g(x))[Dg(x)[1.0]]
                let differentiated_components_of_g = g_exprs.iter().map(
                    |g_i| analytic_partial_derivative(g_i, wrt, extra_vars, env)
                ).collect::<Result<Vec<_>, _>>()?;
                Ok(defaults::get_default_derivative(function_name.as_str(), &g_exprs, &differentiated_components_of_g)?)
            }
            else {
                // Importantly, note that the directional derivative is a separate function. Therefore, we can assume w.l.o.g. that `f \circ g` maps from `\R` to `\R`.
                // For each component of `g` (note that `g` maps from `\R` to `\R^n`), analytically differentiate that component w.r.t. `wrt` (which is the input of `g`).
                // We save these into a vector already to avoid calling `analytic_derivative` more often than necessary.
                // The returned expression should be (writing `x` for `wrt`)
                // ```d/dx f(g(x)) |_x
                //     = D(f \circ g)(x)[1]        (since `f \circ g` maps from `\R` to `\R`)
                //     = Df(g(x))[Dg(x)[1]]        (chain rule)```
                // In the program's syntax, this is equivalent to calling `___diff_num_f` with arguments `arg_expressions` concatenated with `(d/dx g_1, ... d/dx g_n)})`
                g_exprs.reserve(g_exprs.len());
                g_exprs.extend(g_exprs.iter()
                    .map(|g_i| analytic_partial_derivative(g_i, wrt, extra_vars, env))
                    .collect::<Result<Vec<_>, _>>()?);
                Ok(Expression::Function(
                    format!("___diff_num_{}", function_name),
                    g_exprs
                ))
            }
        }
    }
}

/// Analytically differentiates `expr` at point `point` in direction `direction` w.r.t. the variables in `vars`.
/// 
/// The object `point[i]` corresponds to the variable `vars[i]` and analogously for `direction`.
pub fn analytic_directional_derivative(
    vars: &[String],
    expr: &Expression,
    point: &[Object],
    direction: &[Object],
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Object, String> {
    match expr {
        Expression::None => Err("Cannot differentiate expression `None`.".to_string()),
        Expression::Identifier(ident) => Ok(
            if let Some(i) = vars.iter().position(|n| n == ident) { direction[i].clone() } else { Object::Float(0.0) }
        ),
        Expression::Number(_) => Ok(Object::Float(0.0)),
        Expression::Tuple(entries) => {
            Ok(Object::Tuple(
                entries
                .iter()
                .map(|x| analytic_directional_derivative(vars, x, point, direction, extra_vars, env))
                .collect::<Result<Vec<_>, _>>()
                ?
            ))
        }
        Expression::Vector(entries) => {
            let mut new_entries = Vec::<f64>::with_capacity(entries.len());
            // We instantiate this vector via a loop to allow us to return a special error message if a component can be evaluated, but not to a float.
            for x in entries {
                match analytic_directional_derivative(vars, x, point, direction, extra_vars, env) {
                    Ok(Object::Float(new_entry)) => { new_entries.push(new_entry); }
                    Ok(_) => { return Err(format!("Derivative of entry {:?} is not of type `float`.", *x))} // Entries of vector must be f64
                    other => { return other; } // Redirect error message
                }
            }
            Ok(Object::Vector(Vector {values: new_entries}))
        }
        Expression::Matrix(m, n, entries) => {
            let mut new_entries = Vec::<f64>::with_capacity(entries.len());
            for x in entries {
                match analytic_directional_derivative(vars, x, point, direction, extra_vars, env) {
                    Ok(Object::Float(new_entry)) => { new_entries.push(new_entry); }
                    Ok(_) => { return Err(format!("Derivative of entry {:?} is not of type `float`.", *x))} // Entries of vector must be f64
                    other => { return other; } // Redirect error message
                }
            }
            Ok(Object::Matrix(Matrix::from(*m, *n, new_entries)))
        }
        Expression::UnaryOperation(UnaryOperation::Neg, rhs) => {
            -&analytic_directional_derivative(vars, rhs, point, direction, extra_vars, env)?
        }
        Expression::UnaryOperation(UnaryOperation::Not, _) => Err("Cannot differentiate the operation `Not`.".to_string()),
        Expression::UnaryOperation(UnaryOperation::Factorial, _) => {
            unimplemented!() // TODO when integrals are available via \Gamma'(x) = \int_0^\infty e^{-t} t^{x-1} ln(t) dt.
        }
        Expression::UnaryOperation(UnaryOperation::Abs, rhs) => {
            let diff_r = analytic_directional_derivative(vars, rhs, point, direction, extra_vars, env)?;
            let new_frame = (0..vars.len()).map(|i| (&vars[i], &point[i])).collect();
            let varstack = VarStack::Frame { vars: &new_frame, parent: extra_vars };
            match lang::eval(rhs, &varstack, env)? {
                Object::Float(x) => if x > 0.0 {
                    Ok(diff_r)
                } else if x < 0.0 {
                    -&diff_r
                } else {
                    Ok(Object::Undefined)
                },
                other => Err(format!("Couldn't evaluate {} to float (obtained {}).", &**rhs, other))
            }
        }
        Expression::UnaryOperation(UnaryOperation::Norm(_), _) => {
            // TODO when differentiation of norm is available as partial derivative
            unimplemented!()
        }
        Expression::BinaryOperation(lhs, op, rhs) => {
            let diff_l = analytic_directional_derivative(vars, lhs, point, direction, extra_vars, env)?;
            let diff_r = analytic_directional_derivative(vars, rhs, point, direction, extra_vars, env)?;
            match op {
                BinaryOperation::Add | BinaryOperation::Sub => try_operation(&diff_l, &diff_r, op),
                BinaryOperation::Quo | BinaryOperation::Rem | BinaryOperation::And | BinaryOperation::Or => Err(format!("Cannot differentiate the operation `{op}`.")),
                BinaryOperation::Mul => {
                    let new_frame = (0..vars.len()).map(|i| (&vars[i], &point[i])).collect();
                    let varstack = VarStack::Frame { vars: &new_frame, parent: extra_vars };
                    try_operation(
                        &try_operation(&diff_l, &lang::eval(rhs, &varstack, env)?, &BinaryOperation::Mul)?, // f'(x) * g(x)
                        &try_operation(&lang::eval(lhs, &varstack, env)?, &diff_r, &BinaryOperation::Mul)?,  // f(x) * g'(x)
                        &BinaryOperation::Add
                    )
                },
                BinaryOperation::Div => {
                    let new_frame = (0..vars.len()).map(|i| (&vars[i], &point[i])).collect();
                    let varstack = VarStack::Frame { vars: &new_frame, parent: extra_vars };
                    let eval_lhs = lang::eval(lhs, &varstack, env)?;
                    let eval_rhs = lang::eval(rhs, &varstack, env)?;
                    try_operation( // d/dx (f(x) / g(x)) = (f'(x)g(x) - f(x)g'(x)) / g(x)²
                        &try_operation(
                            &try_operation(&diff_l, &eval_rhs, &BinaryOperation::Mul)?,
                            &try_operation(&eval_lhs, &diff_r, &BinaryOperation::Mul)?,
                            &BinaryOperation::Sub
                        )?,
                        &try_operation(&eval_rhs, &Object::Float(2.0), &BinaryOperation::Pow)?,
                        &BinaryOperation::Div
                    )
                }
                BinaryOperation::Pow => {
                    let new_frame = (0..vars.len()).map(|i| (&vars[i], &point[i])).collect();
                    let varstack = VarStack::Frame { vars: &new_frame, parent: extra_vars };
                    let eval_lhs = lang::eval(lhs, &varstack, env)?;
                    let eval_rhs = lang::eval(rhs, &varstack, env)?;
                    try_operation( // d/dx (f(x) ^ g(x)) = f(x)^(g(x)-1) * (f'(x)g(x) + f(x)g'(x)ln(f(x)))
                        &try_operation(
                            &eval_lhs,
                            &try_operation(&eval_rhs, &Object::Float(1.0), &BinaryOperation::Sub)?,
                            &BinaryOperation::Pow
                        )?,
                        &try_operation(
                            &try_operation(&diff_l, &eval_rhs, &BinaryOperation::Mul)?,
                            // The following argument `rhs` should be f(x)g'(x)ln(f(x)). However, if g'(x) = 0, then
                            // f(x) may be negative, so we then want to avoid calling f(x).ln().
                            &match (try_operation(&eval_lhs, &diff_r, &BinaryOperation::Mul)?, eval_lhs) {
                                (Object::Float(x), _) if approx_eq(x, 0.0) => Ok(Object::Float(0.0)),
                                (l, Object::Float(x)) => try_operation(&l, &Object::Float(x.ln()), &BinaryOperation::Mul),
                                _ => {return Err(format!("Evaluation of {:?} is not of type `float`.", lhs));},
                            }?,
                            &BinaryOperation::Add
                        )?,
                        &BinaryOperation::Mul
                    )
                }
                BinaryOperation::Comp(..) => Err(format!("Cannot differentiate comparison {:?}", expr)),
            }
        }
        Expression::FoldedOperation(FoldedOperation::Sum, varname, from, conditions, to, inner) => {
            // Note: since the bounds of the sum must be integers, taking them into consideration when differentiating is useless.
            // Therefore, we simply treat `D sum_{i=a}^b ...(p)[d]` as `sum_{i=a(p)}^{b(p)} D ... (p)[d]`.
            // The following code is adapted from lang::evaluator::eval (case Expression::FoldedOperation).
            // Copying and adapting it is more efficient than to try to call `eval` instead.
            let varstack = VarStack::Frame { vars: &zip(vars, point).collect(), parent: extra_vars };
            let mut i = lang::eval(from, &varstack, env)?.expect_int()?;
            if i > lang::eval(to, &VarStack::Frame { vars: &HashMap::from([(varname, &Object::Float(i))]), parent: &varstack }, env)?.expect_float()? {
                // TODO when the corresponding TODO in eval is done
                return Ok(Object::Float(0.0));
            }
            let mut res = Object::Float(0.0); // TODO same
            'outer: while i <= lang::eval(to, &VarStack::Frame { vars: &HashMap::from([(varname, &Object::Float(i))]), parent: &varstack }, env)?.expect_float()? {
                for cond in conditions {
                    match lang::eval(cond, &VarStack::Frame { vars: &HashMap::from([(varname, &Object::Float(i))]), parent: extra_vars }, env)? {
                        Object::Float(1.0) => {}
                        Object::Float(0.0) => { i += 1.0; continue 'outer; }
                        other => return Err(format!("Expected 1 or 0 when evaluating condition, got {:?}.", other))
                    }
                }
                let next_term = analytic_directional_derivative(
                    vars, inner, point, direction,
                    &VarStack::Frame { vars: &HashMap::from([(varname, &Object::Float(i))]), parent: extra_vars },
                    env
                )?;
                res = try_operation(&res, &next_term, &BinaryOperation::Add)?;
                i += 1.0;
            }
            Ok(res)
        }
        Expression::FoldedOperation(FoldedOperation::Product, varname, from, conditions, to, inner) => {
            // TODO
            unimplemented!()
        }
        Expression::Function(function_name, arg_expressions) => {
            // For simplicity, I'll subsequently write `f` instead of `function_name`.
            // Define `g` such that `f(arg_expressions) = f(g(wrt))`. We aim to use the chain rule:
            //     D(f \circ g)(p)[d] = Df(g(p))[Dg(p)[d]]
            // First, compute Dg(p)[d], which may be a vector, so simply differentiate componentwise.
            let differentiated_components_of_g = arg_expressions.iter().map(
                |g_i| analytic_directional_derivative(vars, g_i, point, direction, extra_vars, env)
            ).collect::<Result<Vec<_>, _>>()?;
            // Then, compute g(p).
            let new_frame = (0..vars.len()).map(|i| (&vars[i], &point[i])).collect();
            let varstack = VarStack::Frame { vars: &new_frame, parent: extra_vars };
            let g_of_point = arg_expressions.iter().map(
                |g_i| lang::eval(g_i, &varstack, env)
            ).collect::<Result<Vec<_>, _>>()?;
            // Finally, apply the chain rule. If `f` has a representation via expression, we can get Df by a recursive call of this function.
            // In case of a direct representation, we have to fall back on a numerical directional derivative.
            let reinsert_later = env.functions.remove(function_name).ok_or(format!("No such function: {}", function_name))?;
            let res = match reinsert_later {
                FunctionRepr::ByExpression(ref argnames, ref function_expr) => analytic_directional_derivative(
                    argnames, function_expr, &g_of_point, &differentiated_components_of_g, &varstack, env
                ),
                FunctionRepr::Direct(ref f) => numerical_directional_derivative(
                    f, g_of_point, differentiated_components_of_g
                )
            };
            env.functions.insert(function_name.clone(), reinsert_later);
            res
        }
        // You can't differentiate expressions like `y := ...`, that makes no sense. If the user wants `y := d/dx ...`, he should have typed that. 
        Expression::Assignment(..) => Err("Assignment cannot be differentiated.".to_string()),
        Expression::PartialDerivative(wrt_other, inner) => {
            // Idea is simple: d/dx (d/dy f(x, y)) -> First evaluate the inner derivative, then differentiate the result.
            analytic_directional_derivative(vars, &analytic_partial_derivative(inner, wrt_other, extra_vars, env)?, point, direction, extra_vars, env)
        }
        Expression::DirectionalDerivative(..) => {
            // The directional derivative is an object, so whatever it actually is, its derivative is zero.
            Ok(Object::Float(0.0))
        }
        Expression::IfElse(condition, iftrue, iffalse) => {
            // D (if c(x) {a(x)} else {b(x)})(x)[d] = if c(x) {Da(x)[d]} else {Db(x)[d]}
            let new_frame = (0..vars.len()).map(|i| (&vars[i], &point[i])).collect();
            let varstack = VarStack::Frame { vars: &new_frame, parent: extra_vars };
            match lang::eval(condition, &varstack, env) {
                Ok(Object::Float(1.0)) => analytic_directional_derivative(vars, iftrue, point, direction, &varstack, env),
                Ok(Object::Float(0.0)) => analytic_directional_derivative(vars, iffalse, point, direction, &varstack, env),
                Ok(x) => Err(format!("Couldn't evaluate condition {condition} to 0 or 1; got {x}")),
                other => other
            }
        }
    }
}

/// For `f: R -> R^{mxn}`, we could use the "three-point central difference formula" (proof by Taylor expansion):
///     `f'(x) = \frac{f(x+h) - f(x-h)}{2h} + O(h²)`
/// for `h` close to zero (here, `h = 1e-9`).
/// 
/// For general `f`, we generalize this method.
/// 
/// Note: this can also be used for functions from `\R` to `\R` by using `direction = vec![Object::Float(1.0)]`.
/// 
/// Unfortunately, `point` has to be owned (or we'd have to clone it) since we want to modify it and the original passed vector need not to be mutable.
/// Moreover, also owning `direction` allows to decrease the number of required operations.
pub fn numerical_directional_derivative(f: &DirectFunction, mut point: Vec<Object>, mut direction: Vec<Object>) -> Result<Object, String> {
    if point.len() != direction.len() {
        return Err("`point` and `direction` for derivative must be vectors of the same length (possibly 1).".to_string());
    }
    let h = 1e-9;
    for (i, coord) in point.iter_mut().enumerate() {
        direction[i] = h * &direction[i]; // Spares us another operation later
        *coord = try_operation(coord, &direction[i], &BinaryOperation::Add)?; // point + h*direction
    }
    let left_res = f(&point)?;
    for (i, coord) in point.iter_mut().enumerate() {
        // If the previous loop worked, this one will too.
        *coord = try_operation(coord, &(2.0 * &direction[i]), &BinaryOperation::Sub).unwrap();
    }
    let right_res = f(&point)?;
    match (left_res, right_res) {
        (Object::Float(lhs), Object::Float(rhs)) => Ok(Object::Float((lhs - rhs) / (2.0 * h))),
        (Object::Vector(lhs), Object::Vector(rhs)) => {
            Ok(Object::Vector(
                &(&lhs - &rhs).ok_or("Couldn't evaluate f(x+h) - f(x-h). Traceback: Vectors of different sizes returned.")?
                / (2.0 * h)
            ))
        }
        (Object::Matrix(lhs), Object::Matrix(rhs)) => {
            Ok(Object::Matrix(
                &(&lhs - &rhs).ok_or("Couldn't evaluate f(x+h) - f(x-h). Traceback: Vectors of different sizes returned.")?
                / (2.0 * h)
            ))
        }
        _ => Err("Couldn't evaluate f(x+h) - f(x-h). Traceback: Objects have different types.".to_string())
    }
}