use std::collections::HashMap;

use dioxus_logger::tracing;

use crate::math::objects::{try_operation};
use crate::math::expressions::*;
use crate::math::utils::approx_eq;
use crate::math::{Object, DirectFunction, FunctionRepr, Vector, Matrix};
use crate::math::operations::{BinaryOperation, UnaryOperation};
use crate::parser;

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
    functions: &HashMap<String, FunctionRepr>
) -> Result<Expression, String> {
    match expr {
        Expression::None => Ok(Expression::None),
        Expression::Identifier(ident) => Ok(Expression::Number(if ident == wrt { 1.0 } else { 0.0 })),
        Expression::Number(_) => Ok(Expression::Number(0.0)),
        Expression::Vector(entries) => {
            Ok(Expression::Vector(
                entries.iter()
                .map(|x| analytic_partial_derivative(x, wrt, functions))
                .collect::<Result<Vec<_>, _>>()?
            ))
        }
        Expression::Matrix(m, n, entries) => {
            Ok(Expression::Matrix(*m, *n,
                entries.iter()
                .map(|x| analytic_partial_derivative(x, wrt, functions))
                .collect::<Result<Vec<_>, _>>()?
            ))
        }
        Expression::UnaryOperation(UnaryOperation::Neg, rhs) => {
            Ok(Expression::UnaryOperation(UnaryOperation::Neg, Box::new(
                analytic_partial_derivative(rhs, wrt, functions)?
            )))
        }
        Expression::UnaryOperation(UnaryOperation::Abs, rhs) => {
            // TODO. Safe to say that `wrt` is real.
            // Add "if (...) {...} else {...}" as Expression so this can differentiate the cases x!=0 and x==0 in a mathematically correct way.
            unimplemented!()
        }
        Expression::BinaryOperation(lhs, op, rhs) => {
            let diff_l = analytic_partial_derivative(lhs, wrt, functions)?;
            let diff_r = analytic_partial_derivative(rhs, wrt, functions)?;
            tracing::info!("Diff_l: {}", diff_l); tracing::info!("Diff_r: {}", diff_r);
            match op {
                BinaryOperation::Add => Ok(simplify_add(diff_l, diff_r)),
                BinaryOperation::Sub => Ok(simplify_sub(diff_l, diff_r)),
                BinaryOperation::Quo | BinaryOperation::Rem => Err(format!("Differentiating the operation {} makes no sense.", op)),
                BinaryOperation::Mul => Ok(simplify_add(
                    simplify_mul(diff_l, *rhs.clone()), // f'(x) * g(x)
                    simplify_mul(*lhs.clone(), diff_r)  // f(x) * g'(x)
                )),
                BinaryOperation::Div => Ok(simplify_div( // d/dx (f(x) / g(x)) = (f'(x)g(x) - f(x)g'(x)) / g(x)²
                    simplify_sub(
                        simplify_mul(diff_l, *rhs.clone()),
                        simplify_mul(*lhs.clone(), diff_r)
                    ),
                    simplify_pow(*lhs.clone(), Expression::Number(2.0))
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
        Expression::Function(function_name, g_exprs) => {
            // Standard trick. To be able to create mutable references of `functions` within the `match` block, we don't call
            // `functions.get` but `functions.remove` and later reinsert the function. The only caveat is that we'll have
            // to clone `function_name` once, but this is fast since `function_name` typically only is `f`, `g`, etc.
            // For simplicity, I'll subsequently write `f` instead of `function_name`.
            // Define `g` such that `f(arg_expressions) = f(g(wrt))`. This explains the above name `g_exprs`
            match functions.get(function_name) {
                Some(FunctionRepr::ByExpression(f_argnames, f_expr)) => {
                    // As discussed in the case `FunctionRepr::Direct`, we aim to return `Df(g(x))[Dg(x)[1]]` as an expression,
                    // not as a value.
                    if g_exprs.len() == 1 {
                        let mut diff_f = analytic_partial_derivative(f_expr, &f_argnames[0], functions)?;
                        diff_f.replace_identifiers(&f_argnames[0], &g_exprs[0]); // Plug in g(x) into f'
                        // If g only outputs one value, we can simply apply the 1d chain rule, (f \circ g)'(x) = g'(x) * f'(g(x)).
                        Ok(Expression::BinaryOperation(
                            Box::new(analytic_partial_derivative(&g_exprs[0], wrt, functions)?),
                            BinaryOperation::Mul,
                            Box::new(diff_f)
                        ))
                    }
                    else {
                        // Otherwise, the idea is to resolve Dg(x)[1] and then return an `Expression::DirectionalDerivative`.
                        Ok(Expression::DirectionalDerivative(
                            f_argnames.clone(),
                            Box::new(f_expr.clone()),
                            g_exprs.clone(),
                            g_exprs.iter()
                                .map(|g_i| analytic_partial_derivative(g_i, wrt, functions))
                                .collect::<Result<Vec<_>, _>>()?
                        ))
                    }
                }
                Some(FunctionRepr::Direct(_)) => {
                    // Importantly, note that the directional derivative is a separate function. Therefore, we can assume w.l.o.g. that `f \circ g` maps from `\R` to `\R`.
                    // For each component of `g` (note that `g` maps from `\R` to `\R^n`), analytically differentiate that component w.r.t. `wrt` (which is the input of `g`).
                    // We save these into a vector already to avoid calling `analytic_derivative` more often than necessary.
                    // The returned expression should be (writing `x` for `wrt`)
                    // ```d/dx f(g(x)) |_x
                    //     = D(f \circ g)(x)[1]        (since `f \circ g` maps from `\R` to `\R`)
                    //     = Df(g(x))[Dg(x)[1]]        (chain rule)```
                    // In the program's syntax, this is equivalent to calling `___diff_num_f` with arguments `arg_expressions` concatenated with `(d/dx g_1, ... d/dx g_n)})`
                    let mut new_args = g_exprs.clone(); // We don't own `expr`, so we have to clone `arg_expressions`.
                    new_args.reserve(g_exprs.len());
                    new_args.extend(g_exprs.iter()
                        .map(|g_i| analytic_partial_derivative(g_i, wrt, functions))
                        .collect::<Result<Vec<_>, _>>()?);
                    Ok(Expression::Function(
                        format!("___diff_num_{}", function_name),
                        new_args
                    ))
                }
                None => Err(format!("No such function: {}", function_name))
            }
        }
        // You can't differentiate expressions like `y := ...`, that makes no sense. If the user wants `y := d/dx ...`, he should have typed that. 
        Expression::Assignment(..) => Err("Assignment cannot be differentiated.".to_string()),
        Expression::PartialDerivative(wrt_other, inner) => {
            // Idea is simple: d/dx (d/dy f(x, y)) -> First evaluate the inner derivative, then differentiate the result.
            let res = analytic_partial_derivative(inner, wrt_other, functions)?;
            tracing::info!("{}", res);
            analytic_partial_derivative( // Outer derivative
                &res, // Inner derivative
                wrt, functions
            )
        }
        Expression::DirectionalDerivative(..) => {
            // The directional derivative is an object, so whatever it actually is, its derivative is zero.
            Ok(Expression::Number(0.0))
        }
    }
}

/// Analytically differentiates `expr` at point `point` in direction `direction` w.r.t. the variables in `vars`.
/// 
/// The object `point[i]` corresponds to the variable `vars[i]` and analogously for `direction`.
pub fn analytic_directional_derivative(
    vars: &Vec<String>,
    expr: &Expression,
    point: &Vec<Object>,
    direction: &Vec<Object>,
    constants: &mut HashMap<String, Object>,
    functions: &mut HashMap<String, FunctionRepr>
) -> Result<Object, String> {
    match expr {
        Expression::None => Err("Cannot differentiate expression `None`.".to_string()),
        Expression::Identifier(ident) => Ok(
            if let Some(i) = vars.iter().position(|n| n == ident) { direction[i].clone() } else { Object::Float(0.0) }
        ),
        Expression::Number(_) => Ok(Object::Float(0.0)),
        Expression::Vector(entries) => {
            let mut new_entries = Vec::<f64>::with_capacity(entries.len());
            // We instantiate this vector via a loop to allow us to return a special error message if a component can be evaluated, but not to a float.
            for x in entries {
                match analytic_directional_derivative(vars, x, point, direction, constants, functions) {
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
                match analytic_directional_derivative(vars, x, point, direction, constants, functions) {
                    Ok(Object::Float(new_entry)) => { new_entries.push(new_entry); }
                    Ok(_) => { return Err(format!("Derivative of entry {:?} is not of type `float`.", *x))} // Entries of vector must be f64
                    other => { return other; } // Redirect error message
                }
            }
            Ok(Object::Matrix(Matrix::from(*m, *n, new_entries)))
        }
        Expression::UnaryOperation(UnaryOperation::Neg, rhs) => {
            Ok(-&analytic_directional_derivative(vars, rhs, point, direction, constants, functions)?)
        }
        Expression::UnaryOperation(UnaryOperation::Abs, rhs) => {
            // TODO: somehow determine what the RHS outputs? Probably not efficiently doable.
            // Rather, replace UnaryOperation::Abs with Abs, Norm, Det. Then split this into 3 cases.
            // Also, add "if (...) {...} else {...}" as Expression so this can differentiate the cases x!=0 and x==0 in a mathematically correct way.
            unimplemented!()
        }
        Expression::BinaryOperation(lhs, op, rhs) => {
            let diff_l = analytic_directional_derivative(vars, lhs, point, direction, constants, functions)?;
            let diff_r = analytic_directional_derivative(vars, rhs, point, direction, constants, functions)?;
            match op {
                BinaryOperation::Add | BinaryOperation::Sub => try_operation(&diff_l, &diff_r, op),
                BinaryOperation::Quo | BinaryOperation::Rem => Err(format!("Differentiating the operation {} makes no sense.", op)),
                BinaryOperation::Mul => {
                    let extra_vars = (0..vars.len()).map(|i| (&vars[i], &point[i])).collect();
                    try_operation(
                        &try_operation(&diff_l, &parser::eval(rhs, &extra_vars, constants, functions)?, &BinaryOperation::Mul)?, // f'(x) * g(x)
                        &try_operation(&parser::eval(lhs, &extra_vars, constants, functions)?, &diff_r, &BinaryOperation::Mul)?,  // f(x) * g'(x)
                        &BinaryOperation::Add
                    )
                },
                BinaryOperation::Div => {
                    let extra_vars = (0..vars.len()).map(|i| (&vars[i], &point[i])).collect();
                    let eval_lhs = parser::eval(lhs, &extra_vars, constants, functions)?;
                    try_operation( // d/dx (f(x) / g(x)) = (f'(x)g(x) - f(x)g'(x)) / g(x)²
                        &try_operation(
                            &try_operation(&diff_l, &parser::eval(rhs, &extra_vars, constants, functions)?, &BinaryOperation::Mul)?,
                            &try_operation(&eval_lhs, &diff_r, &BinaryOperation::Mul)?,
                            &BinaryOperation::Sub
                        )?,
                        &try_operation(&eval_lhs, &Object::Float(2.0), &BinaryOperation::Pow)?,
                        &BinaryOperation::Div
                    )
                }
                BinaryOperation::Pow => {
                    let extra_vars = (0..vars.len()).map(|i| (&vars[i], &point[i])).collect();
                    let eval_lhs = parser::eval(lhs, &extra_vars, constants, functions)?;
                    let eval_rhs = parser::eval(rhs, &extra_vars, constants, functions)?;
                    tracing::info!("LHS: {}; RHS: {}; dLHS: {}; dRHS: {}", eval_lhs, eval_rhs, diff_l, diff_r);
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
                                (Object::Float(x), _) if approx_eq(&x, &0.0) => Ok(Object::Float(0.0)),
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
        Expression::Function(function_name, arg_expressions) => {
            // For simplicity, I'll subsequently write `f` instead of `function_name`.
            // Define `g` such that `f(arg_expressions) = f(g(wrt))`. We aim to use the chain rule:
            //     D(f \circ g)(p)[d] = Df(g(p))[Dg(p)[d]]
            // First, compute Dg(p)[d], which may be a vector, so simply differentiate componentwise.
            let differentiated_components_of_g = arg_expressions.iter().map(
                |g_i| analytic_directional_derivative(vars, g_i, point, direction, constants, functions)
            ).collect::<Result<Vec<_>, _>>()?;
            // Then, compute g(p).
            let extra_vars = (0..vars.len()).map(|i| (&vars[i], &point[i])).collect();
            let g_of_point = arg_expressions.iter().map(
                |g_i| parser::eval(g_i, &extra_vars, constants, functions)
            ).collect::<Result<Vec<_>, _>>()?;
            // Finally, apply the chain rule. If `f` has a representation via expression, we can get Df by a recursive call of this function.
            // In case of a direct representation, we have to fall back on a numerical directional derivative.
            let reinsert_later = functions.remove(function_name).ok_or(format!("No such function: {}", function_name))?;
            let res = match reinsert_later {
                FunctionRepr::ByExpression(ref argnames, ref function_expr) => analytic_directional_derivative(
                    argnames, function_expr, &g_of_point, &differentiated_components_of_g, constants, functions
                ),
                FunctionRepr::Direct(ref f) => numerical_directional_derivative(
                    f, g_of_point, differentiated_components_of_g
                )
            };
            functions.insert(function_name.clone(), reinsert_later);
            res
        }
        // You can't differentiate expressions like `y := ...`, that makes no sense. If the user wants `y := d/dx ...`, he should have typed that. 
        Expression::Assignment(..) => Err("Assignment cannot be differentiated.".to_string()),
        Expression::PartialDerivative(wrt_other, inner) => {
            // Idea is simple: d/dx (d/dy f(x, y)) -> First evaluate the inner derivative, then differentiate the result.
            analytic_directional_derivative(vars, &analytic_partial_derivative(inner, wrt_other, functions)?, point, direction, constants, functions)
        }
        Expression::DirectionalDerivative(..) => {
            // The directional derivative is an object, so whatever it actually is, its derivative is zero.
            Ok(Object::Float(0.0))
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