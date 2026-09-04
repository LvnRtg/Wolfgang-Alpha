use std::borrow::Cow;

use crate::{defaults, expr_1arg_func, expr_binop, expr_compare, expr_if_else, expr_unary_op};
use crate::lang::eval;
use crate::math::{Env, Expression, FunctionRepr, integration, Object, ObjType, Matrix, VarStack, Vector};
use crate::math::expressions::simplification::*;
use crate::math::matrices_and_vectors::{VectorNorm, MatrixNorm};
use crate::math::objects::{try_operation};
use crate::math::operations::{BinaryOperation, FoldedOperation, UnaryOperation};
use crate::math::operations::folded_operations::{compute_folded_operation, compute_product_derivative_helper};
use crate::math::utils::{approx_eq, min};
use crate::status::{ExtResult, Status};

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
) -> Result<Status<Expression>, String> {
    match expr {
        Expression::None => Ok(Status::ok(Expression::None)),
        Expression::Identifier(ident) => Ok(Status::ok(Expression::Number(if ident == wrt { 1.0 } else { 0.0 }))),
        Expression::Number(_) => Ok(Status::ok(Expression::Number(0.0))),
        Expression::Tuple(entries) => Status::from_iter(
            entries.iter(),
            |x| analytic_partial_derivative(x, wrt, extra_vars, env)
        ).map(|s| s.map(Expression::Tuple)),
        Expression::Vector(entries) => Status::from_iter(
            entries.iter(),
            |x| analytic_partial_derivative(x, wrt, extra_vars, env)
        ).map(|s| s.map(Expression::Vector)),
        Expression::Matrix(m, n, entries) => Status::from_iter(
            entries.iter(),
            |x| analytic_partial_derivative(x, wrt, extra_vars, env)
        ).map(|s| s.map(|val| Expression::Matrix(*m, *n, val))),
        Expression::UnaryOperation(UnaryOperation::Neg, rhs) => analytic_partial_derivative(rhs, wrt, extra_vars, env).map(
            |s| s.map(
                |e| expr_unary_op!(Neg, e)
            )
        ),
        Expression::UnaryOperation(UnaryOperation::Not, _) => Err("Cannot differentiate the operation `Not`.".to_string()),
        Expression::UnaryOperation(UnaryOperation::Factorial, inner) => {
            // We treat the factorial as gamma function here since differentiating a function with discrete domain makes no sense.
            // We use: d/dx x! = Γ'(x+1) = \int_0^\infty e^{-t} t^x ln(t) dt
            //      => d/dx f(x)! = f'(x) * \int_0^\infty e^{-t} t^{f(x)} ln(t) dt
            let wrt = inner.get_new_free_identifier("t");
            Ok(Status::ok(Expression::Integral(
                Box::new(expr_binop!(
                    expr_1arg_func!("exp", expr_unary_op!(Neg, Expression::Identifier(wrt.clone()))),
                    Mul,
                    expr_1arg_func!("ln", Expression::Identifier(wrt.clone())),
                    expr_binop!(Expression::Identifier(wrt.clone()), Pow(true), *inner.clone())
                )),
                Box::new(Expression::Number(0.0)),
                Box::new(Expression::Identifier("inf".to_string())),
                wrt
            )))
        }
        Expression::UnaryOperation(UnaryOperation::Abs, rhs) => analytic_partial_derivative(rhs, wrt, extra_vars, env)
        .map(|s| s.map(
            |diff_r| expr_if_else!(
                expr_compare!(*rhs.clone(), Gt, Expression::Number(0.0)),
                diff_r.clone(),
                expr_if_else!(
                    expr_compare!(*rhs.clone(), Lt, Expression::Number(0.0)),
                    expr_unary_op!(Neg, diff_r.clone()),
                    Expression::None
                )
            )
        )),
        Expression::UnaryOperation(UnaryOperation::Norm(opt), rhs) => {
            let (rhs_toplevel, rhs_type) = rhs.make_type_top_level(
                false,
                &extra_vars.with(wrt, Cow::Owned(Object::Real(1.0))),
                env
            )?;
            match rhs_toplevel {
                Expression::Vector(components) => apd_of_norm_for_vector(wrt, opt, &components, extra_vars, env),
                Expression::Matrix(m, n, components) => apd_of_norm_for_matrix(wrt, opt, &components, m, n, extra_vars, env),
                _ if matches!(rhs_type, ObjType::Tuple | ObjType::NonObject) => Err(format!("Operation 'Norm' invalid for operand {:?}.", rhs_toplevel)),
                other if rhs_type == ObjType::LiteralExpression => Ok(Status::ok(Expression::UnaryOperation(UnaryOperation::Norm(opt.clone()), Box::new(other)))),
                other => { // Scalar type
                    // In this case, the norm should simply be an absolute value, regardless of `opt`.
                    analytic_partial_derivative(&other, wrt, extra_vars, env)
                    .map(|s| s.map(
                        |diff_r| expr_if_else!(
                            expr_compare!(other.clone(), Gt, Expression::Number(0.0)),
                            diff_r.clone(),
                            expr_if_else!(
                                expr_compare!(other, Lt, Expression::Number(0.0)),
                                expr_unary_op!(Neg, diff_r),
                                Expression::None
                            )
                        )
                    ))
                }
            }
            
        }
        Expression::BinaryOperation(lhs, op, rhs) => Status::combine(
            analytic_partial_derivative(lhs, wrt, extra_vars, env)?,
            analytic_partial_derivative(rhs, wrt, extra_vars, env)?,
            |diff_l, diff_r| match op {
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
                BinaryOperation::Pow(_) => Ok(simplify_mul( // d/dx (f(x) ^ g(x)) = f(x)^(g(x)-1) * (f'(x)g(x) + f(x)g'(x)ln(f(x)))
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
        ),
        Expression::FoldedOperation(FoldedOperation::Sum, varname, from, conditions, to, inner) => {
            // As for the product (but slightly simpler since `d/dx sum_{i=a}^b f(i, x) = sum_{i=a}^b d/dx f(i, x)`),
            // see case `Expression::FoldedOperation` below.
            let Status{value: inner_diff, mut warnings} = analytic_partial_derivative(inner, wrt, extra_vars, env)?;
            match (from.contains_identifier(wrt), to.contains_identifier(wrt)) { // Typically, both expressions must be checked anyway
                (true, true) => warnings.push(format!("Assuming that both `{}` and `{}` are continuous in {} to differentiate product.", from, to, wrt)),
                (true, false) => warnings.push(format!("Assuming that `{}` is continuous in {} to differentiate product.", from, wrt)),
                (false, true) => warnings.push(format!("Assuming that `{}` is continuous in {} to differentiate product.", to, wrt)),
                (false, false) => {}
            };
            Ok(Status{
                value: Expression::FoldedOperation(
                    FoldedOperation::Sum,
                    varname.clone(),
                    from.clone(),
                    conditions.clone(),
                    to.clone(),
                    Box::new(inner_diff)
                ),
                warnings
            })
        }
        Expression::FoldedOperation(FoldedOperation::Product, index_var, from, conditions, to, inner) => {
            // If the product is of the form `\prod_{i=a(x)}^{b(x)} f(i,x)`, it isn't immediately clear how to differentiate it w.r.t. `x`.
            // However, as long as `a, b` are sufficiently well-behaved (it suffices for them to be continuous or be càdlàg/càglàd with jumps of size <1),
            // we will have `\prod_{i=a(y)}^{b(y)} f(i,y) = \prod_{i=a(x)}^{b(x)} f(i,y)` for `|x-y|` sufficiently small, so
            // `d/dx \prod_{i=a(x)}^{b(x)} f(i,x) |_{x_0} = d/dx \prod_{i=a(x_0)}^{b(x_0)} f(i,x) |_{x_0}`,
            // meaning we can apply the standard product rule.
            // I have never encountered a product that is of this form for badly behaved `a, b`, so I decided that `a, b` are always assumed
            // to be continuous so we can apply the above formula. The same assumption is made for the conditions. A warning is then emitted.
            let Status{value: inner_diff, mut warnings} = analytic_partial_derivative(inner, wrt, extra_vars, env)?;
            match (from.contains_identifier(wrt), to.contains_identifier(wrt)) { // Typically, both expressions must be checked anyway
                (true, true) => warnings.push(format!("Assuming that both `{}` and `{}` are continuous in {} to differentiate product.", from, to, wrt)),
                (true, false) => warnings.push(format!("Assuming that `{}` is continuous in {} to differentiate product.", from, wrt)),
                (false, true) => warnings.push(format!("Assuming that `{}` is continuous in {} to differentiate product.", to, wrt)),
                (false, false) => {}
            };
            let mut args = vec![
                Expression::Identifier(wrt.clone()), // x, once to be evaluated
                Expression::Identifier(wrt.clone()), // x, once as literal expression
                Expression::Identifier(index_var.clone()), // i
                *from.clone(), // a(x)
                *to.clone(), // b(x)
                *inner.clone(), // f(i, x)
                inner_diff // f'(i, x)
            ];
            args.append(&mut conditions.clone()); // Add conditions at the end
            Ok(Status{
                value: Expression::Function("___helper_prod_rule".to_string(), args),
                warnings
            })
        }
        Expression::Function(function_name, g_exprs) => {
            // Standard trick. To be able to create mutable references of `functions` within the `match` block, we don't call
            // `functions.get` but `functions.remove` and later reinsert the function. The only caveat is that we'll have
            // to clone `function_name` once, but this is fast since `function_name` typically only is `f`, `g`, etc.
            // For simplicity, I'll subsequently write `f` instead of `function_name`.
            // Define `g` such that `f(arg_expressions) = f(g(wrt))`. This explains the above name `g_exprs`
            let f = env.functions.remove(function_name).ok_or(format!("No such function \"{}\".", function_name))?;
            let res = apd_for_function(wrt, function_name, &f, g_exprs.clone(), extra_vars, env);
            env.functions.insert(function_name.clone(), f);
            res
        }
        // You can't differentiate expressions like `y := ...`, that makes no sense. If the user wants `y := d/dx ...`, he should have typed that. 
        Expression::Assignment(..) => Err("Assignment cannot be differentiated.".to_string()),
        Expression::PartialDerivative(wrt_other, inner) => {
            // Idea is simple: d/dx (d/dy f(x, y)) -> First evaluate the inner derivative, then differentiate the result.
            analytic_partial_derivative(inner, wrt_other, extra_vars, env)
            .and_then(|s| s.try_map_flatten(
                |diff_inner| analytic_partial_derivative(&diff_inner, wrt, extra_vars, env)
            ))
        }
        // The directional derivative is an object, so whatever it actually is, its derivative is zero.
        Expression::DirectionalDerivative(..) => Ok(Status::ok(Expression::Number(0.0))),
        Expression::Integral(inner, a, b, int_var) => {
            // Since we can't always exchange differentiation and integration, we proceed as follows. First,
            // check if the integral is of the special form \int_{a(x)}^{b(x)} h(y) dy where x = wrt and h does not involve x.
            // Then, the derivative would be h(b(x)) b'(x) - h(a(x)) a'(x). Otherwise, define a hidden function ___int_...(x) := \int_{a(x)}^{b(x)} h(x, y) dy
            // and return ___diff_num_...
            if !inner.contains_identifier(wrt) {
                Status::combine(
                    analytic_partial_derivative(a, wrt, extra_vars, env)?,
                    analytic_partial_derivative(b, wrt, extra_vars, env)?,
                    |da, db| Ok(expr_binop!(
                        simplify_mul(
                            inner.replace_identifiers(int_var, &b.clone()),
                            db
                        ),
                        Sub,
                        simplify_mul(
                            inner.replace_identifiers(int_var, &a.clone()),
                            da
                        )
                    ))
                )
            } else {
                let n = (0..).find(|i| !env.functions.contains_key(&format!("___int_{i}"))).unwrap();
                let function_name = format!("___int_{n}");
                let res = apd_for_direct_function(wrt, &function_name, vec![Expression::Identifier(int_var.clone())], extra_vars, env);
                env.functions.insert(function_name, FunctionRepr::ByExpression(vec![wrt.clone()], expr.clone()));
                res
            }
        }
        Expression::IfElse(x, y, z) => Status::combine(
            analytic_partial_derivative(y, wrt, extra_vars, env)?,
            analytic_partial_derivative(z, wrt, extra_vars, env)?,
            |y, z| Ok(expr_if_else!(*x.clone(), y, z))
        )
    }
}

/// Computes the analytic partial derivative of `f(*g_exprs)` w.r.t. `wrt`.
fn apd_for_function(
    wrt: &String,
    function_name: &String,
    f: &FunctionRepr,
    g_exprs: Vec<Expression>,
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Status<Expression>, String> {
    match f {
        FunctionRepr::ByExpression(f_argnames, f_expr) => {
            // As discussed in the case `FunctionRepr::Direct`, we aim to return `Df(g(x))[Dg(x)[1]]` as an expression,
            // not as a value.
            if g_exprs.len() == 1 {
                Status::combine(
                    analytic_partial_derivative(f_expr, &f_argnames[0], extra_vars, env)?,
                    analytic_partial_derivative(&g_exprs[0], wrt, extra_vars, env)?,
                    |mut diff_f, diff_g| {
                        diff_f.replace_identifiers_in_place(&f_argnames[0], &g_exprs[0]); // Plug in g(x) into f'
                        // If g only outputs one value, we can simply apply the 1d chain rule, (f \circ g)'(x) = g'(x) * f'(g(x)).
                        Ok(simplify_mul(diff_g, diff_f))
                    }
                )
            } else {
                // Otherwise, the idea is to resolve Dg(x)[1] and then return an `Expression::DirectionalDerivative`.
                Status::from_iter(
                    g_exprs.iter(),
                    |g_i| analytic_partial_derivative(g_i, wrt, extra_vars, env)
                )
                .map(|s| s.map(|direction| Expression::DirectionalDerivative(
                    f_argnames.clone(),
                    Box::new(f_expr.clone()),
                    g_exprs,
                    direction
                )))
            }
        }
        FunctionRepr::Direct(..) => {
            // If `function_name` refers to a default function (e.g. `exp`), we can spare ourselves the below code.
            if defaults::FUNCTIONS_WITH_PROVIDED_DERIVATIVE.contains(&function_name.as_str()) {
                // Similar to the chain rule block in `analytic_directional_derivative`, with a little change: with the same f, g as there, we have
                //     D(f \circ g)(x)[1.0] = Df(g(x))[Dg(x)[1.0]]
                Status::from_iter(
                    g_exprs.iter(),
                    |g_i| analytic_partial_derivative(g_i, wrt, extra_vars, env)
                )
                .and_then(|s| s.try_map(
                    |differentiated_components_of_g|
                    defaults::get_default_derivative(function_name.as_str(), &g_exprs, &differentiated_components_of_g)
                ))
            } else {
                apd_for_direct_function(wrt, function_name, g_exprs, extra_vars, env)
            }
        }
    }
}

/// Computes the analytic partial derivative of `f(*g_exprs)` w.r.t. `wrt` where `f` is an existing function in `env` with direct representation
/// by simply packing it into a `__diff_num` expression.
/// 
/// Note: this also works if `f` has another representation but shouldn't be used in that case since it loses precision.
fn apd_for_direct_function(
    wrt: &String,
    function_name: &String,
    mut g_exprs: Vec<Expression>,
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Status<Expression>, String> {
    // Importantly, note that the directional derivative is a separate function. Therefore, we can assume w.l.o.g. that `f \circ g` maps from `\R` to `\R`.
    // For each component of `g` (note that `g` maps from `\R` to `\R^n`), analytically differentiate that component w.r.t. `wrt` (which is the input of `g`).
    // We save these into a vector already to avoid calling `analytic_derivative` more often than necessary.
    // The returned expression should be (writing `x` for `wrt`)
    // ```d/dx f(g(x)) |_x
    //     = D(f \circ g)(x)[1]        (since `f \circ g` maps from `\R` to `\R`)
    //     = Df(g(x))[Dg(x)[1]]        (chain rule)```
    // In the program's syntax, this is equivalent to calling `___diff_num_f` with arguments `arg_expressions` concatenated with `(d/dx g_1, ... d/dx g_n)})`
    g_exprs.reserve(g_exprs.len());
    Status::from_iter(
        g_exprs.iter(),
        |g_i| analytic_partial_derivative(g_i, wrt, extra_vars, env)
    ).map(|s| s.map(|new_g_exprs| {
        g_exprs.extend(new_g_exprs);
        Expression::Function(
            format!("___diff_num_{}", function_name),
            g_exprs
        )
    }))
}

/// Computes the analytic partial derivative of `||*components||_{normtype_opt}` w.r.t. `wrt`,
/// where `components` is to be interpreted as vector.
fn apd_of_norm_for_vector(
    wrt: &String,
    normtype_opt: &Option<Box<Expression>>,
    components: &Vec<Expression>,
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Status<Expression>, String> {
    // As discussed in the case `Expression::Function`, we need to return `Df(g(x))[Dg(x)[1]]`
    // where `f(v) = ||v||_{opt}` and `g(x) = [g_exprs[0](x), ...]`.
    let Status{value: normtype, mut warnings} = VectorNorm::from_expr(normtype_opt, extra_vars, env)?;
    match normtype {
        VectorNorm::P(f64::INFINITY) => {
            // Derivative: undefined if there exist i != j s.t. |g_exprs[i](x)| = |g_exprs[j](x)|.
            // Otherwise, equals sign(g_exprs[m](x)) * diff_g[m](x) with m := argmax_k |x_k|.
            unimplemented!() // TODO when any() or something similar is available
        }
        VectorNorm::P(p) => {
            // In this case, \partial_j ||y||_p = (|y_j| / ||y||_p)^{p-1} sign(y_j).
            // Hence, D(f(g(x)))[Dg(x)[1]] = (\partial_j ||y||_p |_{g(x)})_j * (g'_j(x))_j
            //                             = ((|g_j(x)| / ||g(x)||_p)^{p-1} sign(g_j(x)))_j * (g'_j(x))_j
            //                               (vector multiplication)
            Ok(Status {
                value: expr_binop!(
                    Expression::Vector(components.iter().map(|g_j|
                        expr_binop!(
                            expr_1arg_func!("sign", g_j.clone()),
                            Mul,
                            expr_binop!(
                                expr_binop!(
                                    expr_unary_op!(Abs, g_j.clone()),
                                    Div,
                                    expr_unary_op!(Norm(normtype_opt.clone()), Expression::Vector(components.clone()))
                                ),
                                Pow(true),
                                expr_binop!(Expression::Number(p), Sub, Expression::Number(1.0))
                            )
                        )
                    ).collect()),
                    Mul,
                    Expression::Vector(Status::from_iter(
                        components.iter(),
                        |g_j| analytic_partial_derivative(g_j, wrt, extra_vars, env)
                    )?.unpack_into(&mut warnings))
                ),
                warnings
            })
        }
    }
}

/// Computes the analytic partial derivative of `||*components||_{normtype_opt}` w.r.t. `wrt`,
/// where `components` forms a matrix of size `m`x`n`.
fn apd_of_norm_for_matrix(
    wrt: &String,
    normtype_opt: &Option<Box<Expression>>,
    components: &Vec<Expression>,
    m: usize,
    n: usize,
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Status<Expression>, String> {
    // TODO when the above is implemented
    unimplemented!()
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
) -> ExtResult {
    match expr {
        Expression::None => Err("Cannot differentiate expression `None`.".to_string()),
        Expression::Identifier(ident) => Ok(Status::ok(
            if let Some(i) = vars.iter().position(|n| n == ident) { direction[i].clone() } else { Object::Real(0.0) }
        )),
        Expression::Number(_) => Ok(Status::ok(Object::Real(0.0))),
        Expression::Tuple(entries) => Status::from_iter(
            entries.iter(),
            |x| analytic_directional_derivative(vars, x, point, direction, extra_vars, env)
        ).map(|s| s.map(Object::Tuple)),
        Expression::Vector(entries) => Status::from_iter(
            entries.iter(),
            |x| analytic_directional_derivative(vars, x, point, direction, extra_vars, env).and_then(|s| s.try_map(|o| o.expect_float()))
        ).map(|s| s.map(
            |values| Object::Vector(Vector{values})
        )),
        Expression::Matrix(m, n, entries) => Status::from_iter(
            entries.iter(),
            |x| analytic_directional_derivative(vars, x, point, direction, extra_vars, env).and_then(|s| s.try_map(|o| o.expect_float()))
        ).map(|s| s.map(
            |values| Object::Matrix(Matrix::from(*m, *n, values))
        )),
        Expression::UnaryOperation(UnaryOperation::Neg, rhs)
            => analytic_directional_derivative(vars, rhs, point, direction, extra_vars, env)?.neg(),
        Expression::UnaryOperation(UnaryOperation::Not, _) => Err("Cannot differentiate the operation `Not`.".to_string()),
        Expression::UnaryOperation(UnaryOperation::Factorial, f_expr) => {
            // Using D(Γ \circ f)(p)[d] = DΓ(f(p))[Df(p)[d]] = Γ'(f(p)) * Df(p)[d] = \int_0^\infty e^{-t} t^{f(p)-1} ln(t) dt * Df(p)[d]
            // and interpreting x! as Γ(x+1):
            let wrt = f_expr.get_new_free_identifier("t");
            Status::combine(
                analytic_directional_derivative(vars, f_expr, point, direction, extra_vars, env)?,
                integration::integrate(
                    &expr_binop!(
                        expr_1arg_func!("exp", expr_unary_op!(Neg, Expression::Identifier(wrt.clone()))),
                        Mul,
                        expr_1arg_func!("ln", Expression::Identifier(wrt.clone())),
                        expr_binop!(Expression::Identifier(wrt.clone()), Pow(true), *f_expr.clone()) // Notice the power is f(p) and not f(p)-1
                    ),
                    0.0,
                    f64::INFINITY,
                    &wrt,
                    extra_vars,
                    env
                )?,
                |diff_f, integral| {
                    Ok(Object::Real(integral.expect_float()? * diff_f.expect_float()?))
                }
            )
        }
        Expression::UnaryOperation(UnaryOperation::Abs, rhs) => Status::combine(
            analytic_directional_derivative(vars, rhs, point, direction, extra_vars, env)?, // Evaluate this with old vars
            eval(rhs, &extra_vars.with_multiple(vars.iter(), point.iter()), env)?, // But this with `vars := point`
            |diff_r, rhs_eval| match rhs_eval {
                Object::Real(x) => if x > 0.0 {
                    Ok(diff_r)
                } else if x < 0.0 {
                    -&diff_r
                } else {
                    Ok(Object::Undefined)
                },
                other => Err(format!("Couldn't evaluate {} to float (obtained {}).", &**rhs, other))
            }
        ),
        Expression::UnaryOperation(UnaryOperation::Norm(_), _) => {
            // TODO when differentiation of norm is available as partial derivative
            unimplemented!()
        }
        Expression::BinaryOperation(lhs, op, rhs) => {
            let Status{value: diff_l, mut warnings} = analytic_directional_derivative(vars, lhs, point, direction, extra_vars, env)?;
            let diff_r = analytic_directional_derivative(vars, rhs, point, direction, extra_vars, env)?.unpack_into(&mut warnings);
            match op {
                BinaryOperation::Add | BinaryOperation::Sub => try_operation(&diff_l, &diff_r, op),
                BinaryOperation::Quo | BinaryOperation::Rem | BinaryOperation::And | BinaryOperation::Or => Err(format!("Cannot differentiate the operation `{op}`.")),
                BinaryOperation::Mul => {
                    let varstack = extra_vars.with_multiple(vars.iter(), point.iter());
                    try_operation(
                        &try_operation(&diff_l, &eval(rhs, &varstack, env)?.unpack_into(&mut warnings), &BinaryOperation::Mul)?, // f'(x) * g(x)
                        &try_operation(&eval(lhs, &varstack, env)?.unpack_into(&mut warnings), &diff_r, &BinaryOperation::Mul)?,  // f(x) * g'(x)
                        &BinaryOperation::Add
                    )
                },
                BinaryOperation::Div => {
                    let varstack = extra_vars.with_multiple(vars.iter(), point.iter());
                    let eval_lhs = eval(lhs, &varstack, env)?.unpack_into(&mut warnings);
                    let eval_rhs = eval(rhs, &varstack, env)?.unpack_into(&mut warnings);
                    try_operation( // d/dx (f(x) / g(x)) = (f'(x)g(x) - f(x)g'(x)) / g(x)²
                        &try_operation(
                            &try_operation(&diff_l, &eval_rhs, &BinaryOperation::Mul)?,
                            &try_operation(&eval_lhs, &diff_r, &BinaryOperation::Mul)?,
                            &BinaryOperation::Sub
                        )?,
                        &try_operation(&eval_rhs, &Object::Real(2.0), &BinaryOperation::Pow(true))?,
                        &BinaryOperation::Div
                    )
                }
                BinaryOperation::Pow(_) => {
                    let varstack = extra_vars.with_multiple(vars.iter(), point.iter());
                    let eval_lhs = eval(lhs, &varstack, env)?.unpack_into(&mut warnings);
                    let eval_rhs = eval(rhs, &varstack, env)?.unpack_into(&mut warnings);
                    try_operation( // d/dx (f(x) ^ g(x)) = f(x)^(g(x)-1) * (f'(x)g(x) + f(x)g'(x)ln(f(x)))
                        &try_operation(
                            &eval_lhs,
                            &try_operation(&eval_rhs, &Object::Real(1.0), &BinaryOperation::Sub)?,
                            &BinaryOperation::Pow(true)
                        )?,
                        &try_operation(
                            &try_operation(&diff_l, &eval_rhs, &BinaryOperation::Mul)?,
                            // The following argument `rhs` should be f(x)g'(x)ln(f(x)). However, if g'(x) = 0, then
                            // f(x) may be negative, so we then want to avoid calling f(x).ln().
                            &match (try_operation(&eval_lhs, &diff_r, &BinaryOperation::Mul)?, eval_lhs) {
                                (Object::Real(x), _) if approx_eq(x, 0.0) => Ok(Object::Real(0.0)),
                                (l, Object::Real(x)) => try_operation(&l, &Object::Real(x.ln()), &BinaryOperation::Mul),
                                _ => {return Err(format!("Evaluation of {:?} is not of type `float`.", lhs));},
                            }?,
                            &BinaryOperation::Add
                        )?,
                        &BinaryOperation::Mul
                    )
                }
                BinaryOperation::Comp(..) => Err(format!("Cannot differentiate comparison {:?}", expr)),
            }
            .map(|value| Status{value, warnings})
        }
        Expression::FoldedOperation(FoldedOperation::Sum, index_var, from, conditions, to, inner) => {
            // As in `analytic_partial_derivative`,
            // `D sum_{i=a}^b ...(p)[d]` is interpreted as `sum_{i=a(p)}^{b(p)} D ... (p)[d]`.
            let varstack = extra_vars.with_multiple(vars.iter(), point.iter());
            let Status{value: from_eval, mut warnings} = eval(from, &varstack, env)?;
            let to_eval = eval(to, &varstack, env)?.unpack_into(&mut warnings);
            match (from.contains_identifier(index_var), to.contains_identifier(index_var)) {
                (true, true) => warnings.push(format!("Assuming that both `{}` and `{}` are continuous in {} to differentiate sum.", from, to, index_var)),
                (true, false) => warnings.push(format!("Assuming that `{}` is continuous in {} to differentiate sum.", from, index_var)),
                (false, true) => warnings.push(format!("Assuming that `{}` is continuous in {} to differentiate sum.", to, index_var)),
                (false, false) => {}
            };
            compute_folded_operation(
                &FoldedOperation::Sum,
                index_var,
                |_, _| Ok(Status::ok(from_eval)),
                conditions.iter().map(|condition: &Expression| {
                    |_varstack: &VarStack<'_, '_>, _env: &mut Env| eval(condition, _varstack, _env)
                }).collect(),
                |_, _| Ok(Status::ok(Cow::Borrowed(&to_eval))),
                |_varstack, _env| analytic_directional_derivative(
                    vars,
                    inner,
                    point,
                    direction,
                    extra_vars, // Use old varstack here
                    _env
                ),
                // The type of Df(p)[d] is the same as the type of f(p)
                |_some_index_var_value, _varstack, _env| inner.get_type(
                    &_varstack.with(index_var, Cow::Borrowed(_some_index_var_value)),
                    _env
                )
                .map(|t| FoldedOperation::Sum.if_empty(&t))
                .map(|o| Status::ok(o)),
                extra_vars,
                env
            )
        }
        Expression::FoldedOperation(FoldedOperation::Product, varname, from, conditions, to, inner) => {
            // As for the analytic partial derivative since the directional derivative follows the standard product rule too.
            let varstack = extra_vars.with_multiple(vars.iter(), point.iter());
            compute_product_derivative_helper(
                varname,
                eval(from, &varstack, env)?,
                eval(to, &varstack, env)?,
                conditions.iter().map(|condition: &Expression| {
                    |_varstack: &VarStack<'_, '_>, _env: &mut Env| eval(
                        condition,
                        &_varstack.with_multiple(vars.iter(), point.iter()),
                        _env
                    )
                }).collect(),
                |_varstack, _env| eval(
                    inner,
                    &_varstack.with_multiple(vars.iter(), point.iter()),
                    _env
                ),
                |_varstack, _env| analytic_directional_derivative(vars, inner, point, direction, _varstack, _env),
                extra_vars,
                env
            )
        }
        Expression::Function(function_name, arg_expressions) => {
            // For simplicity, I'll subsequently write `f` instead of `function_name`.
            // Define `g` such that `f(arg_expressions) = f(g(wrt))`. We aim to use the chain rule:
            //     D(f \circ g)(p)[d] = Df(g(p))[Dg(p)[d]]
            // First, compute Dg(p)[d], which may be a vector, so simply differentiate componentwise.
            let Status{value: differentiated_components_of_g, mut warnings} = Status::from_iter(
                arg_expressions.iter(),
                |g_i| analytic_directional_derivative(vars, g_i, point, direction, extra_vars, env)
            )?;
            // Then, compute g(p).
            let varstack = extra_vars.with_multiple(vars.iter(), point.iter());
            let g_of_point = Status::from_iter(
                arg_expressions.iter(),
                |g_i| eval(g_i, &varstack, env)
            )?.unpack_into(&mut warnings);
            // Finally, apply the chain rule. If `f` has a representation via expression, we can get Df by a recursive call of this function.
            // In case of a direct representation, we have to fall back on a numerical directional derivative.
            let mut reinsert_later = env.functions.remove(function_name).ok_or(format!("No such function: {}", function_name))?;
            let res = match reinsert_later {
                FunctionRepr::ByExpression(ref argnames, ref function_expr) => analytic_directional_derivative(
                    argnames, function_expr, &g_of_point, &differentiated_components_of_g, &varstack, env
                ),
                FunctionRepr::Direct(ref mut f, _) => numerical_directional_derivative(
                    f, g_of_point, differentiated_components_of_g, extra_vars, env
                )
            }?.unpack_into(&mut warnings);
            env.functions.insert(function_name.clone(), reinsert_later);
            Ok(Status{value: res, warnings})
        }
        // You can't differentiate expressions like `y := ...`, that makes no sense. If the user wants `y := d/dx ...`, he should have typed that. 
        Expression::Assignment(..) => Err("Assignment cannot be differentiated.".to_string()),
        Expression::PartialDerivative(wrt_other, inner) => {
            // Idea is simple: d/dx (d/dy f(x, y)) -> First evaluate the inner derivative, then differentiate the result.
            analytic_partial_derivative(inner, wrt_other, extra_vars, env)
            .and_then(|s| s.try_map_flatten(|diff_inner| {
                analytic_directional_derivative(vars, &diff_inner, point, direction, extra_vars, env)
            }))
        }
        // The directional derivative is an object, so whatever it actually is, its derivative is zero.
        Expression::DirectionalDerivative(..) => Ok(Status::ok(Object::Real(0.0))),
        Expression::Integral(inner, a_expr, b_expr, int_var) => {
            // Proceed as in `analytic_partial_derivative`. Notice that for a, b: \R^n \to \R, we still have
            // D_v \int_{a(x)}^{b(x)} h(y) dy = h(b(x)) D_v b(x) - h(a(x)) D_v a(x) for every x, v \in \R^n.
            if !vars.iter().any(|var| inner.contains_identifier(var)) {
                let Status{value: dva, mut warnings} = analytic_directional_derivative(vars, a_expr, point, direction, extra_vars, env)?;
                let dvb = analytic_directional_derivative(vars, b_expr, point, direction, extra_vars, env)?.unpack_into(&mut warnings);
                let varstack = extra_vars.with_multiple(vars.iter(), point.iter());
                let bx = eval(
                    b_expr,
                    &varstack,
                    env
                )?.unpack_into(&mut warnings);
                let ax = eval(
                    a_expr,
                    &varstack,
                    env
                )?.unpack_into(&mut warnings);
                let hbx = eval(
                    inner,
                    &extra_vars.with(int_var, Cow::Owned(bx)),
                    env
                )?.unpack_into(&mut warnings);
                let hax = eval(
                    inner,
                    &extra_vars.with(int_var, Cow::Owned(ax)),
                    env
                )?.unpack_into(&mut warnings);
                try_operation(
                    &try_operation(&hbx, &dvb, &BinaryOperation::Mul)?,
                    &try_operation(&hax, &dva, &BinaryOperation::Mul)?,
                    &BinaryOperation::Sub
                ).map(|value| Status{value, warnings})
            } else {
                numerical_directional_derivative(&mut (|parsed_args: &[Object], _: &[Expression], context: Option<(&VarStack, &mut Env)>| {
                    let (_varstack, _env) = context.ok_or("[Unreachable] Function needs varstack and environment.".to_string())?;
                    eval(
                        expr,
                        &_varstack.with_multiple(vars.iter(), parsed_args.iter()),
                        _env
                    )}
                ), point.to_vec(), direction.to_vec(), extra_vars, env)
            }
        }
        Expression::IfElse(condition, iftrue, iffalse) => {
            // D (if c(x) {a(x)} else {b(x)})(x)[d] = if c(x) {Da(x)[d]} else {Db(x)[d]}
            let varstack = extra_vars.with_multiple(vars.iter(), point.iter());
            let Status{value: condition_met, mut warnings} = eval(condition, &varstack, env)?.try_map(|o| o.expect_bool())?;
            Ok(Status {
                value: if condition_met {
                    analytic_directional_derivative(vars, iftrue, point, direction, &varstack, env)?.unpack_into(&mut warnings)
                } else {
                    analytic_directional_derivative(vars, iffalse, point, direction, &varstack, env)?.unpack_into(&mut warnings)
                },
                warnings
            })
        }
    }
}

/// For `f: R -> R^{mxn}`, we could use the "three-point central difference formula" (proof by Taylor expansion):
///     `f'(x) = \frac{f(x+h) - f(x-h)}{2h} + O(h²)`
/// for `h` close to zero (here, `h = 1e-9`).
/// 
/// For general `f`, we generalize this method.
/// 
/// Note: this can also be used for functions from `\R` to `\R` by using `direction = vec![Object::Real(1.0)]`.
/// 
/// Unfortunately, `point` has to be owned (or we'd have to clone it) since we want to modify it and the original passed vector need not to be mutable.
/// Moreover, also owning `direction` allows to decrease the number of required operations.
/// 
/// Note: to see why we even need to pass around varstacks and envs, see `evaluator::eval`.
pub fn numerical_directional_derivative<F: FnMut(&[Object], &[Expression], Option<(&VarStack, &mut Env)>) -> ExtResult>(
    f: &mut F,
    mut point: Vec<Object>,
    mut direction: Vec<Object>,
    extra_vars: &VarStack,
    env: &mut Env
) -> ExtResult {
    if point.len() != direction.len() {
        return Err("`point` and `direction` for derivative must be vectors of the same length (possibly 1).".to_string());
    }
    // We use h = 1e-6 * (1 + |point|)
    let norm_of_point = point.iter().map(|x| match x {
        Object::Undefined | Object::Success | Object::LiteralExpression(_) | Object::Tuple(_) => Err(format!("Point can't contain object of type {:?}.", x)),
        Object::Real(x) => Ok(x.abs()),
        Object::Complex(x) => Ok(x.modulus()),
        Object::Vector(x) => Ok(x.norm(&VectorNorm::P(2.0))),
        Object::Matrix(x) => x.norm(&MatrixNorm::Frobenius)
    }).collect::<Result<Vec<_>, _>>()?;
    let h = 1e-6 * (1.0 + min(norm_of_point.into_iter()).unwrap_or(0.0));
    for (i, coord) in point.iter_mut().enumerate() {
        direction[i] = h * &direction[i]; // Spares us another operation later
        *coord = try_operation(coord, &direction[i], &BinaryOperation::Add)?; // point + h*direction
    }
    let Status{value: left_res, mut warnings} = f(&point, &[], Some((extra_vars, env)))?;
    for (i, coord) in point.iter_mut().enumerate() {
        // If the previous loop worked, this one will too.
        *coord = try_operation(coord, &(2.0 * &direction[i]), &BinaryOperation::Sub).unwrap();
    }
    let right_res = f(&point, &[], Some((extra_vars, env)))?.unpack_into(&mut warnings);
    match (left_res, right_res) {
        (Object::Real(lhs), Object::Real(rhs)) => Ok(Object::Real((lhs - rhs) / (2.0 * h))),
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
    .map(|value| Status{value, warnings})
}