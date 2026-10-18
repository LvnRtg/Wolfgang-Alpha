use std::borrow::Cow;
use std::collections::HashMap;

use crate::{defaults, dispatch_matrix, dispatch_vector, expr_binop, ok};
use crate::lang::eval;
use crate::math::{Env, Expression, FunctionRepr, integration, Object, ObjType, VarStack};
use crate::math::expressions::Repeat;
use crate::math::expressions::simplification::*;
use crate::math::expressions::type_checking::MakeTypeTopLevelError;
use crate::math::matrices::MatrixNorm;
use crate::math::operations::{BinaryOperation, Comparison, FoldedOperation, UnaryOperation};
use crate::math::operations::folded_operations::{compute_folded_operation, compute_product_derivative_helper};
use crate::math::traits::*;
use crate::math::utils;
use crate::math::vectors::VectorNorm;
use crate::status::{ExtResult, Status};


/// Returns `Some(n)` if all items `(s, m)` in `diff_wrt` satisfy either `s == wrt` or `m == 0`; then, `n` is the sum over all such `m`.
/// 
/// Special case: when `n == 0`, this function also returns `Ok(None)`.
/// 
/// Short-circuits if a `Repeat` couldn't be evaluated.
pub fn all_partial_derivatives_wrt_same_var(diff_wrt: &[(String, Repeat)], wrt: &String, varstack: &VarStack, env: &mut Env) -> Result<Option<Status<usize>>, String> {
    let mut warnings = Vec::new();
    let mut n = 0;
    for (s, r) in diff_wrt {
        let m = r.get(varstack, env)?.unpack_into(&mut warnings);
        if s != wrt && m != 0 {
            return Ok(None);
        } else {
            n += m;
        }
    }
    if n == 0 {
        Ok(None)
    } else {
        Ok(Some(Status{value: n, warnings}))
    }
}

/// Applies all given partial derivatives to `expr` in order.
/// 
/// Clones `expr` if `seq` is empty.
pub fn apply_seq_of_partial_derivatives(expr: &Expression, seq: &[(String, Repeat)], varstack: &VarStack, env: &mut Env) -> Result<Status<Expression>, String> {
    let mut it = seq.iter();
    if let Some((s, r)) = it.next() {
        let mut warnings = Vec::new();
        it.fold(
            apply_n_partial_derivatives(expr, s, r, varstack, env).map(|s| s.unpack_into(&mut warnings)),
            |acc, (s, r)| acc.and_then(
                |_acc|
                apply_n_partial_derivatives(&_acc, s, r, varstack, env)
                .map(|s| s.unpack_into(&mut warnings))
            )
        )
        .map(|value| Status{value, warnings})
    } else {
        ok!(expr.clone())
    }
}

/// Differentiates `expr` `n` times w.r.t. `wrt`.
/// 
/// Clones `expr` if `n == 0`.
pub fn apply_n_partial_derivatives(expr: &Expression, wrt: &String, n: &Repeat, varstack: &VarStack, env: &mut Env) -> Result<Status<Expression>, String> {
    let Status{value: n, mut warnings} = n.get(varstack, env)?;
    Ok(Status{
        value: if n == 0 {
            expr.clone()
        } else {
            (0..n-1).fold(
                analytic_partial_derivative(expr, wrt, varstack, env).map(|s| s.unpack_into(&mut warnings)),
                |acc, _| acc.and_then(
                    |_acc| analytic_partial_derivative(&_acc, wrt, varstack, env).map(|s| s.unpack_into(&mut warnings))
                )
            )?
        },
        warnings
    })
}


/// Differentiates the given expression w.r.t. the variable `wrt` analytically, that is, by parsing the expression recursively and
/// applying known differentiation rules (e.g. product rule, chain rule).
/// 
/// If an function `f` with representation `FunctionRepr::Direct` is encountered for which the derivative is not provided (as it is for default identifiers),
/// we cannot differentiate it analytically. Then, we use the helper function `___diff_num`; the function `eval` then processes it as the function
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
        Expression::None => ok!(Expression::None),
        Expression::Identifier(ident) => ok!(Expression::Number(if ident == wrt { 1.0 } else { 0.0 })),
        Expression::Number(_) => ok!(Expression::Number(0.0)),
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
                |e| simplify_neg(Decomposed::from(e)).to_expr()
            )
        ),
        Expression::UnaryOperation(UnaryOperation::Not, _) => Err("Cannot differentiate the operation `Not`.".to_string()),
        Expression::UnaryOperation(UnaryOperation::Factorial, inner) => {
            // We treat the factorial as gamma function here since differentiating a function with discrete domain makes no sense.
            // We use: d/dx x! = Γ'(x+1) = \int_0^\infty e^{-t} t^x ln(t) dt
            //      => d/dx f(x)! = f'(x) * \int_0^\infty e^{-t} t^{f(x)} ln(t) dt
            let wrt = inner.get_new_free_identifier("t");
            ok!(Expression::Integral(
                Box::new(
                    simplify_mul(
                        simplify_exp(simplify_neg(Decomposed::from(Expression::Identifier(wrt.clone())))),
                        simplify_mul(
                            simplify_ln(Decomposed::from(Expression::Identifier(wrt.clone()))),
                            simplify_pow(
                                Decomposed::from(Expression::Identifier(wrt.clone())),
                                simplify(*inner.clone())
                            )
                        )
                    )
                    .to_expr()
                ),
                Box::new(Expression::Number(0.0)),
                Box::new(Expression::Identifier("inf".to_string())),
                wrt
            ))
        }
        Expression::UnaryOperation(UnaryOperation::Abs, rhs) => analytic_partial_derivative(rhs, wrt, extra_vars, env)
        .map(|s| s.map(
            |diff_r| {
                let rhs = simplify(*rhs.clone());
                // Notice that `diff_r` is already simplified, so it suffices to wrap it in
                // `Decomposed` instead of simplifying the whole thing again.
                let diff_r = Decomposed::from(diff_r);
                simplify_if_else(
                    simplify_gt_zero(rhs.clone()),
                    diff_r.clone(),
                    simplify_if_else(
                        simplify_lt_zero(rhs),
                        simplify_neg(diff_r),
                        Decomposed::from(Expression::None)
                    )
                )
                .to_expr()
            }
        )),
        Expression::UnaryOperation(UnaryOperation::Norm(opt), rhs) => {
            match rhs.make_type_top_level(
                false,
                &extra_vars.with(wrt, Cow::Owned(Object::Real(1.0))),
                env
            ) {
                Ok((rhs_toplevel, rhs_type)) => match rhs_toplevel {
                    Expression::Vector(components) => apd_of_norm_for_vector(wrt, opt, &components, extra_vars, env),
                    Expression::Matrix(m, n, components) => apd_of_norm_for_matrix(wrt, opt, &components, m, n, extra_vars, env),
                    _ if matches!(rhs_type, ObjType::Tuple | ObjType::NonObject) => Err(format!("Operation 'Norm' invalid for operand {:?}.", rhs_toplevel)),
                    other if rhs_type == ObjType::LiteralExpression => ok!(Expression::UnaryOperation(UnaryOperation::Norm(opt.clone()), Box::new(other))),
                    other => { // Scalar type
                        // In this case, the norm should simply be an absolute value, regardless of `opt`.
                        analytic_partial_derivative(&other, wrt, extra_vars, env)
                        .map(|s| s.map(
                            |diff_r| {
                                let other = simplify(other);
                                let diff_r = Decomposed::from(diff_r);
                                simplify_if_else(
                                    simplify_gt_zero(other.clone()),
                                    diff_r.clone(),
                                    simplify_if_else(
                                        simplify_lt_zero(other),
                                        simplify_neg(diff_r),
                                        Decomposed::from(Expression::None)
                                    )
                                )
                                .to_expr()
                            }
                        ))
                    }
                },
                Err(MakeTypeTopLevelError::GoToNumeric) => {
                    Ok(Status {
                        value: Expression::Function(
                            "___diff_num".to_string(),
                            vec![
                                Expression::UnaryOperation(UnaryOperation::Norm(opt.clone()), rhs.clone()),
                                Expression::Identifier(wrt.clone()),
                                Expression::Identifier(wrt.clone()),
                                Expression::Number(1.0)
                            ]
                        ),
                        warnings: vec![format!(
                            "Couldn't reveal the components of the expression `{}`; defaulting to numeric derivative.",
                            &**rhs
                        )]
                    })
                }
                Err(MakeTypeTopLevelError::Err(e)) => Err(e)
            }
        }
        Expression::BinaryOperation(lhs, op, rhs) => Status::combine_flatten(
            analytic_partial_derivative(lhs, wrt, extra_vars, env)?,
            analytic_partial_derivative(rhs, wrt, extra_vars, env)?,
            |diff_l, diff_r| {
                macro_rules! simplify_if {
                    ($x:expr, $cond:expr) => {
                        if $cond {
                            simplify(*$x.clone())
                        } else {
                            Decomposed::from_indecomposable(*$x.clone())
                        }
                    }
                }
                let l_is_scalar = matches!(lhs.get_type(&extra_vars.with(wrt, Cow::Owned(Object::Real(42.0))), env), Ok(ObjType::Scalar));
                let r_is_scalar = matches!(rhs.get_type(&extra_vars.with(wrt, Cow::Owned(Object::Real(42.0))), env), Ok(ObjType::Scalar));
                let merge_inbetween = !l_is_scalar || !r_is_scalar;
                let diff_l = if l_is_scalar {
                    Decomposed::from(diff_l)
                } else {
                    Decomposed::from_indecomposable(diff_l)
                };
                let diff_r = if r_is_scalar {
                    Decomposed::from(diff_r)
                } else {
                    Decomposed::from_indecomposable(diff_r)
                };
                match op {
                    BinaryOperation::Add => ok!(simplify_add(diff_l, diff_r).to_expr()),
                    BinaryOperation::Sub => ok!(simplify_sub(diff_l, diff_r).to_expr()),
                    BinaryOperation::Quo | BinaryOperation::Rem | BinaryOperation::And | BinaryOperation::Or
                        => Err(format!("Cannot differentiate the operation `{op}`.")),
                    BinaryOperation::Mul => ok!(
                        simplify_add(
                            simplify_mul(diff_l, simplify_if!(rhs, r_is_scalar)) // f'(x) * g(x)
                            .to_indecomposable_if(merge_inbetween),
                            simplify_mul(simplify_if!(lhs, l_is_scalar), diff_r)  // f(x) * g'(x)
                            .to_indecomposable_if(merge_inbetween),
                        )
                        .to_expr()
                    ),
                    BinaryOperation::Div => {
                        // d/dx (f(x) / g(x)) = (f'(x)g(x) - f(x)g'(x)) / g(x)²
                        let rhs = simplify_if!(rhs, r_is_scalar);
                        ok!(
                            simplify_div(
                                simplify_sub(
                                    simplify_mul(diff_l, rhs.clone()).to_indecomposable_if(merge_inbetween),
                                    simplify_mul(simplify_if!(lhs, l_is_scalar), diff_r).to_indecomposable_if(merge_inbetween)
                                ).to_indecomposable_if(merge_inbetween),
                                simplify_pow(rhs, Decomposed::from_f64(2.0))
                            )
                            .to_expr()
                        )
                    }
                    BinaryOperation::Pow(_) => {
                        if let Ok(Status{value: i, warnings}) = eval(rhs, extra_vars, env).and_then(|s| s.try_map(|o| o.expect_int::<f64>())) {
                            // Exponent is a constant integer, so d/dx (f(x) ^ i) = i * f'(x) * f(x)^(i-1).
                            Ok(Status {
                                value: simplify_mul(
                                    Decomposed::from_f64(i),
                                    simplify_mul(
                                        diff_l,
                                        simplify_pow(
                                            simplify_if!(lhs, l_is_scalar),
                                            Decomposed::from_f64(i - 1.0)
                                        )
                                    )
                                )
                                .to_expr(),
                                warnings
                            })
                        } else {
                            // d/dx (f(x) ^ g(x)) = f(x)^(g(x)-1) * (f'(x)g(x) + f(x)g'(x)ln(f(x)))
                            // (provided f(x) > 0)
                            let (lhs, rhs) = (simplify_if!(lhs, l_is_scalar), simplify_if!(rhs, r_is_scalar));
                            ok!(
                                simplify_if_else(
                                    simplify_gt_zero(lhs.clone()),
                                    simplify_mul(
                                        simplify_pow(
                                            lhs.clone(),
                                            simplify_sub(
                                                rhs.clone(),
                                                Decomposed::from_f64(1.0)
                                            )
                                        ).to_indecomposable_if(merge_inbetween),
                                        simplify_add(
                                            simplify_mul(diff_l, rhs).to_indecomposable_if(merge_inbetween),
                                            simplify_mul(
                                                simplify_mul(lhs.clone(), diff_r).to_indecomposable_if(merge_inbetween),
                                                simplify_ln(lhs).to_indecomposable_if(merge_inbetween)
                                            ).to_indecomposable_if(merge_inbetween)
                                        ).to_indecomposable_if(merge_inbetween)
                                    ),
                                    Decomposed::Indecomposable(Expression::None)
                                )
                                .to_expr()
                            )
                        }
                    }
                    BinaryOperation::Comp(..) => Err(format!("Cannot differentiate comparison {:?}", expr)),
                }
            }
        ),
        Expression::FoldedOperation(FoldedOperation::Sum, varname, from, conditions, to, inner) => {
            // As for the product (but slightly simpler since `d/dx sum_{i=a}^b f(i, x) = sum_{i=a}^b d/dx f(i, x)`),
            // see case `Expression::FoldedOperation` below.
            let Status{value: inner_diff, mut warnings} = analytic_partial_derivative(inner, wrt, extra_vars, env)?;
            match (from.contains_identifier(wrt), to.contains_identifier(wrt)) { // Typically, both expressions must be checked anyway
                (true, true) => warnings.push(format!("Assuming that both `{}` and `{}` are continuous in {} to differentiate sum.", from, to, wrt)),
                (true, false) => warnings.push(format!("Assuming that `{}` is continuous in {} to differentiate sum.", from, wrt)),
                (false, true) => warnings.push(format!("Assuming that `{}` is continuous in {} to differentiate sum.", to, wrt)),
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
            // Note: in the following strucutre, we have to change the inner index of the inner folded operation
            // from `index_var` to something else; otherwise, the condition `i != j` will not be valid.
            let new_index_var = Expression::get_new_free_identifier_in_none_of(
                {
                    let index_var_prefix = index_var.chars().take_while(|c| !c.is_ascii_digit()).collect::<String>(); // Always non-empty
                    if index_var_prefix.len() == 1 {
                        let c = index_var_prefix.chars().next().unwrap() as u32;
                        if c >= 65 && c < 90 || c >= 97 && c < 122 { // Exclude 'Z', 'z'
                            char::from_u32(c+1).unwrap().to_string()
                        } else {
                            index_var_prefix.clone()
                        }
                    } else {
                        index_var_prefix.clone()
                    }
                },
                std::iter::chain(
                    std::iter::chain(
                        std::iter::once(&**to),
                        std::iter::once(&**inner)
                    ),
                    conditions.iter()
                )
            );
            Ok(Status {
                value: Expression::FoldedOperation(
                    FoldedOperation::Sum,
                    index_var.clone(),
                    from.clone(),
                    conditions.clone(),
                    to.clone(),
                    Box::new(
                        // Checking the type is useful here because if `inner` is anything but a scalar,
                        // we do not necessarily have commutativity of multiplication and will have to
                        // write out `prod_{j<i} inner(j) * diff_inner(i) * prod_{j>i} inner(j)`.
                        // If `inner` is a scalar, this reduces to `diff_inner(i) * prod_{j!=i} inner(j)`.
                        if let Ok(ObjType::Scalar) = inner.get_type(&extra_vars.with(wrt, Cow::Owned(Object::Real(42.0))), env) {
                            simplify_mul(
                                Decomposed::from(inner_diff),
                                simplify_product(
                                    new_index_var.clone(),
                                    simplify(*from.clone()),
                                    {
                                        let mut cond = conditions.iter().map(|c| simplify(
                                            c.replace_identifiers(&HashMap::from([(index_var, &Expression::Identifier(new_index_var.clone()))]))
                                        )).collect::<Vec<_>>();
                                        cond.push(Decomposed::from(expr_binop!(
                                            Expression::Identifier(new_index_var.clone()),
                                            Comp(Comparison::Neq, None),
                                            Expression::Identifier(index_var.clone())
                                        )));
                                        cond
                                    },
                                    simplify(to.replace_identifiers(&HashMap::from([(index_var, &Expression::Identifier(new_index_var.clone()))]))),
                                    simplify(inner.replace_identifiers(&HashMap::from([(index_var, &Expression::Identifier(new_index_var))])))
                                )
                            )
                            .to_expr()
                        } else {
                            expr_binop!(
                                Expression::FoldedOperation(
                                    FoldedOperation::Product,
                                    new_index_var.clone(),
                                    from.clone(),
                                    {
                                        let mut cond = conditions.iter().map(
                                            |c| c.replace_identifiers(&HashMap::from([(index_var, &Expression::Identifier(new_index_var.clone()))]))
                                        ).collect::<Vec<_>>();
                                        cond.push(expr_binop!(
                                            Expression::Identifier(new_index_var.clone()),
                                            Comp(Comparison::Lt, None),
                                            Expression::Identifier(index_var.clone())
                                        ));
                                        cond
                                    },
                                    Box::new(to.replace_identifiers(&HashMap::from([(index_var, &Expression::Identifier(new_index_var.clone()))]))),
                                    Box::new(inner.replace_identifiers(&HashMap::from([(index_var, &Expression::Identifier(new_index_var.clone()))])))
                                ),
                                Mul,
                                inner_diff,
                                Expression::FoldedOperation(
                                    FoldedOperation::Product,
                                    new_index_var.clone(),
                                    from.clone(),
                                    {
                                        let mut cond = conditions.iter().map(
                                            |c| c.replace_identifiers(&HashMap::from([(index_var, &Expression::Identifier(new_index_var.clone()))]))
                                        ).collect::<Vec<_>>();
                                        cond.push(expr_binop!(
                                            Expression::Identifier(new_index_var.clone()),
                                            Comp(Comparison::Gt, None),
                                            Expression::Identifier(index_var.clone())
                                        ));
                                        cond
                                    },
                                    Box::new(to.replace_identifiers(&HashMap::from([(index_var, &Expression::Identifier(new_index_var.clone()))]))),
                                    Box::new(inner.replace_identifiers(&HashMap::from([(index_var, &Expression::Identifier(new_index_var))])))
                                )
                            )
                        }
                    )
                ),
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
        Expression::PartialDerivative(inner_seq, inner) => {
            // Idea is simple: d/dx (d/dy f(x, y)) -> First evaluate the inner derivative, then differentiate the result.
            apply_seq_of_partial_derivatives(inner, inner_seq, extra_vars, env)
            .and_then(|s| s.try_map_flatten(
                |diff_inner| analytic_partial_derivative(&diff_inner, wrt, extra_vars, env)
            ))
        }
        // For the directional derivative of a directional derivative, we must currently use numerical differentiation because directional derivatives can only be evaluated pointwise.
        // This might be extended later.
        Expression::DirectionalDerivative(inner_vars, inner_expr, inner_point, inner_direction) => {
            ok!(Expression::Function(
                "___diff_num".to_string(),
                vec![
                    Expression::DirectionalDerivative(inner_vars.clone(), inner_expr.clone(), inner_point.clone(), inner_direction.clone()),
                    Expression::Identifier(wrt.clone()),
                    Expression::Identifier(wrt.clone()),
                    Expression::Number(1.0)
                ]
            ))
        }
        Expression::Integral(inner, a, b, int_var) => {
            // Since we can't always exchange differentiation and integration, we proceed as follows. First,
            // check if the integral is of the special form \int_{a(x)}^{b(x)} h(y) dy where x = wrt and h does not involve x.
            // Then, the derivative would be h(b(x)) b'(x) - h(a(x)) a'(x).
            // Otherwise, return ___diff_num(...).
            if !inner.contains_identifier(wrt) {
                Status::combine(
                    analytic_partial_derivative(a, wrt, extra_vars, env)?,
                    analytic_partial_derivative(b, wrt, extra_vars, env)?,
                    |da, db| Ok(
                        // We can safely assume that `da` and `db` are reals; however, `inner` could be something else.
                        if let Ok(ObjType::Scalar) = inner.get_type(&extra_vars.with(wrt, Cow::Owned(Object::Real(42.0))), env) {
                            simplify_sub(
                                simplify_mul(
                                    simplify(inner.replace_identifiers(&HashMap::from([(int_var, &(**b).clone())]))),
                                    Decomposed::from(db)
                                ),
                                simplify_mul(
                                    simplify(inner.replace_identifiers(&HashMap::from([(int_var, &(**a).clone())]))),
                                    Decomposed::from(da)
                                )
                            )
                            .to_expr()
                        } else {
                            simplify_sub(
                                // N.b.: the call below can't create problematic factors for non-scalar `inner`, so `.to_indecomposable()` after it is unnecessary.
                                simplify_mul(
                                    simplify(inner.replace_identifiers(&HashMap::from([(int_var, &(**b).clone())]))).to_indecomposable(),
                                    Decomposed::from(db)
                                ), // <--- No `.to_indecomposable()` needed here
                                simplify_mul(
                                    simplify(inner.replace_identifiers(&HashMap::from([(int_var, &(**a).clone())]))).to_indecomposable(),
                                    Decomposed::from(da)
                                )
                            )
                            .to_expr()
                        }
                    )
                )
            } else {
                ok!(
                    Expression::Function(
                        "___diff_num".to_string(),
                        vec![
                            Expression::Integral(inner.clone(), a.clone(), b.clone(), int_var.clone()),
                            Expression::Identifier(wrt.clone()),
                            Expression::Identifier(wrt.clone()),
                            Expression::Number(1.0)
                        ]
                    )
                )
            }
        }
        Expression::Limit(idx_name, direction, idx_limit, inner) => ok!(Expression::Function(
            "___diff_num".to_string(),
            vec![
                Expression::Limit(idx_name.clone(), *direction, idx_limit.clone(), inner.clone()),
                Expression::Identifier(idx_name.clone()),
                Expression::Identifier(idx_name.clone()),
                Expression::Number(1.0)
            ]
        )),
        Expression::IfElse(x, y, z) => Status::combine(
            analytic_partial_derivative(y, wrt, extra_vars, env)?,
            analytic_partial_derivative(z, wrt, extra_vars, env)?,
            |y, z| Ok(
                simplify_if_else(
                    simplify(*x.clone()),
                    Decomposed::from(y),
                    Decomposed::from(z)
                )
                .to_expr()
            )
        )
    }
}

/// Computes the analytic partial derivative of `f(*given_arg_exprs)` w.r.t. `wrt`.
fn apd_for_function(
    wrt: &String,
    function_name: &String,
    f: &FunctionRepr,
    given_arg_exprs: Vec<Expression>,
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Status<Expression>, String> {
    match f {
        FunctionRepr::ByExpression(f_argnames, f_expr) => {
            // Idea: replace `function_name(given_arg_exprs)` by `f_expr` and differentiate that instead.
            analytic_partial_derivative(
                &f_expr.replace_identifiers(&f_argnames.iter().zip(given_arg_exprs.iter()).collect()),
                wrt,
                extra_vars,
                env
            )
        }
        FunctionRepr::Direct(..) => {
            // If `function_name` refers to a default function (e.g. `exp`), we can spare ourselves the below code.
            if defaults::FUNCTIONS_WITH_PROVIDED_DERIVATIVE.contains(&function_name.as_str()) {
                // Similar to the chain rule block in `analytic_directional_derivative`, with a little change: with the same f, g as there, we have
                //     D(f \circ g)(x)[1.0] = Df(g(x))[Dg(x)[1.0]]
                Status::from_iter(
                    given_arg_exprs.iter(),
                    |g_i| analytic_partial_derivative(g_i, wrt, extra_vars, env)
                )
                .and_then(|s| s.try_map(
                    |differentiated_components_of_g|
                    defaults::get_default_derivative(function_name.as_str(), &given_arg_exprs, &differentiated_components_of_g)
                ))
            } else {
                apd_for_direct_function(wrt, function_name, given_arg_exprs, extra_vars, env)
            }
        }
    }
}

/// Computes the analytic partial derivative of `f(*given_arg_exprs)` w.r.t. `wrt` where `f` is an existing function in `env` with direct representation
/// by simply packing it into a `__diff_num` expression.
/// 
/// Note: this also works if `f` has another representation but shouldn't be used in that case since it loses precision.
fn apd_for_direct_function(
    wrt: &String,
    function_name: &String,
    given_arg_exprs: Vec<Expression>,
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Status<Expression>, String> {
    // Importantly, note that the directional derivative is a separate function. Therefore, we can assume w.l.o.g. that `f \circ g` maps from `\R` to `\R`.
    // For each component of `g` (note that `g` maps from `\R` to `\R^n`), analytically differentiate that component w.r.t. `wrt` (which is the input of `g`).
    // The returned expression should be (writing `x` for `wrt`)
    // ```
    // d/dx f(g(x)) |_x
    // = D(f \circ g)(x)[1]        (since `f \circ g` maps from `\R` to `\R`)
    // = Df(g(x))[Dg(x)[1]]        (chain rule)
    // ```
    // In the program's syntax, we achieve the latter using `___diff_num`.
    // Note: we could just be lazy and call `___diff_num` on `f(g(x))` entirely, but this would potentially lead to more imprecise results.
    let Status{value: diff_g, warnings} = Status::from_iter(
        given_arg_exprs.iter(),
        |g_i| analytic_partial_derivative(g_i, wrt, extra_vars, env)
    )?;
    // It is safe to pick the following variable names since the expression constructed from them is just `function_name(v_1, ..., v_n)`,
    // so they can't clash with any existing variables (since there are none).
    let mut args = (0..given_arg_exprs.len()).map(|i| Expression::Identifier(format!("v_{i}"))).collect::<Vec<_>>();
    // Add `f(v_1, ..., v_n)` at the beginning
    args.insert(
        0,
        Expression::Function(
            function_name.clone(),
            args.clone()
        )
    );
    // Add the point to differentiate at to `args`, that is, `g(x)`
    args.extend(given_arg_exprs.into_iter());
    // Add the direction, i.e. `Dg(x)[1]`
    args.extend(diff_g.into_iter());
    Ok(Status {
        value: Expression::Function(
            "___diff_num".to_string(),
            args
        ),
        warnings
    })
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
            // Simplifications below are fine because all `g_j` are scalars.
            // We only have to use the non-simplify-version on the outside,
            // but we still want to take care of the signs, which we thus do explicitly.
            let simpl_g_j = components.iter().map(|g_j| simplify(g_j.clone())).collect::<Vec<_>>();
            let left_vec = simplify_vec(
                simpl_g_j.iter().map(
                    |g_j| simplify_mul(
                        simplify_sign(g_j.clone()),
                        simplify_pow(
                            simplify_div(
                                simplify_abs(g_j.clone()),
                                Decomposed::from(crate::expr_unary_op!(
                                    Norm(normtype_opt.clone()),
                                    simplify_vec(simpl_g_j.clone()).to_expr()
                                ))
                            ),
                            simplify_sub(
                                Decomposed::from_f64(p),
                                Decomposed::from_f64(1.0)
                            )
                        )
                    )
                )
                .collect()
            );
            let right_vec = simplify_vec(Status::from_iter(
                components.iter(),
                |g_j| analytic_partial_derivative(g_j, wrt, extra_vars, env)
                .map(|s| s.map(Decomposed::from))
            )?.unpack_into(&mut warnings));
            Ok(Status {
                value: simplify_mul(left_vec.to_indecomposable(), right_vec.to_indecomposable()).to_expr(),
                warnings
            })
        }
    }
}

/// Computes the analytic partial derivative of `||*components||_{normtype_opt}` w.r.t. `wrt`,
/// where `components` forms a matrix of size `m`x`n`.
#[allow(unused_variables)]
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
        Expression::Identifier(ident) => ok!(
            if let Some(i) = vars.iter().position(|n| n == ident) { direction[i].clone() } else { Object::Real(0.0) }
        ),
        Expression::Number(_) => ok!(Object::Real(0.0)),
        Expression::Tuple(entries) => Status::from_iter(
            entries.iter(),
            |x| analytic_directional_derivative(vars, x, point, direction, extra_vars, env)
        ).map(|s| s.map(Object::Tuple)),
        Expression::Vector(entries) => {
            let mut warnings = Vec::new();
            Object::expect_scalar_vector(
                entries.iter()
                .map(
                    |x|
                    analytic_directional_derivative(vars, x, point, direction, extra_vars, env)
                    .map(|s| s.unpack_into(&mut warnings))
                )
            )
            .map(|value| Status{value, warnings})
        }
        Expression::Matrix(m, n, entries) => {
            let mut warnings = Vec::new();
            Object::expect_scalar_matrix(
                *m, *n,
                entries.iter()
                .map(
                    |x|
                    analytic_directional_derivative(vars, x, point, direction, extra_vars, env)
                    .map(|s| s.unpack_into(&mut warnings))
                )
            )
            .map(|value| Status{value, warnings})
        }
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
                    &simplify_mul(
                        simplify_exp(simplify_neg(Decomposed::from(Expression::Identifier(wrt.clone())))),
                        simplify_mul(
                            simplify_ln(Decomposed::from(Expression::Identifier(wrt.clone()))),
                            simplify_pow(
                                Decomposed::from(Expression::Identifier(wrt.clone())),
                                simplify(*f_expr.clone()) // Notice the power is f(p) and not f(p)-1
                            )
                        )
                    )
                    .to_expr(),
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
                    diff_r.neg()
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
                BinaryOperation::Add => diff_l.add(diff_r),
                BinaryOperation::Sub => diff_l.sub(diff_r),
                BinaryOperation::Quo | BinaryOperation::Rem | BinaryOperation::And | BinaryOperation::Or => Err(format!("Cannot differentiate the operation `{op}`.")),
                BinaryOperation::Mul => {
                    let varstack = extra_vars.with_multiple(vars.iter(), point.iter());
                    // f'(x) * g(x) + f(x) * g'(x)
                    diff_l.mul(eval(rhs, &varstack, env)?)? // f'(x) * g(x)
                    .add(eval(lhs, &varstack, env)?.mul(diff_r)?)  // f(x) * g'(x)
                },
                BinaryOperation::Div => {
                    let varstack = extra_vars.with_multiple(vars.iter(), point.iter());
                    let eval_rhs = eval(rhs, &varstack, env)?.unpack_into(&mut warnings);
                    // d/dx (f(x) / g(x)) = (f'(x)g(x) - f(x)g'(x)) / g(x)²
                    diff_l.mul(&eval_rhs)?
                    .sub(eval(lhs, &varstack, env)?.mul(diff_r)?)?
                    .div(eval_rhs.squared()?.unpack_into(&mut warnings))
                }
                BinaryOperation::Pow(_) => {
                    let varstack = extra_vars.with_multiple(vars.iter(), point.iter());
                    let eval_lhs = eval(lhs, &varstack, env)?.unpack_into(&mut warnings);
                    let eval_rhs = eval(rhs, &varstack, env)?.unpack_into(&mut warnings);
                    // d/dx (f(x) ^ g(x)) = f(x)^(g(x)-1) * (f'(x)g(x) + f(x)g'(x)ln(f(x)))
                    eval_lhs.pow_ref(((&eval_rhs).sub(Object::Real(1.0)))?)?
                    .mul(
                        diff_l
                        .mul(eval_rhs)?
                        .add(
                            match ((&eval_lhs).mul(diff_r)?.unpack_into(&mut warnings), eval_lhs) {
                                (Object::Real(x), _) if utils::approx_eq(x, 0.0) => Object::Real(0.0),
                                (l, Object::Real(x)) => (l.mul(Object::Real(x.ln())))?.unpack_into(&mut warnings),
                                _ => return Err(format!("Evaluation of {:?} is not of type `float`.", lhs)),
                            }
                        )?
                    )
                }
                BinaryOperation::Comp(..) => Err(format!("Cannot differentiate comparison {:?}", expr)),
            }
            .map(|s| Status{value: s.unpack_into(&mut warnings), warnings})
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
                |_, _| ok!(from_eval),
                conditions.iter().map(|condition: &Expression| {
                    |_varstack: &VarStack<'_, '_>, _env: &mut Env| eval(condition, _varstack, _env)
                }).collect(),
                |_, _| ok!(Cow::Borrowed(&to_eval)),
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
        Expression::PartialDerivative(inner_seq, inner) => {
            // Idea is simple: d/dx (d/dy f(x, y)) -> First evaluate the inner derivative, then differentiate the result.
            apply_seq_of_partial_derivatives(inner, inner_seq, extra_vars, env)
            .and_then(|s| s.try_map_flatten(|diff_inner| {
                analytic_directional_derivative(vars, &diff_inner, point, direction, extra_vars, env)
            }))
        }
        // For the directional derivative of a directional derivative, we must currently use numerical differentiation because directional derivatives can only be evaluated pointwise.
        // This might be extended later.
        Expression::DirectionalDerivative(inner_vars, inner_expr, inner_point, inner_direction) => {
            numerical_directional_derivative(
                // Function to numerically differentiate, i.e. the inner directional derivative.
                // This is supposed to be a function of `vars`, that is, the variables w.r.t. which the outer derivative is taken.
                &mut |outer_var_values: &[Object], _: &[Expression], context: Option<(&VarStack, &mut Env)>| {
                    let (_varstack, _env) = context.ok_or("Closure of numerical directional derivative requires `VarStack` and `Env`.")?;
                    let new_stack = _varstack.with_multiple(vars.iter(), outer_var_values.iter());
                    let mut warnings = Vec::new();
                    let res = analytic_directional_derivative(
                        inner_vars,
                        inner_expr,
                        &Status::from_iter(inner_point.iter(), |e| eval(e, &new_stack, _env))?.unpack_into(&mut warnings),
                        &Status::from_iter(inner_direction.iter(), |e| eval(e, &new_stack, _env))?.unpack_into(&mut warnings),
                        &new_stack,
                        _env
                    )?.unpack_into(&mut warnings);
                    Ok(Status{value: res, warnings})
                },
                point.to_vec(),
                direction.to_vec(),
                extra_vars,
                env
            )
        }
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
                Ok(Status {
                    value: hbx.mul(dvb)?.sub(hax.mul(dva)?)?.unpack_into(&mut warnings),
                    warnings
                })
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
        Expression::Limit(..) => numerical_directional_derivative(
            &mut (|parsed_args: &[Object], _: &[Expression], context: Option<(&VarStack, &mut Env)>| {
                let (_varstack, _env) = context.ok_or("[Unreachable] Function needs varstack and environment.".to_string())?;
                eval(
                    expr,
                    &_varstack.with_multiple(vars.iter(), parsed_args.iter()),
                    _env
                )}
            ),
            point.to_vec(),
            direction.to_vec(),
            extra_vars,
            env
        ),
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
    let mut warnings = Vec::new();
    // We use h = 1e-6 * (1 + |point|)
    let norm_of_point = point.iter().map(|x| match x {
        Object::Undefined | Object::Success | Object::LiteralExpression(_) | Object::Tuple(_) => Err(format!("Point can't contain object of type {:?}.", x)),
        Object::Real(x) => Ok(x.abs()),
        Object::Complex(x) => Ok(x.modulus()),
        Object::Vector(x) => Ok(dispatch_vector!(x, v => v.norm(&VectorNorm::P(2.0)))),
        Object::Matrix(x) => dispatch_matrix!(x, m => m.norm(&MatrixNorm::Frobenius))
    }).collect::<Result<Vec<_>, _>>()?;
    let h = 1e-6 * (1.0 + utils::min_of(norm_of_point.into_iter()).unwrap_or(0.0));
    for (i, coord) in point.iter_mut().enumerate() {
        direction[i] = h.mul(&direction[i]); // Spares us another operation later
        *coord = (&*coord).add(&direction[i])?.unpack_into(&mut warnings); // point + h*direction
    }
    let left_res = f(&point, &[], Some((extra_vars, env)))?.unpack_into(&mut warnings);
    for (i, coord) in point.iter_mut().enumerate() {
        *coord = (&*coord).sub((2.0).mul(&direction[i]))?.unpack_into(&mut warnings);
    }
    let right_res = f(&point, &[], Some((extra_vars, env)))?.unpack_into(&mut warnings);
    match (left_res, right_res) {
        (Object::Real(lhs), Object::Real(rhs)) => Ok(Object::Real((lhs - rhs) / (2.0 * h))),
        (Object::Vector(lhs_t), Object::Vector(rhs_t)) => {
            Ok(Object::Vector(dispatch_vector!(lhs_t, lhs => dispatch_vector!(
                rhs_t, rhs => {
                    lhs
                    .sub(rhs)
                    .ok_or("Couldn't evaluate f(x+h) - f(x-h). Traceback: Vectors of different sizes returned.")?
                    .div((2.0).mul(h))
                    .wrap_in_type()
                }
            ))))
        }
        (Object::Matrix(lhs_t), Object::Matrix(rhs_t)) => {
            Ok(Object::Matrix(dispatch_matrix!(lhs_t, lhs => dispatch_matrix!(
                rhs_t, rhs => {
                    lhs
                    .sub(rhs)
                    .ok_or("Couldn't evaluate f(x+h) - f(x-h). Traceback: Matrices of different sizes returned.")?
                    .div((2.0).mul(h))
                    .wrap_in_type()
                }
            ))))
        }
        _ => Err("Couldn't evaluate f(x+h) - f(x-h). Traceback: Objects have different types.".to_string())
    }
    .map(|value| Status{value, warnings})
}