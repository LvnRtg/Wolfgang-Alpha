//! Implements function around type-checking of expressions, e.g. `get_type` and `make_type_top_level`.

use itertools::Itertools;
use std::borrow::Cow;
use std::collections::HashMap;

use crate::{expr_binop, expr_binop_from_enum};
use crate::math::operations::{BinaryOperation, FoldedOperation, UnaryOperation};
use crate::math::{Env, FunctionRepr, Object, ObjType, VarStack, VarStackLookup};
use super::Expression;

impl Expression {
    /// Recursively determines what type this expression should output for the given context.
    /// 
    /// Returns `Err` if the type couldn't be determined.
    /// 
    /// Useful to return the correct types of object for e.g. empty sums. Still takes a regular
    /// `VarStack<Object>` instead of a custom `VarStack<ObjType>` because in
    /// general, the overhead of picking arbitrary representatives and wrapping them in `Object`
    /// is cheaper than converting the entire `VarStack<Object>` into a `VarStack<ObjType>`
    /// (the stack could be large).
    pub fn get_type(&self, extra_vars: &VarStack, env: &Env) -> Result<ObjType, String> {
        match self {
            Expression::None => Ok(ObjType::NonObject),
            Expression::Identifier(s) => {
                if let Some(obj) = extra_vars.lookup(s).or_else(|| env.constants.get(s)) {
                    Ok(obj.get_type())
                } else {
                    Err(format!("Unknown identifier: {:?}", s))
                }
            }
            Expression::Number(_) => Ok(ObjType::Scalar),
            Expression::Tuple(_) => Ok(ObjType::Tuple),
            Expression::Vector(v) => Ok(ObjType::Vector(v.len())),
            Expression::Matrix(m, n, _) => Ok(ObjType::Matrix(*m, *n)),
            Expression::UnaryOperation(op, r) => match op {
                UnaryOperation::Neg => r.get_type(extra_vars, env),
                UnaryOperation::Not => r.get_type(extra_vars, env),
                UnaryOperation::Factorial => match r.get_type(extra_vars, env)? {
                    t @ (ObjType::Scalar | ObjType::LiteralExpression | ObjType::NonObject) => Ok(t),
                    other => Err(format!("Operation 'Factorial' not valid for operand of type {:?}.", other))
                }
                UnaryOperation::Abs => match r.get_type(extra_vars, env)? {
                    t @ (ObjType::Scalar | ObjType::LiteralExpression | ObjType::NonObject) => Ok(t),
                    other => Err(format!("Operation 'Factorial' not valid for operand of type {:?}.", other))
                }
                UnaryOperation::Norm(_) => match r.get_type(extra_vars, env)? {
                    t @ (ObjType::LiteralExpression | ObjType::NonObject) => Ok(t),
                    _ => Ok(ObjType::Scalar)
                }
            }
            Expression::BinaryOperation(l, op, r) => {
                let ltype = l.get_type(extra_vars, env)?;
                let rtype = r.get_type(extra_vars, env)?;
                let err = || Err(format!("Operation '{}' invalid for operands {:?} and {:?}.", op, l, r));
                if matches!(ltype, ObjType::NonObject | ObjType::Tuple) || matches!(rtype, ObjType::NonObject | ObjType::Tuple) {
                    return err();
                }
                if matches!(ltype, ObjType::LiteralExpression) || matches!(rtype, ObjType::LiteralExpression) {
                    return Ok(ObjType::LiteralExpression)
                }
                // Remaining types: scalar, vector, matrix.
                match op {
                    BinaryOperation::Add | BinaryOperation::Sub if ltype == rtype => Ok(ltype),
                    BinaryOperation::Mul => match ltype {
                        ObjType::Scalar => Ok(rtype),
                        ObjType::Vector(k) => match rtype {
                            ObjType::Scalar => Ok(ObjType::Vector(k)),
                            ObjType::Vector(n) if n == k => Ok(ObjType::Scalar),
                            ObjType::Matrix(m, n) if m == k => Ok(ObjType::Vector(n)),
                            _ => err()
                        }
                        ObjType::Matrix(m, n) => match rtype {
                            ObjType::Scalar => Ok(ObjType::Matrix(m, n)),
                            ObjType::Vector(k) if k == n => Ok(ObjType::Vector(m)),
                            ObjType::Matrix(k, l) if k == n => Ok(ObjType::Matrix(m, l)),
                            _ => err()
                        }
                        _ => err()
                    }
                    BinaryOperation::Div | BinaryOperation::Rem | BinaryOperation::Quo if ltype == ObjType::Scalar => Ok(rtype),
                    BinaryOperation::Div | BinaryOperation::Rem | BinaryOperation::Quo if rtype == ObjType::Scalar => Ok(ltype),
                    BinaryOperation::Pow(_) if rtype == ObjType::Scalar => match ltype {
                        ObjType::Matrix(m, n) if m == n => Ok(ltype),
                        ObjType::Scalar => Ok(ltype),
                        _ => err()
                    }
                    BinaryOperation::And | BinaryOperation::Or if ltype == ObjType::Scalar && rtype == ObjType::Scalar => Ok(ObjType::Scalar),
                    BinaryOperation::Comp(..) => Ok(ObjType::Scalar),
                    _ => err()
                }
            }
            Expression::FoldedOperation(_, index_var_name, from, .., inner) => {
                /*
                Here, we trust that `inner` always returns the same type of object.
                There is one special case where this is not necessarily true: if the folded operation is a product of the form
                `prod_i f(i)` and `f(i)` returns a `g(i) x g(i+1)`-matrix. Then, the operation can successfully be evaluted
                even though its inner term is of variable type.
                However, this case can (currently) be disregarded for the following reason. The (currently) only application
                of this method `get_type` is to determine the type a folded operation should return so that when it runs over
                an empty range, we can return a default value of the correct type. In the case presented above, the returned
                type depends on the range ran over, so if the outer operation has an empty range, we couldn't always determine
                the true returned type anyway (only that it is a matrix, but not its dimension).
                */
                inner.get_type(&extra_vars.with(index_var_name, Cow::Owned(from.get_type(extra_vars, env)?.representative())), env)
            }
            Expression::Function(name, args) => {
                match env.functions.get(name) {
                    Some(FunctionRepr::ByExpression(varnames, expr)) => {
                        expr.get_type(
                            &VarStack::Frame {
                                vars: Cow::Owned(
                                    varnames.iter().zip(args)
                                    .map(|(v, a)| a.get_type(extra_vars, env).map(|t| (v, Cow::Owned(t.representative()))))
                                    .collect::<Result<HashMap<_, _>, _>>()?
                                ),
                                parent: extra_vars
                            },
                            env
                        )
                    }
                    Some(FunctionRepr::Direct(_, (m, n, b))) => {
                        // Accordingly with the mask, obtain the type of some arguments and leave others unchanged.
                        if args.len() < m + n {
                            return Err(format!("Wrong number of arguments provided for function '{}' (expected at least {}).", name, m + n));
                        }
                        let mut evaluated_arg_types = args.iter().take(*m).map(|a| a.get_type(extra_vars, env)).collect::<Result<Vec<_>, _>>()?;
                        if *b {
                            evaluated_arg_types.extend(args.iter().skip(m+n).map(|a| a.get_type(extra_vars, env)).collect::<Result<Vec<_>, _>>()?)
                        }
                        crate::defaults::get_default_fn_type(
                            name,
                            &evaluated_arg_types,
                            if *b {&args[*m .. (m+n)]} else {&args[*m..]},
                            extra_vars,
                            env
                        )
                    },
                    None => Err(format!("No such function: \"{name}\"."))
                }
            }
            Expression::Assignment(_, rhs) => rhs.get_type(extra_vars, env),
            Expression::PartialDerivative(..) => Ok(ObjType::LiteralExpression),
            Expression::DirectionalDerivative(vars, expr, point, _) => {
                expr.get_type(
                    &VarStack::Frame {
                        vars: Cow::Owned(
                            vars.iter().zip(point)
                            .map(|(v, a)| a.get_type(extra_vars, env).map(|t| (v, Cow::Owned(t.representative()))))
                            .collect::<Result<HashMap<_, _>, _>>()?
                        ),
                        parent: extra_vars
                    },
                    env
                )
            }
            Expression::Integral(func, .., wrt) => {
                // This time, we can assume truly w.l.o.g. that `func` always returns the same type,
                // otherwise the integral wouldn't be defined.
                // The integration variable has to be real.
                func.get_type(&extra_vars.with(wrt, Cow::Owned(Object::Real(1.0))), env)
            }
            // Below, we can have a problem if `iftrue` and `iffalse` are different. Logically, this shouldn't be the case,
            // but the user _can_ do this. However, there is no way to solve this since without knowing the free variables,
            // we can't simply check the condition to know which expression is returned.
            // Therefore, I decided to use `iftrue`.
            Expression::IfElse(_, iftrue, iffalse) => {
                let iftrue_type = iftrue.get_type(extra_vars, env)?;
                let iffalse_type = iffalse.get_type(extra_vars, env)?;
                if iftrue_type == iffalse_type {
                    Ok(iftrue_type)
                } else {
                    Err(format!("`if` arms have incompatible types: {:?}, {:?}.", iftrue_type, iffalse_type))
                }
            }
        }
    }

    /// Expands the given expression as to obtain one of the following:
    /// - An expression the type of which is a scalar (real/complex number)
    /// - An expression the type of which is a tuple
    /// - An `Expression::Vector`
    /// - An `Expression::Matrix`
    /// - An `ObjType::NonObject`
    /// - An `Expression::Assignment` the RHS of which is one of the above.
    /// For example, `(a, b) + (x, y)` would become `(a+x, b+y)`.
    /// 
    /// Returns this expression along with the corresponding `ObjType`.
    /// 
    /// Notes:
    /// - This function is used e.g. to analytically differentiate expressions like `d/dx ||f(x)||`, because
    ///   we then need to know the individual components of `f`.
    /// - The new expression may be slightly longer to evaluate because this function expands some matrix/vector
    ///   operations into folded operations (e.g. `A * v`) in order to make the distinct components apparent.
    ///   This causes the loss of strategies like parallelization and tiling.
    /// - This function does _not_ necessarily expand expressions like `x * (y + z)`.
    pub fn make_type_top_level(&self, extra_vars: &VarStack, env: &Env) -> Result<(Expression, ObjType), String> {
        // This implementation is more or less an extended version of `get_type()`.
        match self {
            // In quite a few cases, we can simply leave the expression as is if we know that it will be a real number anyway (e.g. `||f(x)||`).
            Expression::None | Expression::Identifier(_) | Expression::Number(_) | Expression::Tuple(_) | Expression::Vector(_) | Expression::Matrix(..)
                => self.get_type(extra_vars, env).map(|t| (self.clone(), t)),
            Expression::UnaryOperation(UnaryOperation::Abs, inner) | Expression::UnaryOperation(UnaryOperation::Factorial, inner) => {
                // No need to call `make_type_top_level`, since if `inner` is a scalar, `Abs` will return a scalar anyway.
                if matches!(inner.get_type(extra_vars, env)?, ObjType::Scalar | ObjType::LiteralExpression) {
                    Ok((Expression::UnaryOperation(UnaryOperation::Abs, inner.clone()), ObjType::Scalar))
                } else {
                    Err(format!("Operation 'Abs' invalid for operand {:?}.", inner))
                }
            }
            Expression::UnaryOperation(UnaryOperation::Norm(_), inner) => {
                if matches!(inner.get_type(extra_vars, env)?, ObjType::Scalar | ObjType::Vector(_) | ObjType::Matrix(..) | ObjType::LiteralExpression) {
                    Ok((Expression::UnaryOperation(UnaryOperation::Abs, inner.clone()), ObjType::Scalar))
                } else {
                    Err(format!("Operation 'Abs' invalid for operand {:?}.", inner))
                }
            }
            // Unary operations that are evaluated componentswise
            Expression::UnaryOperation(op, rhs) => {
                rhs.make_type_top_level(extra_vars, env)
                .map(|(e, t)| (match e {
                    Expression::Tuple(v) => Expression::Tuple(
                        v.into_iter().map(|x| Expression::UnaryOperation(op.clone(), Box::new(x))).collect()
                    ),
                    Expression::Vector(v) => Expression::Vector(
                        v.into_iter().map(|x| Expression::UnaryOperation(op.clone(), Box::new(x))).collect()
                    ),
                    Expression::Matrix(m, n, v) => Expression::Matrix(
                        m, n, v.into_iter().map(|x| Expression::UnaryOperation(op.clone(), Box::new(x))).collect()
                    ),
                    other => Expression::UnaryOperation(op.clone(), Box::new(other))
                }, t))
            }
            Expression::BinaryOperation(l, op, r) => {
                let (lexpr, ltype) = l.make_type_top_level(extra_vars, env)?;
                let (rexpr, rtype) = r.make_type_top_level(extra_vars, env)?;
                let err = || Err(format!("Operation '{}' invalid for operands {:?} and {:?}.", op, l, r));
                if matches!(ltype, ObjType::NonObject | ObjType::Tuple) || matches!(rtype, ObjType::NonObject | ObjType::Tuple) {
                    return err();
                }
                if matches!(ltype, ObjType::LiteralExpression) || matches!(rtype, ObjType::LiteralExpression) {
                    return Ok((expr_binop_from_enum!(lexpr, op.clone(), rexpr), ObjType::LiteralExpression))
                }
                // At this point, `ltype` and `rtype` can only be `ObjType::Scalar`, `ObjType::Vector` or `ObjType::Matrix`.
                // Moreover, `lexpr` and `rexpr` are therefore (by definition of `make_type_top_level`) one of the following:
                // `Expression::Vector`, `Expression::Matrix` or some expression that returns a scalar.
                // Hence, in the below match statements, we generally only need to type out the cases `Vector` and `Matrix`.
                match op {
                    BinaryOperation::Add | BinaryOperation::Sub if ltype == rtype => match (lexpr, rexpr) {
                        (Expression::Vector(v), Expression::Vector(w)) => Ok((
                            Expression::Vector(
                                v.into_iter().zip(w.into_iter()).map(|(x, y)| expr_binop_from_enum!(x, op.clone(), y)).collect()
                            ),
                            ltype
                        )),
                        (Expression::Matrix(m, n, v), Expression::Matrix(_m, _n, w)) => {
                            if _m == m && _n == n {
                                Ok((
                                    Expression::Matrix(
                                        m, n,
                                        v.into_iter().zip(w.into_iter()).map(|(x, y)| expr_binop_from_enum!(x, op.clone(), y)).collect()
                                    ),
                                    ltype
                                ))
                            } else {
                                err()
                            }
                        }
                        (other_l, other_r) => Ok((expr_binop_from_enum!(other_l, op.clone(), other_r), ltype))
                    }
                    BinaryOperation::Mul => match lexpr {
                        Expression::Vector(v) => match rexpr {
                            Expression::Vector(w) => {
                                if w.len() == v.len() {
                                    Ok((
                                        expr_binop!(Expression::Vector(v), Mul, Expression::Vector(w)),
                                        ObjType::Scalar
                                    ))
                                } else {
                                    err()
                                }
                            }
                            Expression::Matrix(m, n, mut w) => {
                                if m == v.len() {
                                    // (v^T * A)_j = \sum_{i=1}^m v_i A_{i,j} = <v, A_{.,j}>
                                    // We can't efficiently formulate this using `Expression::FoldedOperation` (currently)
                                    // because there is not necessarily a clean expression `f(i)` such that `v = (f(1), ..., f(m))`.
                                    Ok((
                                        Expression::Vector({
                                            (0..n).map(|j|
                                                expr_binop!(
                                                    Expression::Vector(v.clone()),
                                                    Mul,
                                                    // Trick to consume `w` non-linearly: use the fact that `Expression: Default` and `std::mem::take`
                                                    // to swap out an existing value for the default. Correctness is guaranteed by the fact that we
                                                    // do not visit an index twice.
                                                    Expression::Vector((0..m).map(|i| std::mem::take(&mut w[i * n + j])).collect())
                                                )
                                            ).collect()
                                        }),
                                        ObjType::Vector(n)
                                    ))
                                } else {
                                    err()
                                }
                            }
                            other_r => {
                                let n = v.len();
                                Ok((
                                    Expression::Vector(v.into_iter().map(|x| expr_binop!(x, Mul, other_r.clone())).collect()),
                                    ObjType::Vector(n)
                                ))
                            }
                        }
                        Expression::Matrix(m, n, v) => match rexpr {
                            Expression::Vector(w) => {
                                if w.len() == n {
                                    // (A * w)_i = \sum_{j=1}^n A_{i,j} * w_j = <A_{i,.}, w>
                                    Ok((
                                        Expression::Vector({
                                            let mut it = v.into_iter();
                                            (0..m).map(|_|
                                                expr_binop!(
                                                    Expression::Vector((0..n).map(|_| it.next().unwrap()).collect()),
                                                    Mul,
                                                    Expression::Vector(w.clone())
                                                )
                                            ).collect()
                                        }),
                                        ObjType::Vector(m)
                                    ))
                                } else {
                                    err()
                                }
                            }
                            Expression::Matrix(_m, _n, w) => {
                                if n == _m {
                                    Ok((
                                        Expression::Matrix(m, _n, {
                                            (0..m).map(|_| 0.._n).multi_cartesian_product().map(
                                                |__v| {
                                                    // `.clone()` below is necessary since the same row of `v` / column of `w` is reused multiple times.
                                                    expr_binop!(
                                                        Expression::Vector((0..n).map(|k| v[__v[0] * n + k].clone()).collect()), // v_{i,.}
                                                        Mul,
                                                        Expression::Vector((0..n).map(|k| w[k * n + __v[1]].clone()).collect())  // w_{.,j}
                                                    )
                                                }
                                            ).collect()
                                        }),
                                        ObjType::Matrix(m, _n)
                                    ))
                                } else {
                                    err()
                                }
                            }
                            other_r => Ok((
                                Expression::Matrix(m, n, v.into_iter().map(|x| expr_binop!(x, Mul, other_r.clone())).collect()),
                                ObjType::Matrix(m, n)
                            ))
                        }
                        other_l => Ok(( // `other_l` has type `Scalar`
                            match rexpr {
                                Expression::Vector(v) => Expression::Vector(v.into_iter().map(|x| expr_binop!(other_l.clone(), Mul, x)).collect()),
                                Expression::Matrix(m, n, v) => Expression::Matrix(
                                    m, n,
                                    v.into_iter().map(|x| expr_binop!(other_l.clone(), Mul, x)).collect()
                                ),
                                other_r => expr_binop!(other_l, Mul, other_r)
                            },
                            rtype
                        ))
                    }
                    // The following operations are valid iff at least one operand is a scalar.
                    op @ (BinaryOperation::Div | BinaryOperation::Rem | BinaryOperation::Quo) => {
                        if ltype == ObjType::Scalar {
                            Ok((match rexpr {
                                Expression::Vector(v) => Expression::Vector(
                                    v.into_iter().map(|x| expr_binop_from_enum!(lexpr.clone(), op.clone(), x)).collect()
                                ),
                                Expression::Matrix(m, n, v) => Expression::Matrix(
                                    m, n,
                                    v.into_iter().map(|x| expr_binop_from_enum!(lexpr.clone(), op.clone(), x)).collect()
                                ),
                                other_r => expr_binop_from_enum!(lexpr, op.clone(), other_r)
                            }, rtype))
                        } else if rtype == ObjType::Scalar {
                            Ok((match lexpr {
                                Expression::Vector(v) => Expression::Vector(
                                    v.into_iter().map(|x| expr_binop_from_enum!(x, op.clone(), rexpr.clone())).collect()
                                ),
                                Expression::Matrix(m, n, v) => Expression::Matrix(
                                    m, n,
                                    v.into_iter().map(|x| expr_binop_from_enum!(x, op.clone(), rexpr.clone())).collect()
                                ),
                                other_l => expr_binop_from_enum!(other_l, op.clone(), rexpr)
                            }, rtype))
                        } else {
                            err()
                        }
                    }
                    BinaryOperation::Pow(_) if rtype == ObjType::Scalar => {
                        match lexpr {
                            Expression::Matrix(m, n, v) => {
                                // We interpret this as `\prod_{i=1}^rexpr self`
                                if m == n {
                                    Ok((
                                        Expression::Matrix(
                                            n, n,
                                            (0..n).map(|_| 0..n).multi_cartesian_product().map(
                                                |__v| Expression::Function(
                                                    "___helper_matrix_prod".to_string(),
                                                    vec![
                                                        Expression::Number(__v[0] as f64), // k_a
                                                        Expression::Number(__v[1] as f64), // k_{b+1}
                                                        Expression::Number(1.0), // a
                                                        rexpr.clone(), // b
                                                        Expression::Identifier("_".to_string()), // i (not used inside, so choose "_")
                                                        Expression::Matrix(m, n, v.clone())
                                                    ]
                                                )
                                            ).collect()
                                        ),
                                        ObjType::Matrix(n, n)
                                    ))
                                } else {
                                    err()
                                }
                            }
                            Expression::Vector(_) => err(),
                            other => Ok((expr_binop!(other, Pow(true), rexpr), ObjType::Scalar))
                        }
                    }
                    // The following operations are valid iff both operands are scalars.
                    op @ (BinaryOperation::And | BinaryOperation::Or) if ltype == ObjType::Scalar && rtype == ObjType::Scalar => {
                        Ok((expr_binop_from_enum!(lexpr, op.clone(), rexpr), ObjType::Scalar))
                    }
                    BinaryOperation::Comp(c, opt) => Ok((expr_binop!(lexpr, Comp(c.clone(), opt.clone()), rexpr), ObjType::Scalar)),
                    _ => err()
                }
            }
            Expression::FoldedOperation(FoldedOperation::Sum, index_var_name, from, conditions, to, inner) => {
                let (iexpr, itype) = inner.make_type_top_level(
                    &extra_vars.with(index_var_name, Cow::Owned(Object::Real(1.0))),
                    env
                )?;
                if matches!(itype, ObjType::NonObject | ObjType::Tuple) {
                    Err(format!("Operation 'Sum' invalid for operand {:?}.", iexpr))
                } else {
                    Ok((
                        match iexpr {
                            Expression::Vector(v) => Expression::Vector(
                                v.into_iter().map(|x| Expression::FoldedOperation(
                                    FoldedOperation::Sum, index_var_name.clone(), from.clone(), conditions.clone(), to.clone(), Box::new(x)
                                )).collect()
                            ),
                            Expression::Matrix(m, n, v) => Expression::Matrix(
                                m, n,
                                v.into_iter().map(|x| Expression::FoldedOperation(
                                    FoldedOperation::Sum, index_var_name.clone(), from.clone(), conditions.clone(), to.clone(), Box::new(x)
                                )).collect()
                            ),
                            other => Expression::FoldedOperation(
                                FoldedOperation::Sum, index_var_name.clone(), from.clone(), conditions.clone(), to.clone(), Box::new(other)
                            )
                        },
                        itype
                    ))
                }
            }
            Expression::FoldedOperation(FoldedOperation::Product, index_var_name, from, conditions, to, inner) => {
                let (iexpr, itype) = inner.make_type_top_level(
                    &extra_vars.with(index_var_name, Cow::Owned(Object::Real(1.0))),
                    env
                )?;
                if matches!(itype, ObjType::NonObject | ObjType::Tuple | ObjType::Vector(_)) {
                    Err(format!("Operation 'Product' invalid for operand {:?}.", iexpr))
                } else {
                    match iexpr {
                        Expression::Matrix(m, n, v) => {
                            if m == n {
                                Ok((
                                    Expression::Matrix(
                                        n, n,
                                        (0..n).map(|_| 0..n).multi_cartesian_product().map(
                                            |__v| {
                                                let mut args = vec![
                                                    Expression::Number(__v[0] as f64), // k_a
                                                    Expression::Number(__v[1] as f64), // k_{b+1}
                                                    *from.clone(),
                                                    *to.clone(),
                                                    Expression::Identifier(index_var_name.clone()),
                                                    Expression::Matrix(m, n, v.clone())
                                                ];
                                                args.extend(conditions.iter().cloned());
                                                Expression::Function(
                                                    "___helper_matrix_prod".to_string(),
                                                    args
                                                )
                                            }
                                        ).collect()
                                    ),
                                    ObjType::Matrix(n, n)
                                ))
                            } else {
                                Err(format!("Operation 'FoldedOperation::Product' invalid for non-square matrix (got {}x{}).", m, n))
                            }
                        }
                        other => Ok((
                            Expression::FoldedOperation(
                                FoldedOperation::Sum, index_var_name.clone(), from.clone(), conditions.clone(), to.clone(), Box::new(other)
                            ),
                            ObjType::Scalar
                        ))
                    }
                }
            }
            Expression::Function(name, args) => {
                match env.functions.get(name) {
                    Some(FunctionRepr::ByExpression(varnames, defining_expr)) => {
                        // If this is a `FunctionRepr::ByExpression`:
                        // Idea: make the type of `defining_expr` top-level and then simply replace `f(x)` by `defining_expr`.
                        // Evidently, this requires us to replace `varnames` within `defining_expr` by the corresponding given argument in `args`.
                        let (mut iexpr, itype) = defining_expr.make_type_top_level(
                            &VarStack::Frame {
                                vars: Cow::Owned(
                                    varnames.iter().zip(args)
                                    .map(|(v, a)| a.get_type(extra_vars, env).map(|t| (v, Cow::Owned(t.representative()))))
                                    .collect::<Result<HashMap<_, _>, _>>()?
                                ),
                                parent: extra_vars
                            },
                            env
                        )?;
                        for (varname, arg) in varnames.iter().zip(args) {
                            iexpr.replace_identifiers_in_place(varname, arg);
                        }
                        Ok((iexpr, itype))
                    }
                    Some(FunctionRepr::Direct(_, (m, n, b))) => {
                        // Accordingly with the mask, obtain the type of some arguments and leave others unchanged.
                        if args.len() < m + n {
                            return Err(format!("Wrong number of arguments provided for function '{}' (expected at least {}).", name, m + n));
                        }
                        let mut evaluated_args = args.iter().take(*m).map(|a| a.make_type_top_level(extra_vars, env)).collect::<Result<Vec<_>, _>>()?;
                        if *b {
                            evaluated_args.extend(args.iter().skip(m+n).map(|a| a.make_type_top_level(extra_vars, env)).collect::<Result<Vec<_>, _>>()?)
                        }
                        crate::defaults::make_default_fn_type_top_level(
                            name,
                            evaluated_args,
                            if *b {&args[*m .. (m+n)]} else {&args[*m..]},
                            extra_vars,
                            env
                        )
                    }
                    None => Err(format!("No such function: \"{name}\"."))
                }
            }
            Expression::Assignment(lhs, rhs) => {
                rhs.make_type_top_level(extra_vars, env)
                .map(|(rexpr, rtype)| (
                    Expression::Assignment(lhs.clone(), Box::new(rexpr)),
                    rtype
                ))
            }
            Expression::PartialDerivative(wrt, inner) => {
                let (iexpr, itype) = inner.make_type_top_level(
                    &extra_vars.with(wrt, Cow::Owned(Object::Real(1.0))),
                    env
                )?;
                if matches!(itype, ObjType::NonObject | ObjType::Tuple) {
                    Err(format!("Operation 'PartialDerivative' invalid for operand {:?}.", iexpr))
                } else {
                    Ok((
                        match iexpr {
                            Expression::Vector(v) => Expression::Vector(v.into_iter().map(
                                |x| Expression::PartialDerivative(wrt.clone(), Box::new(x))
                            ).collect()),
                            Expression::Matrix(m, n, v) => Expression::Matrix(m, n, v.into_iter().map(
                                |x| Expression::PartialDerivative(wrt.clone(), Box::new(x))
                            ).collect()),
                            other => Expression::PartialDerivative(wrt.clone(), Box::new(other))
                        },
                        ObjType::LiteralExpression
                    ))
                }
            }
            Expression::DirectionalDerivative(vars, inner, point, direction) => {
                let varstack = VarStack::Frame {
                    vars: Cow::Owned(
                        vars.iter().zip(point)
                        .map(|(v, a)| a.get_type(extra_vars, env).map(|t| (v, Cow::Owned(t.representative()))))
                        .collect::<Result<HashMap<_, _>, _>>()?
                    ),
                    parent: extra_vars
                };
                let (iexpr, itype) = inner.make_type_top_level(&varstack, env)?;
                if matches!(itype, ObjType::NonObject | ObjType::Tuple) {
                    Err(format!("Operation 'DirectionalDerivative' invalid for operand {:?}.", iexpr))
                } else {
                    Ok((
                        match iexpr {
                            Expression::Vector(v) => Expression::Vector(v.into_iter().map(
                                |x| Expression::DirectionalDerivative(vars.clone(), Box::new(x), point.clone(), direction.clone())
                            ).collect()),
                            Expression::Matrix(m, n, v) => Expression::Matrix(m, n, v.into_iter().map(
                                |x| Expression::DirectionalDerivative(vars.clone(), Box::new(x), point.clone(), direction.clone())
                            ).collect()),
                            other => Expression::DirectionalDerivative(vars.clone(), Box::new(other), point.clone(), direction.clone())
                        },
                        itype
                    ))
                }
            }
            Expression::Integral(inner, a, b, wrt) => {
                let (iexpr, itype) = inner.make_type_top_level(
                    &extra_vars.with(wrt, Cow::Owned(Object::Real(1.0))),
                    env
                )?;
                if matches!(itype, ObjType::NonObject | ObjType::Tuple) {
                    Err(format!("Operation 'Integral' invalid for operand {:?}.", iexpr))
                } else {
                    Ok((
                        match iexpr {
                            Expression::Vector(v) => Expression::Vector(v.into_iter().map(
                                |x| Expression::Integral(Box::new(x), a.clone(), b.clone(), wrt.clone())
                            ).collect()),
                            Expression::Matrix(m, n, v) => Expression::Matrix(m, n, v.into_iter().map(
                                |x| Expression::Integral(Box::new(x), a.clone(), b.clone(), wrt.clone())
                            ).collect()),
                            other => Expression::Integral(Box::new(other), a.clone(), b.clone(), wrt.clone())
                        },
                        itype
                    ))
                }
            }
            Expression::IfElse(condition, iftrue, iffalse) => {
                let (texpr, ttype) = iftrue.make_type_top_level(extra_vars, env)?;
                let (fexpr, ftype) = iffalse.make_type_top_level(extra_vars, env)?;
                if ttype == ftype {
                    Ok((match (texpr, fexpr) {
                        (Expression::Vector(v), Expression::Vector(w)) => Expression::Vector(
                            v.into_iter().zip(w.into_iter()).map(
                                |(x, y)| crate::expr_if_else!(*condition.clone(), x, y)
                            ).collect()
                        ),
                        (Expression::Matrix(m, n, v), Expression::Matrix(.., w)) => Expression::Matrix(
                            m, n,
                            v.into_iter().zip(w.into_iter()).map(
                                |(x, y)| crate::expr_if_else!(*condition.clone(), x, y)
                            ).collect()
                        ),
                        (tother, fother) => crate::expr_if_else!(*condition.clone(), tother, fother)
                    }, ttype))
                } else {
                    Err(format!("`if` arms have incompatible types: {:?}, {:?}.", ttype, ftype))
                }
            }
        }
    }
}