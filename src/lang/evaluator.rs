//! Responsible for evaluating an `Expression` to an `Object`.

use std::collections::HashMap;
use std::collections::HashSet;
use statrs::function::gamma;
use itertools::Itertools;

use crate::math;
use crate::math::{Env, Expression, FunctionRepr, Object, VarStack};
use crate::math::objects::try_operation;
use crate::math::operations::{BinaryOperation, UnaryOperation};
use crate::math::utils::{approx_eq, linspace_as_objects};
use crate::status::{ExtResult, Status};


const DEFAULT_TESTEQ_REPETITIONS: usize = 20;

/// Used to prevent the user from defining constants/functions with names such as "if".
const KEYWORDS: [&str; 2] = [
    "if", "else"
];


/// When an function definition is encountered, the expression on the RHS is processed in a special way.
/// Generally, it has to be cloned (cleanest way to work with the 'eval' function below), which is the main action this function performs.
/// 
/// Parsing the expression recursively, every identifier that is NOT declared as an argument of the function is replaced
/// by the constant it represents in the current environment. Identifiers can be declared as arguments even if they exist in the environment;
/// the environmental value will then be ignored. Moreover, every identifier that is declared as an argument is prefixed with three underscores
/// (this will be needed for evaluation). For example, if 'constants = {"x": 1, "y": 2}', the RHS of the literal expression
/// "f(y, z) := x + 3*y + z" will become "1 + 3*___tmp_y + ___tmp_z".
/// 
/// If an `Expression::Function(f, args)` is encountered where `f` corresponds to a direct function with mask `(m, n, b)`,
/// we only recursively process the elements of `args` that would be evaluated by `f` when called. The other elements of `args`
/// are left untouched. For instance, if `b == false`, we only recursively parse `args[..m]`.
/// 
/// I have decided that if the definition depends on another function (say, "f(x, y) = g(x) + y"), the other function shall
/// _not_ be replaced by its literal expression. It makes sense to me to capture the current values of free variables because
/// if this were not intended, one could simply include them as parameters, but this solution isn't available for functions
/// (yet), hence the decision.
/// 
/// This cannot avoid cloning objects (e.g. matrices) because if a variable (say, "x" in above example) is changed later, we
/// still want the function to behave the same, so we'd have to keep the old value stored somewhere anyway.
/// However, is doesn't matter if this function is relatively expensive to call since function definitions are rare.
/// 
/// Note: for the definition of constants, this is not necessary, since constants have to be computable at the moment they are defined.
pub fn parse_function_definition(
    expr: &Expression,
    argument_names: &Vec<String>,
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Status<Expression>, String> {
    match expr {
        Expression::None => Ok(Status::ok(Expression::None)),
        Expression::Identifier(x) => Ok(Status::ok(
            if argument_names.contains(x) {
                Expression::Identifier(format!("___tmp_{}", x))
            } else if let Some(y) = env.constants.get(x) {
                y.to_expression()
            } else if let Some(y) = extra_vars.lookup(x) {
                y.to_expression()
            } else {
                Expression::Identifier(x.clone())
            }
        )),
        Expression::Number(x) => Ok(Status::ok(Expression::Number(*x))),
        Expression::Tuple(v) => Status::from_iter(
            v.iter(),
            |x| parse_function_definition(x, argument_names, extra_vars, env)
        )
        .map(|s| s.map(Expression::Tuple)),
        Expression::Vector(v) => Status::from_iter(
            v.iter(),
            |x| parse_function_definition(x, argument_names, extra_vars, env)
        )
        .map(|s| s.map(Expression::Vector)),
        Expression::Matrix(m, n, v) => Status::from_iter(
            v.iter(),
            |x| parse_function_definition(x, argument_names, extra_vars, env)
        )
        .map(|s| s.map(|val| Expression::Matrix(*m, *n, val))),
        Expression::UnaryOperation(op, rhs)
            => parse_function_definition(rhs, argument_names, extra_vars, env)
            .map(|s| s.map(
                |expr| Expression::UnaryOperation(op.clone(), Box::new(expr)))
            ),
        Expression::BinaryOperation(lhs, op, rhs) => Status::combine(
            parse_function_definition(lhs, argument_names, extra_vars, env)?,
            parse_function_definition(rhs, argument_names, extra_vars, env)?,
            |lhs, rhs| Ok(Expression::BinaryOperation(Box::new(lhs), op.clone(), Box::new(rhs)))
        ),
        Expression::FoldedOperation(op, varname, from, conditions, to, inner) => {
            let mut warnings = Vec::<String>::new();
            Ok(Status {
                value: Expression::FoldedOperation(
                    op.clone(),
                    varname.clone(),
                    Box::new(parse_function_definition(from, argument_names, extra_vars, env)?.unpack_into(&mut warnings)),
                    Status::from_iter(
                        conditions.iter(),
                        |x| parse_function_definition(x, argument_names, extra_vars, env)
                    )?.unpack_into(&mut warnings),
                    Box::new(parse_function_definition(to, argument_names, extra_vars, env)?.unpack_into(&mut warnings)),
                    // Notice that if `varname` is simultaneously an argument of the function, it shouldn't be replace by ___tmp_...
                    // within `inner`. For example, `g(x) := \sum_{x=1}^2 x` should be equivalent to `g(x) := \sum_{i=1}^2 i`.
                    if let Some(i) = argument_names.iter().position(|n| n == varname) {
                        if argument_names.len() >= 2 {
                            // Still replace all identifiers in `inner` except the one at index `i`. Note that cloning is fine here since the list of argument names
                            // should be very small (having more than two would already be very rare).
                            Box::new(parse_function_definition(
                                inner,
                                &argument_names.iter().enumerate().filter(|&(idx, _)| idx != i).map(|(_, x)| x.clone()).collect(),
                                extra_vars,
                                env
                            )?.unpack_into(&mut warnings))
                        } else {
                            inner.clone()
                        }
                    } else {
                        Box::new(parse_function_definition(inner, argument_names, extra_vars, env)?.unpack_into(&mut warnings))
                    }
                ),
                warnings
            })
        }
        Expression::Function(function_name, args) => {
            // Direct function => only parse the elements of `args` that shall be evaluated in a call of `function_name`,
            // i.e. `args[..m]`, and if `b == true`, then in addition `args[m+n..]`
            // Below `opt` allows us to use `env` again later. It is just a copy-operation anyway.
            let opt = if let Some(FunctionRepr::Direct(_, (m, n, b))) = env.functions.get(function_name) {
                Some((*m, *n, *b))
            } else {None};
            if let Some((m, n, b)) = opt {
                if args.len() < m+n {
                    return Err(format!("Wrong number of arguments provided for function '{}' (expected at least {}, got {}).", function_name, m+n, args.len()));
                }
                let Status{value: mut processed_args, mut warnings} = Status::from_iter(
                    args.iter().take(m),
                    |x| parse_function_definition(x, argument_names, extra_vars, env)
                )?;
                processed_args.extend(args.iter().skip(m).take(n).cloned());
                if b {
                    processed_args.reserve(args.len() - (m+n));
                    for x in args.iter().skip(m+n) {
                        processed_args.push(parse_function_definition(x, argument_names, extra_vars, env)?.unpack_into(&mut warnings))
                    }
                } else {
                    processed_args.extend(args.iter().skip(m+n).cloned());
                }
                Ok(Status {
                    value: Expression::Function(
                        function_name.clone(),
                        processed_args
                    ),
                    warnings
                })
            } else {
                Status::from_iter(
                    args.iter(),
                    |x| parse_function_definition(x, argument_names, extra_vars, env)
                )
                .map(|s| s.map(
                    |processed_args| Expression::Function(function_name.clone(), processed_args)
                ))
            }
        }
        Expression::Assignment(lhs, rhs) => Status::combine(
            parse_function_definition(lhs, argument_names, extra_vars, env)?,
            parse_function_definition(rhs, argument_names, extra_vars, env)?,
            |lhs, rhs| Ok(Expression::Assignment(Box::new(lhs), Box::new(rhs)))
        ),
        Expression::PartialDerivative(wrt, expr) => {
            math::differentiation::analytic_partial_derivative(expr, wrt, extra_vars, env)?
            .try_map_flatten(|expr| parse_function_definition(&expr, argument_names, extra_vars, env))
        }
        Expression::DirectionalDerivative(vars, expr, point, direction) => Status::combine_three(
            parse_function_definition(expr, argument_names, extra_vars, env)?,
            Status::from_iter(point.iter(), |x| parse_function_definition(x, argument_names, extra_vars, env))?,
            Status::from_iter(direction.iter(), |x| parse_function_definition(x, argument_names, extra_vars, env))?,
            |processed_expr, point, direction| Ok(
                Expression::DirectionalDerivative(vars.clone(), Box::new(processed_expr), point, direction)
            )
        ),
        Expression::Integral(inner, a, b, wrt) => Status::combine_three(
            // As for folded operations
            if let Some(i) = argument_names.iter().position(|n| n == wrt) {
                if argument_names.len() >= 2 {
                    parse_function_definition(
                        inner,
                        &argument_names.iter().enumerate().filter(|&(idx, _)| idx != i).map(|(_, x)| x.clone()).collect(),
                        extra_vars,
                        env
                    )?
                } else {
                    Status::ok(*inner.clone())
                }
            } else {
                parse_function_definition(inner, argument_names, extra_vars, env)?
            },
            parse_function_definition(a, argument_names, extra_vars, env)?,
            parse_function_definition(b, argument_names, extra_vars, env)?,
            |u, v, w| Ok(Expression::Integral(
                Box::new(u),
                Box::new(v),
                Box::new(w),
                wrt.clone()
            ))
        ),
        Expression::IfElse(x, y, z) => Status::combine_three(
            parse_function_definition(x, argument_names, extra_vars, env)?,
            parse_function_definition(y, argument_names, extra_vars, env)?,
            parse_function_definition(z, argument_names, extra_vars, env)?,
            |u, v, w| Ok(crate::expr_if_else!(u, v, w))
        ),
    }
}


/// Evaluates a given expression and returns the computed value (as reference, see below).
/// Requires knowledge of the environment, i.e. the hashmaps 'constants' and 'functions'.
/// 1. If the expression can be computed directly (e.g. "2+3" or "5*x" where constants.contains("x")), returns its value as type 'Object'.
/// 2. If the expression is a valid definition (e.g. "x := 7" or "f(x) := 5*x+2"), modifies the environment accordingly and returns 'Object.Success'.
///    
/// Moreover, `extra_vars` allows to specify identifiers that temporarily should have a certain value. Each hashmap in `extra_vars` should map
/// identifiers to objects. The outer `Vec` acts as stack: this function first searches for identifers in the last hashmap in `extra_vars`, then
/// in the fore-last, etc. until a match is found or the start of the vector is reached. The reason for this becomes apparent in the case
/// `Expression::Function`: for recursive function calls, it is simpler to pass more and more hashmap references to `eval` than to modify
/// the existing hashmap and later revert it to its old value.
/// 
/// If the evaluation fails, returns the corresponding error message (wrapped in a 'Result').
pub fn eval(
    expr: &Expression,
    extra_vars: &VarStack,
    env: &mut Env
) -> ExtResult {
    match expr {
        Expression::None => Err("Received empty expression.".to_string()),
        Expression::Identifier(ident) => {
            // First, iterate `extra_vars` in reverse order and search for `ident`.
            if let Some(x) = extra_vars.lookup(ident) {
                Ok(Status::ok(x.clone()))
            }
            // If nothing is found, look in `constants`.
            else if let Some(x) = env.constants.get(ident) {
                Ok(Status::ok(x.clone()))
                // We only call 'clone' for every time a variable from 'constants' is used, which can only happen so often
                // since the user still has to enter at least one character per time it is used. Therefore,
                // even if these are large matrices, it is a totally acceptable runtime.
            }
            // If still, nothing is found, this is an error.
            else {
                Err(format!("Unknown identifier: {:?}", ident))
            }
        }
        Expression::Number(x) => Ok(Status::ok(Object::Real(*x))),
        Expression::Tuple(entries) => {
            // As mentioned in the docs, we capture the environment for tuple evaluation.
            // Two approaches:
            // 1. Capture `env` at the start, clone it for every `x` in `entries`, call `eval(x)` with is, merge it into `env` after `eval(x)`.
            //    Slightly more overhead (+1 clone) but O(1) space.
            // 2. No capture at the start, proceed as above but only merge with `env` at the end of ALL `eval` calls.
            //    Less overhead but O(n) space.
            // => Choose 1.
            let captured_env = env.clone();
            let mut res = Status::ok(Vec::<Object>::with_capacity(entries.len()));
            for x in entries.iter() {
                let mut tmp_env = captured_env.clone();
                res.push(
                    eval(x, extra_vars, &mut tmp_env).map_err(|e| format!("Couldn't evaluate entry {}. Traceback: {}", x, e))?
                );
                env.update(tmp_env);
            }
            Ok(res.map(Object::Tuple))
        }
        Expression::Vector(entries) => Ok(
            eval_mul_exprs_to_f64(entries, extra_vars, env)?
            .map(|values| Object::Vector(math::Vector{ values }))
        ),
        Expression::Matrix(m, n, entries) => Ok(
            eval_mul_exprs_to_f64(entries, extra_vars, env)?
            .map(|values| Object::Matrix(math::Matrix::from(*m, *n, values)))
        ),
        Expression::UnaryOperation(op, rhs) => {
            match op {
                UnaryOperation::Neg => eval(rhs, extra_vars, env)?.neg(),
                UnaryOperation::Not => eval(rhs, extra_vars, env)?.not(),
                UnaryOperation::Factorial => eval(rhs, extra_vars, env)?.try_map(|o| {
                    match o {
                        Object::Success => Ok(Object::Success),
                        Object::Real(x) => Ok(Object::Real({
                            let r = x.round();
                            if approx_eq(x, r) && r >= 0.0 { // Avoid calling the gamma function if unnecessary
                                if r <= 1.0 {
                                    1.0
                                } else {
                                    let n = r as u64;
                                    (1..=n).try_fold(1, u64::checked_mul).ok_or(format!("Overflow occured while computing {n}!"))? as f64
                                }
                            } else {
                                gamma::gamma(x + 1.0)
                            }
                        })),
                        Object::LiteralExpression(e) => Ok(Object::LiteralExpression(Expression::UnaryOperation(UnaryOperation::Factorial, Box::new(e)))),
                        other => Err(format!("Operation 'Factorial' not valid for operand {other}.")),
                    }
                }),
                UnaryOperation::Abs => eval(rhs, extra_vars, env)?.try_map(|o| {
                    match o {
                        Object::Success => Ok(Object::Success),
                        Object::Real(x) => Ok(Object::Real(x.abs())),
                        Object::Complex(x) => Ok(Object::Real(x.modulus())),
                        Object::LiteralExpression(e) => Ok(Object::LiteralExpression(Expression::UnaryOperation(UnaryOperation::Abs, Box::new(e)))),
                        other => Err(format!("Operation 'Abs' not valid for operand {other}.")),
                    }
                }),
                UnaryOperation::Norm(opt) => {
                    let Status{value: obj, mut warnings} = eval(rhs, extra_vars, env)?;
                    Ok(Status {
                        value: match obj {
                            Object::Success => Object::Success,
                            Object::Real(x) => {
                                warnings.push(format!("Called `||x||` on real number `x = {}`; prefer `|x|` instead.", x));
                                Object::Real(x.abs())
                            }
                            Object::Complex(x) => {
                                warnings.push(format!("Called `||x||` on complex number `x := {}`; prefer `|x|` instead.", x));
                                Object::Real(x.modulus())
                            }
                            Object::Vector(x) => {
                                let Status{value: norm_type, warnings: new_warnings} = math::matrices_and_vectors::VectorNorm::from_expr(opt, extra_vars, env)?;
                                warnings.extend(new_warnings);
                                Object::Real(x.norm(&norm_type))
                            }
                            Object::Matrix(x) => {
                                let Status{value: norm_type, warnings: new_warnings} = math::matrices_and_vectors::MatrixNorm::from_expr(opt, extra_vars, env)?;
                                warnings.extend(new_warnings);
                                Object::Real(x.norm(&norm_type)?)
                            }
                            Object::LiteralExpression(e) => Object::LiteralExpression(
                                Expression::UnaryOperation(UnaryOperation::Norm(opt.clone()), Box::new(e))
                            ),
                            other => return Err(format!("Operation 'Norm' not valid for operand {other}.")),
                        },
                        warnings
                    })
                }
            }
        }
        Expression::BinaryOperation(lhs, op, rhs) => {
            // Check if the operation is a comparison and at least one of `lhs`, `rhs` is a function (which we'll call `this`; we'll call the remaining one `other`).
            // Here, being a function means having unknown identifiers within.
            if let BinaryOperation::Comp(_, precision_expr) = op {
                let mut lhs_free_variables = HashSet::<String>::new();
                lhs.list_unknown_identifiers(extra_vars, env, &mut lhs_free_variables);
                let mut rhs_free_variables = HashSet::<String>::new();
                rhs.list_unknown_identifiers(extra_vars, env, &mut rhs_free_variables);
                if !lhs_free_variables.is_empty() {
                    return test_function_equality(lhs, rhs, lhs_free_variables, rhs_free_variables, op, false, precision_expr, extra_vars, env);
                } else if !rhs_free_variables.is_empty() {
                    return test_function_equality(rhs, lhs, rhs_free_variables, lhs_free_variables, op, true, precision_expr, extra_vars, env);
                }
            }
            // Otherwise, simply evaluate the binary operation.
            let Status{value: lhs_eval, mut warnings} = eval(lhs, extra_vars, env)?;
            // If the LHS is evaluated to zero and `op` is `*` or `&&`, we can skip evaluating the RHS.
            // Furthermore, we actually SHOULD skip it, since this enables us to use indicator functions smartly.
            if let Object::Real(x) = &lhs_eval && x.is_finite() && approx_eq(*x, 0.0) && (*op == BinaryOperation::Mul || *op == BinaryOperation::And) {
                Ok(Status{value: rhs.get_type(extra_vars, env).map(|t| t.zero())?, warnings})
            } else {
                try_operation(&lhs_eval, &eval(rhs, extra_vars, env)?.unpack_into(&mut warnings), op)
                .map(|value| Status{value, warnings})
            }
        }
        Expression::FoldedOperation(op, index_var, from, conditions, to, inner)
            => math::operations::folded_operations::folded_operation_helper(
                op,
                index_var,
                from,
                conditions,
                to,
                inner,
                |_varstack, _env| eval(inner, _varstack, _env),
                |_some_index_var_value, _varstack, _env| {
                    inner.get_type(
                        &VarStack::Frame { vars: &HashMap::from([(index_var, _some_index_var_value)]), parent: _varstack },
                        _env
                    )
                },
                extra_vars,
                env
            ),
        Expression::Function(function_name, given_arg_exprs) => {
            // Note this case can only occur when we actually have a function call, not an assignment.
            // We can be sure about this because the assignment operator is given the lowest priority level by the tokenizer
            // and the case `Expression::Assignment` in this function does not call itself recursively on the LHS
            // of an assignment operation.
            
            // If `function_name` is of the form with `___diff_num_f`, this isn't a function contained in `functions` but the request to numerically differentiate `f`.
            if let Some(real_function_name) = function_name.strip_prefix("___diff_num_") {
                // Ensure that `given_arg_exprs` is even. There is a special case where an uneven number is tolerated: if only a single argument
                // is provided, simply set the direction as 1.0 (default for 1d derivative).
                let mut tmp: Vec<Expression>;
                let arg_exprs = if given_arg_exprs.len() % 2 != 0 {
                    if given_arg_exprs.len() == 1 {
                        tmp = given_arg_exprs.clone();
                        tmp.push(Expression::Number(1.0));
                        &tmp
                    }
                    else {
                        return Err("___diff_num_{{...}} takes an even number of arguments.".to_string()); // See splitting of arguments below
                    }
                } else { given_arg_exprs };
                let rm = env.functions.remove(real_function_name);
                let res = match rm {
                    Some(FunctionRepr::Direct(f_ref, _)) => {
                        let Status{value: (point, direction), warnings} = Status::combine(
                            eval_mul_exprs(arg_exprs[0..arg_exprs.len()/2].iter(), extra_vars, env)?,
                            eval_mul_exprs(arg_exprs[arg_exprs.len()/2..arg_exprs.len()].iter(), extra_vars, env)?,
                            |lhs, rhs| Ok((lhs, rhs))
                        )?;
                        let mut mutable_version = |x: &[Object], y: &[Expression], z: Option<(&VarStack, &mut Env)>| f_ref(x, y, z);
                        math::differentiation::numerical_directional_derivative(&mut mutable_version, point, direction, extra_vars, env)
                        .map(|s| s.with_extra_warnings(warnings))
                    }
                    Some(FunctionRepr::ByExpression(ref f_varnames, ref f_expr)) => {
                        // This is rare, but if e.g. an integral should be differentiated, then we need this case
                        // (cf. `math::differentiation::analytic_partial_derivative`, case `Expression::Integral`).
                        let Status{value: (point, direction), warnings} = Status::combine(
                            eval_mul_exprs(arg_exprs[0..arg_exprs.len()/2].iter(), extra_vars, env)?,
                            eval_mul_exprs(arg_exprs[arg_exprs.len()/2..arg_exprs.len()].iter(), extra_vars, env)?,
                            |lhs, rhs| Ok((lhs, rhs))
                        )?;
                        #[allow(clippy::type_complexity)] 
                        let mut f_as_direct: Box<dyn for<'a, 'b, 'c, 'd> FnMut(&'a [Object], &'b [Expression], Option<(&'c VarStack, &'d mut Env)>) -> ExtResult> = Box::new(
                            |parsed_args, _, context| {
                                if parsed_args.len() != f_varnames.len() {
                                    Err(format!("Wrong number of arguments provided for function '{}' (expected {}, got {}).", real_function_name, f_varnames.len(), parsed_args.len()))
                                } else if let Some((_varstack, _env)) = context {
                                    eval(
                                        f_expr,
                                        &VarStack::Frame {
                                            vars: &f_varnames.iter().zip(parsed_args.iter()).collect(),
                                            parent: _varstack
                                        },
                                        _env
                                    )
                                } else {
                                    Err("[Unreachable] Function requires varstack and environment.".to_string())
                                }
                            }
                        );
                        math::differentiation::numerical_directional_derivative(&mut f_as_direct, point, direction, extra_vars, env)
                        .map(|s| s.with_extra_warnings(warnings))
                    }
                    None => Err(format!("No such function: {:?}", function_name))
                };
                if let Some(x) = rm {
                    env.functions.insert(real_function_name.to_string(), x);
                }
                res
            }

            // Check if `function_name` corresponds to a known `FunctionRepr::ByExpression(argnames, defining_expr)` in `env.functions`.
            // If so, we need to clone `argnames` and `defining_expr`:
            // Indeed, it would theoretically be possible that the final `eval` call in the subsequent block modifies the
            // at least one of `argnames` and `defining_expr`, e.g. `f(x) := [f(x), f(y) := x+y]`. Therefore, not cloning
            // would render this `eval` call impossible since we would need to reborrow `env` as mutable again.
            // Note: we can't just temporarily remove `function_name` from `env.functions` and later reinsert it since this would
            // make expressions like `exp(exp(0))` impossible.
            else if let Some((argnames, defining_expr)) = match env.functions.get(function_name) {
               Some(FunctionRepr::ByExpression(argnames, defining_expr)) => Some((argnames.clone(), defining_expr.clone())),
               _ => None 
            } {
                let Status{value: evaluated_args, warnings} = eval_mul_exprs(given_arg_exprs.iter(), extra_vars, env)?;
                if evaluated_args.len() != argnames.len() {
                    return Err(format!("Wrong number of arguments provided for function '{}' (expected {}, got {}).", function_name, argnames.len(), evaluated_args.len()));
                }
                let tmp_vars: HashMap<&String, &Object> = evaluated_args.iter().enumerate().map(|(i, x)| (&argnames[i], x)).collect();
                let new_stack = VarStack::Frame { vars: &tmp_vars, parent: extra_vars };
                eval(&defining_expr, &new_stack, env).map(|s| s.with_extra_warnings(warnings))
            }

            // Check if `function_name` corresponds to a known `FunctionRepr::Direct` in `env.functions`.
            // Then, proceed similarly as for `FunctionRepr::ByExpression` but take into consideration the function mask.
            else if let Some((f, m, n, b)) = match env.functions.get(function_name) {
               Some(FunctionRepr::Direct(f, (m, n, b))) => Some((&**f, *m, *n, *b)),
               _ => None 
            } {
                if given_arg_exprs.len() < m + n {
                    return Err(format!("Wrong number of arguments provided for function '{}' (expected at least {}).", function_name, m + n));
                }
                let mut evaluated_args_status = eval_mul_exprs(given_arg_exprs.iter().take(m), extra_vars, env)?;
                if b {
                    evaluated_args_status.merge(eval_mul_exprs(given_arg_exprs.iter().skip(m+n), extra_vars, env)?)
                }
                evaluated_args_status.try_map_flatten(
                    |evaluated_args| f(
                        &evaluated_args,
                        if b {&given_arg_exprs[m .. (m+n)]} else {&given_arg_exprs[m..]}, Some((extra_vars, env))
                    )
                )
            }

            else {Err(format!("No such function: {:?}", function_name))}
        }
        Expression::Assignment(lhs, rhs) => {
            eval_assignment(lhs, rhs, extra_vars, env)
        }
        Expression::PartialDerivative(wrt, expr) => {
            math::differentiation::analytic_partial_derivative(expr, wrt, extra_vars, env)
            .map(|s| s.map(|e| Object::LiteralExpression(e)))
        }
        Expression::DirectionalDerivative(vars, expr, point_exprs, direction_exprs) => {
            if point_exprs.len() != vars.len() || point_exprs.len() != direction_exprs.len() {
                return Err("Point and direction of directional derivative must have the same dimension.".to_string());
            }
            Status::combine_flatten(
                eval_mul_exprs(point_exprs.iter(), extra_vars, env)?,
                eval_mul_exprs(direction_exprs.iter(), extra_vars, env)?,
                |point, direction|
                math::differentiation::analytic_directional_derivative(vars, expr, &point, &direction, extra_vars, env)
            )
        }
        Expression::Integral(inner, a_expr, b_expr, wrt) => Status::combine_flatten(
            eval(a_expr, extra_vars, env)?.try_map(|o| o.expect_float())?,
            eval(b_expr, extra_vars, env)?.try_map(|o| o.expect_float())?,
            |a, b| math::integration::integrate(inner, a, b, wrt, extra_vars, env)
        ),
        Expression::IfElse(condition, iftrue, iffalse) => {
            eval(condition, extra_vars, env)?
            .try_map_flatten(|o| match o {
                Object::Real(1.0) => eval(iftrue, extra_vars, env),
                Object::Real(0.0) => eval(iffalse, extra_vars, env),
                x => Err(format!("Couldn't evaluate condition `{}` to 0 or 1; got {}.", &**condition, x))
            })
        }
    }
}

/// For every `x` in `expressions`, evaluates `x`, adds the warnings to `warnings` and the value to the returned `Vec`.
#[inline]
fn eval_mul_exprs<'a>(
    expressions: impl Iterator<Item=&'a Expression>,
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Status<Vec<Object>>, String> {
    Status::from_iter(
        expressions,
        |e| eval(e, extra_vars, env).map_err(|err| format!("Couldn't evaluate `{}`. Traceback: {}", e, err))
    )
}
/// Helper function, tailored to use case
fn eval_mul_exprs_to_f64(expressions: &[Expression], extra_vars: &VarStack, env: &mut Env) -> Result<Status<Vec<f64>>, String> {
    Status::from_iter(
        expressions.iter(),
        |e| eval(e, extra_vars, env)
            .map_err(|err| format!("Couldn't evaluate `{}`. Traceback: {}", e, err))
            ?
            .try_map(|o| o.expect_float())
    )
}

/// Tests whether two expressions `lhs` and `rhs` are equal by plugging in a range of arguments (cf. implementation for details).
/// 
/// * `lhs_free_variables` - Identifiers in `lhs` for which values should be inserted.
/// * `rhs_free_variables` - Identifiers in `rhs` for which values should be inserted.
/// * `op` - The comparison operator to be used. It is assumed to be a comparison and not another binary operation.
/// * `mirror` - Whether the comparison operator `op` should subsequently be mirrored (e.g. `>` becomes `<`) or not.
/// * `precision_expr` - If `Some(e)`, tries to evaluate `e` to an integer and use this as precision. Otherwise, uses `DEFAULT_TESTEQ_REPETITIONS`.
fn test_function_equality(
    lhs: &Expression,
    rhs: &Expression,
    mut lhs_free_variables: HashSet<String>,
    rhs_free_variables: HashSet<String>,
    op: &BinaryOperation,
    mirror: bool,
    precision_expr: &Option<Box<Expression>>,
    extra_vars: &VarStack,
    env: &mut Env
) -> ExtResult {
    let mut warnings = Vec::<String>::new(); // We will accumulate warnings in this list
    let rhs_only_needs_single_eval = rhs_free_variables.is_empty();
    lhs_free_variables.extend(rhs_free_variables.into_iter());

    // Determine number of iterations
    let n = if let Some(p) = precision_expr {
        eval(p, extra_vars, env)
        .and_then(|s| s.try_map(
            |o| o.expect_int::<i64>().map(|i| i.max(0) as usize)
        ))
        ?
        .unpack_into(&mut warnings)
    } else {
        DEFAULT_TESTEQ_REPETITIONS
    };

    let mut rhs_eval = Object::Success; // Placeholder
    // If `rhs` doesn't contain any free variables
    // (<=> the second `list_unknown_identifiers` call in `eval` right before calling `test_function_equality` actually modified the expression),
    // it suffices to evaluate `rhs` once. Then, evaluating every time would be inefficient, especially if many values will be tested.
    // Therefore, it makes sense to check whether this is the case beforehand, and if so, simply evaluate once and save the value for later.
    if rhs_only_needs_single_eval {
        rhs_eval = eval(&rhs, extra_vars, env)?.unpack_into(&mut warnings);
    }

    // Note that the size of the following vector is 6n, so if lhs_free_variables is large, the number of test values can quickly blow up.
    // Generally speaking, this is necessary though, since checking that multivariate functions are equal logically requires us to check
    // various possible combinations of input variables.
    let linspaces: Vec<Object> = [
        linspace_as_objects(0.0, 1.0, n),
        linspace_as_objects(1.0, 100.0, n),
        (101..=100+n).map(|x| Object::Real(x as f64)).collect::<Vec<Object>>(),
        linspace_as_objects(0.0, -1.0, n),
        linspace_as_objects(-1.0, -100.0, n),
        (-100-(n as isize) .. -100).map(|x| Object::Real(x as f64)).collect::<Vec<Object>>()
    ]
    .into_iter()
    .flat_map(|v| v.into_iter())
    .collect();

    for test_values in (0..lhs_free_variables.len()).map(|_| linspaces.iter()).multi_cartesian_product() {
        let tmp_vars: HashMap<&String, &Object> = lhs_free_variables.iter().enumerate().map(|(i, ident)| (ident, test_values[i])).collect();
        let new_stack = VarStack::Frame { vars: &tmp_vars, parent: extra_vars };
        // In order to avoid massive storage usage, we only allow for a fixed number of warnings emitted.
        let lhs_eval = eval(&lhs, &new_stack, env)
            .map_err(|e| format!("Couldn't evaluate `{}` with environment {:?}. Traceback: {}", lhs, tmp_vars, e)) // Add information to the error message
            ?
            .unpack_into_with_cap(&mut warnings, 8);
        if !rhs_only_needs_single_eval {
            rhs_eval = eval(&rhs, &new_stack, env)
                .map_err(|e| format!("Couldn't evaluate `{}` with environment {:?}. Traceback: {}", rhs, tmp_vars, e))
                ?
                .unpack_into_with_cap(&mut warnings, 8);
        }
        // If the objects' comparison yields `false`, return that. If the objects aren't comparable, return the appropriate error. Otherwise, continue.
        match if mirror {try_operation(&rhs_eval, &lhs_eval, op)} else {try_operation(&lhs_eval, &rhs_eval, op)} {
            Ok(Object::Real(0.0)) => { return Ok(Status{value: Object::Real(0.0), warnings}); }
            Err(_) => { return Err(format!("Couldn't compare `{}` and `{}` (arising from environment {:?}).", lhs_eval, rhs_eval, env.constants)); }
            _ => {}
        }
    }
    Ok(Status{ value: Object::Real(1.0), warnings }) // If nothing previous returned, then the expressions fulfill the comparison.
}

fn eval_assignment(
    lhs: &Expression,
    rhs: &Expression,
    extra_vars: &VarStack,
    env: &mut Env
) -> ExtResult {
    // Note that names starting with "___" are forbidden (prefix "___tmp_" reserved for temporary variables, prefix "___diff_" for the derivative of a function with direct representation).
    /// Helper function. We need this because multiple syntax structures lead to a function definition:
    /// - `Expression::Function(function_name, args)`
    /// - `Expression::BinaryOperation(Identifier(function_name), BinaryOperation::Mul, Identifier(arg))`
    /// - `Expression::BinaryOperation(Identifier(function_name), BinaryOperation::Mul, Vector(args))`
    fn define_function(
        function_name: &String,
        unparsed_args: std::slice::Iter<'_, Expression>,
        rhs: &Expression,
        extra_vars: &VarStack,
        env: &mut Env
    ) -> ExtResult {
        if function_name.starts_with("___") { Err("Names starting with \"___\" are forbidden".to_string()) }
        else if function_name == "D" || function_name.starts_with("D_") { Err("The name \"D\" and identifiers starting with \"D_\" are reserved for the total derivative.".to_string()) }
        else if KEYWORDS.contains(&function_name.as_str()) { Err(format!("The identifier \"{function_name}\" is a keyword.")) }
        else {
            // First, check that all declared arguments on the LHS are in fact just identifiers.
            let mut argnames = unparsed_args.into_iter()
                .map(|lh_arg|
                    if let Expression::Identifier(x) = lh_arg {Ok(x.clone())}
                    else {Err("Parameters in LHS of function definition must be identifiers.".to_string())}
                )
                .collect::<Result<Vec<_>, _>>()?;
            // Next, parse the RHS as explained in the documentation of `parse_function_definition`.
            let Status{value: expr, mut warnings} = parse_function_definition(rhs, &argnames, extra_vars, env)?;
            // The argument names have to be prefixed too
            argnames = argnames.into_iter().map(|x| format!("___tmp_{}", x)).collect();
            env.functions.insert(function_name.clone(), FunctionRepr::ByExpression(
                argnames,
                expr
            ));
            // The .clone() above is no problem since function definitions are rare (in the sense that performance doesn't matter for this).
            // Next, if there was already a function `__diff_{function_name}` present in `functions` (cf. `analytic_derivative`),
            // then it is now outdated, so we remove it.
            env.functions.remove(&format!("___diff_num_{}", function_name));
            // Lastly, if `function_name` was already a constant, then we should remove it to avoid ambiguity.
            // We emit a warning if this happens.
            if let Some(old_val) = env.constants.remove(function_name) {
                warnings.push(format!("The constant `{}` with value {} was removed.", function_name, old_val));
                Ok(Status{value: Object::Success, warnings})
            } else {
                Ok(Status::ok(Object::Success))
            }
        }
    }

    fn define_constant(
        constant_name: &String,
        value: Object,
        env: &mut Env
    ) -> ExtResult {
        if constant_name.starts_with("___") {
            Err("Names starting with \"___\" are forbidden".to_string())
        } else if constant_name == "D" || constant_name.starts_with("D_") {
            Err("The name \"D\" and identifiers starting with \"D_\" are reserved for the total derivative.".to_string())
        }
        else if KEYWORDS.contains(&constant_name.as_str()) {
            Err(format!("The identifier \"{constant_name}\" is a keyword."))
        } else {
            // The '.clone()' in below line is due to the fact that we want to save the value on one hand (within 'constants')
            // but also return it (e.g. the expression "x := 5" should not only define x as 5 but also return the value 5 so that
            // one can write "... * (x := ...)" to save intermediate results).
            env.constants.insert(constant_name.clone(), value.clone());
            // Lastly, if `constant_name` was already a function, then we should remove it to avoid ambiguity.
            // We emit a warning if this happens.
            if let Some(old_val) = env.functions.remove(constant_name) {
                Ok(Status{value, warnings: vec![format!("The function `{}` given by {:?} was removed.", constant_name, old_val)]})
            } else {
                Ok(Status::ok(value))
            }
        }
    }

    match lhs {
        Expression::Identifier(ident) => {
            let Status{value, warnings} = eval(rhs, extra_vars, env)?;
            define_constant(ident, value, env).map(|s| s.with_extra_warnings(warnings))
        }
        Expression::BinaryOperation(x, BinaryOperation::Mul, y)
        if let Expression::Identifier(function_name) = &**x => {
            match &**y {
                Expression::Identifier(_)
                    => define_function(function_name, std::slice::from_ref(&**y).iter(), rhs, extra_vars, env),
                Expression::Vector(args) | Expression::Tuple(args)
                    => define_function(function_name, args.iter(), rhs, extra_vars, env),
                _ => Err(format!("Invalid LHS of assignment expression: {}", lhs))
            }
        }
        Expression::Function(function_name, unparsed_args)
            => define_function(function_name, unparsed_args.iter(), rhs, extra_vars, env),
        Expression::Tuple(lhs_exprs) => {
            let Status{value: obj, warnings} = eval(rhs, extra_vars, env)?;
            match obj {
                Object::Tuple(rhs_values) => {
                    if lhs_exprs.len() != rhs_values.len() {
                        return Err(format!("Tuples on both sides of assignment operator must be of equal length (got {}, {}).", lhs_exprs.len(), rhs_values.len()))
                    }
                    let mut rhs_values_iter = rhs_values.into_iter();
                    Status::from_iter(
                        lhs_exprs.iter(),
                        |lhs_expr| if let Expression::Identifier(ident) = lhs_expr {
                            define_constant(ident, rhs_values_iter.next().unwrap(), env)
                        } else {
                            Err("All LHS entries must be identifiers.".to_string())
                        }
                    )
                    .map(|s|
                        s
                        .map(|v| Object::Tuple(v))
                        .with_extra_warnings(warnings)
                    )
                }
                other => Err(format!("RHS couldn't be evaluated to a tuple (result: {}).", other))
            }
        }
        _ => Err(format!("Invalid LHS of assignment expression: `{}`.", lhs))
    }
}