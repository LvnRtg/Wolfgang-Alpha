//! Responsible for evaluating an `Expression` to an `Object`.

use std::collections::HashMap;
use std::collections::HashSet;
use statrs::function::gamma;
use itertools::Itertools;

use crate::math;
use crate::math::utils::{approx_eq, linspace_as_objects};
use crate::math::objects::try_operation;
use crate::math::{BinaryOperation, Env, Expression, FunctionRepr, Object, UnaryOperation, VarStack}; // Common types that will be used several times


const DEFAULT_TESTEQ_REPETITIONS: usize = 20;


/// When an function definition is encountered, the expression on the RHS is processed in a special way.
/// Generally, it has to be cloned (cleanest way to work with the 'eval' function below), which is the main action this function performs.
/// 
/// Parsing the expression recursively, every identifier that is NOT declared as an argument of the function is replaced
/// by the constant it represents in the current environment. Identifiers can be declared as arguments even if they exist in the environment;
/// the environmental value will then be ignored. Moreover, every identifier that is declared as an argument is prefixed with three underscores
/// (this will be needed for evaluation). For example, if 'constants = {"x": 1, "y": 2}', the RHS of the literal expression
/// "f(y, z) := x + 3*y + z" will become "1 + 3*___tmp_y + ___tmp_z".
/// 
/// I have decided that if the definition depends on another function (say, "f(x, y) = g(x) + y"), the other function shall
/// not be replaced by its literal expression. It makes sense to me to capture the current values of free variables because
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
) -> Result<Expression, String> {
    Ok(match expr {
        Expression::None => Expression::None,
        Expression::Identifier(x) => {
            if argument_names.contains(x) {
                Expression::Identifier(format!("___tmp_{}", x))
            } else if let Some(y) = env.constants.get(x) {
                y.to_expression()
            } else if let Some(y) = extra_vars.lookup(x) {
                y.to_expression()
            } else {
                Expression::Identifier(x.clone())
            }
        },
        Expression::Number(x) => Expression::Number(*x),
        Expression::Tuple(x) => Expression::Tuple(
            x.iter()
            .map(|x| parse_function_definition(x, argument_names, extra_vars, env))
            .collect::<Result<Vec<_>, _>>()
            ?
        ),
        Expression::Vector(x) => Expression::Vector(
            x.iter()
            .map(|x| parse_function_definition(x, argument_names, extra_vars, env))
            .collect::<Result<Vec<_>, _>>()
            ?
        ),
        Expression::Matrix(m, n, x) => Expression::Matrix(
            *m,
            *n,
            x.iter()
            .map(|x| parse_function_definition(x, argument_names, extra_vars, env))
            .collect::<Result<Vec<_>, _>>()
            ?
        ),
        Expression::UnaryOperation(op, rhs) => Expression::UnaryOperation(
            op.clone(),
            Box::new(parse_function_definition(rhs, argument_names, extra_vars, env)?)
        ),
        Expression::BinaryOperation(lhs, op, rhs) => Expression::BinaryOperation(
            Box::new(parse_function_definition(lhs, argument_names, extra_vars, env)?),
            op.clone(),
            Box::new(parse_function_definition(rhs, argument_names, extra_vars, env)?)
        ),
        Expression::FoldedOperation(op, varname, from, conditions, to, inner) => Expression::FoldedOperation(
            op.clone(),
            varname.clone(),
            Box::new(parse_function_definition(from, argument_names, extra_vars, env)?),
            conditions.iter().map(|x| parse_function_definition(x, argument_names, extra_vars, env)).collect::<Result<Vec<_>, _>>()?,
            Box::new(parse_function_definition(to, argument_names, extra_vars, env)?),
            // Notice that if `varname` is simultaneously an argument of the function, it shouldn't be replace by ___tmp_...
            // within `inner`. For example, `g(x) := \sum_{x=1}^2 x` should be equivalent to `g(x) := \sum_{i=1}^2 i`.
            if let Some(i) = argument_names.iter().position(|n| n == varname) {
                if argument_names.len() > 2 {
                    // Still replace all identifiers in `inner` except the one at index `i`. Note that cloning is fine here since the list of argument names
                    // should be very small (having more than two would already be very rare).
                    Box::new(parse_function_definition(
                        inner,
                        &argument_names.iter().enumerate().filter(|&(idx, _)| idx != i).map(|(_, x)| x.clone()).collect(),
                        extra_vars,
                        env
                    )?)
                } else {
                    inner.clone()
                }
            } else {
                Box::new(parse_function_definition(inner, argument_names, extra_vars, env)?)
            }
        ),
        Expression::Function(function_name, args) => Expression::Function(
            function_name.clone(),
            args.iter().map(|x| parse_function_definition(x, argument_names, extra_vars, env)).collect::<Result<Vec<_>, _>>()?
        ),
        Expression::Assignment(lhs, rhs) => Expression::Assignment(
            Box::new(parse_function_definition(lhs, argument_names, extra_vars, env)?),
            Box::new(parse_function_definition(rhs, argument_names, extra_vars, env)?)
        ),
        Expression::PartialDerivative(wrt, expr)
            => parse_function_definition(&math::differentiation::analytic_partial_derivative(expr, wrt, extra_vars, env)?, argument_names, extra_vars, env)?,
        Expression::DirectionalDerivative(vars, expr, point, direction) => Expression::DirectionalDerivative(
            vars.clone(),
            Box::new(parse_function_definition(expr, argument_names, extra_vars, env)?),
            point.iter().map(|x| parse_function_definition(x, argument_names, extra_vars, env)).collect::<Result<Vec<_>, _>>()?,
            direction.iter().map(|x| parse_function_definition(x, argument_names, extra_vars, env)).collect::<Result<Vec<_>, _>>()?
        ),
        Expression::Integral(inner, a, b, x) => Expression::Integral(
            Box::new(parse_function_definition(inner, argument_names, extra_vars, env)?),
            Box::new(parse_function_definition(a, argument_names, extra_vars, env)?),
            Box::new(parse_function_definition(b, argument_names, extra_vars, env)?),
            x.clone()
        ),
        Expression::IfElse(x, y, z) => Expression::IfElse(
            Box::new(parse_function_definition(x, argument_names, extra_vars, env)?),
            Box::new(parse_function_definition(y, argument_names, extra_vars, env)?),
            Box::new(parse_function_definition(z, argument_names, extra_vars, env)?),
        )
    })
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
) -> Result<Object, String> {
    match expr {
        Expression::None => Err("Received empty expression.".to_string()),
        Expression::Identifier(ident) => {
            // First, iterate `extra_vars` in reverse order and search for `ident`.
            if let Some(x) = extra_vars.lookup(ident) {
                Ok(x.clone())
            }
            // If nothing is found, look in `constants`.
            else if let Some(x) = env.constants.get(ident) {
                Ok(x.clone())
                // We only call 'clone' for every time a variable from 'constants' is used, which can only happen so often
                // since the user still has to enter at least one character per time it is used. Therefore,
                // even if these are large matrices, it is a totally acceptable runtime.
            }
            // If still, nothing is found, this is an error.
            else {
                Err(format!("Unknown identifier: {:?}", ident))
            }
        },
        Expression::Number(x) => Ok(Object::Float(*x)),
        Expression::Tuple(entries) => {
            // As mentioned in the docs, we capture the environment for tuple evaluation.
            // Two approaches:
            // 1. Capture `env` at the start, clone it for every `x` in `entries`, call `eval(x)` with is, merge it into `env` after `eval(x)`.
            //    Slightly more overhead (+1 clone) but O(1) space.
            // 2. No capture at the start, proceed as above but only merge with `env` at the end of ALL `eval` calls.
            //    Less overhead but O(n) space.
            // => Choose 1.
            let captured_env = env.clone();
            let mut results = Vec::<Object>::with_capacity(entries.len());
            for x in entries.iter() {
                let mut tmp_env = captured_env.clone();
                results.push(eval(x, extra_vars, &mut tmp_env).map_err(|e| format!("Couldn't evaluate entry {}. Traceback: {}", x, e))?);
                // TODO: merge tmp_env into env

            }
            Ok(Object::Tuple(results))
        },
        Expression::Vector(entries) => {
            Ok(Object::Vector(math::Vector{values: entries.iter().map(
                |x| match eval(x, extra_vars, env) {
                    Ok(Object::Float(entry)) => Ok(entry),
                    Ok(_) => Err(format!("Entry {} is not a float.", x)),
                    Err(e) => Err(format!("Couldn't evaluate entry {}. Traceback: {}", x, e))
                }
            ).collect::<Result<Vec<_>, _>>()?}))
        },
        Expression::Matrix(m, n, entries) => {
            Ok(Object::Matrix(math::Matrix::from(*m, *n, entries.iter().map(
                |x| match eval(x, extra_vars, env) {
                    Ok(Object::Float(entry)) => Ok(entry),
                    Ok(_) => Err(format!("Entry {} is not a float.", x)),
                    Err(e) => Err(format!("Couldn't evaluate entry {}. Traceback: {}", x, e))
                }
            ).collect::<Result<Vec<_>, _>>()?)))
        },
        Expression::UnaryOperation(op, rhs) => {
            match op {
                UnaryOperation::Neg => {
                    -&eval(rhs, extra_vars, env)?
                }
                UnaryOperation::Not => {
                    !&eval(rhs, extra_vars, env)?
                }
                UnaryOperation::Factorial => {
                    match eval(rhs, extra_vars, env)? {
                        Object::Success => Ok(Object::Success),
                        Object::Float(x) => Ok(Object::Float({
                            let r = x.round();
                            if approx_eq(x, r) && r >= 0.0 { // Avoid calling the gamma function if unnecessary
                                if r <= 1.0 {
                                    1.0
                                } else {
                                    let n = r as u64;
                                    (1..=n).try_fold(1, u64::checked_mul).ok_or(format!("Overflow occured while computing {n}!"))? as f64
                                }
                            } else {
                                gamma::gamma(x+1.0)
                            }
                        })),
                        Object::LiteralExpression(e) => Ok(Object::LiteralExpression(Expression::UnaryOperation(UnaryOperation::Factorial, Box::new(e)))),
                        other => Err(format!("Operation 'Factorial' not valid for operand {other}.")),
                    }
                }
                UnaryOperation::Abs => {
                    match eval(rhs, extra_vars, env)? {
                        Object::Success => Ok(Object::Success),
                        Object::Float(x) => Ok(Object::Float(x.abs())),
                        Object::LiteralExpression(e) => Ok(Object::LiteralExpression(Expression::UnaryOperation(UnaryOperation::Abs, Box::new(e)))),
                        other => Err(format!("Operation 'Abs' not valid for operand {other}.")),
                    }
                }
                UnaryOperation::Norm(opt) => {
                    match eval(rhs, extra_vars, env)? {
                        Object::Success => Ok(Object::Success),
                        Object::Float(x) => Ok(Object::Float(x.abs())),
                        Object::Vector(x) => Ok(Object::Float(
                            x.norm(&math::matrices_and_vectors::VectorNorm::from_expr(opt, extra_vars, env)?)
                        )),
                        Object::Matrix(x) => Ok(Object::Float(
                            x.norm(&math::matrices_and_vectors::MatrixNorm::from_expr(opt, extra_vars, env)?)?
                        )),
                        Object::LiteralExpression(e) => Ok(Object::LiteralExpression(Expression::UnaryOperation(UnaryOperation::Abs, Box::new(e)))),
                        other => Err(format!("Operation 'Norm' not valid for operand {other}.")),
                    }
                }
            }
        },
        Expression::BinaryOperation(lhs, op, rhs) => {
            // Check if the operation is a comparison and at least one of `lhs`, `rhs` is a function (which we'll call `this`; we'll call the remaining one `other`).
            // Here, being a function means having unknown identifiers within.
            if let BinaryOperation::Comp(_, param) = op {
                let mut lhs_free_variables = HashSet::<String>::new();
                lhs.list_unknown_identifiers(extra_vars, env, &mut lhs_free_variables);
                let mut rhs_free_variables = HashSet::<String>::new();
                rhs.list_unknown_identifiers(extra_vars, env, &mut rhs_free_variables);
                // The following bool `mirror` expresses whether the comparison operator `op` should subsequently be mirrored or not.
                if let Some((this, other, param, mirror)) = if !lhs_free_variables.is_empty() {
                    Some((*lhs.clone(), *rhs.clone(), param, false))
                } else if !rhs_free_variables.is_empty() {
                    Some((*rhs.clone(), *lhs.clone(), param, true))
                } else {None} {
                    let other_only_needs_single_eval = rhs_free_variables.is_empty();
                    lhs_free_variables.extend(rhs_free_variables);
                    let mut other_eval = Object::Success; // Placeholder
                    // If `other` doesn't contain any free variables (<=> the second `list_unknown_identifiers` call above actually modified the expression),
                    // it suffices to evaluate `other` once.
                    // Then, evaluating every time would be inefficient, especially if many values will be tested.
                    // Therefore, it makes sense to check whether this is the case beforehand, and if so, simply evaluate once and save the value for later.
                    if other_only_needs_single_eval {
                        other_eval = eval(&other, extra_vars, env)?;
                    }

                    // If a number of repetitions is given as `param` under the form of an expression, evaluate it and use it. Otherwise, use `DEFAULT_TESTEQ_REPETITIONS`
                    let n = param.as_ref()
                        .map(|p| match eval(p, extra_vars, env) {
                            Ok(Object::Float(x)) => Ok(x.round() as usize),
                            Err(e) => Err(format!("Couldn't resolve number of repetitions `{}`. Traceback: {}", p, e)),
                            _ => Err(format!("Couldn't resolve number of repetitions `{}` to float.", p))
                        })
                        .unwrap_or(Ok(DEFAULT_TESTEQ_REPETITIONS))
                        ?;

                    // Note that the size of the following vector is 6n, so if lhs_free_variables is large, the number of test values can quickly blow up.
                    // Generally speaking, this is necessary though, since checking that multivariate functions are equal logically requires us to check
                    // various possible combinations of input variables.
                    let linspaces: Vec<Object> = [
                        linspace_as_objects(0.0, 1.0, n),
                        linspace_as_objects(1.0, 100.0, n),
                        (101..=100+n).map(|x| Object::Float(x as f64)).collect::<Vec<Object>>(),
                        linspace_as_objects(0.0, -1.0, n),
                        linspace_as_objects(-1.0, -100.0, n),
                        (-100-(n as isize) .. -100).map(|x| Object::Float(x as f64)).collect::<Vec<Object>>()
                    ].iter().flat_map(|v| v.iter()).cloned().collect();

                    for test_values in (0..lhs_free_variables.len()).map(|_| linspaces.iter()).multi_cartesian_product() {
                        let tmp_vars: HashMap<&String, &Object> = lhs_free_variables.iter().enumerate().map(|(i, ident)| (ident, test_values[i])).collect();
                        let new_stack = VarStack::Frame { vars: &tmp_vars, parent: extra_vars };
                        let first_eval = eval(&this, &new_stack, env)
                            .map_err(|e| format!("Couldn't evaluate `{}` with environment {:?}. Traceback: {}", this, tmp_vars, e)) // Add information to the error message
                            ?;
                        if !other_only_needs_single_eval {
                            other_eval = eval(&other, &new_stack, env)
                                .map_err(|e| format!("Couldn't evaluate `{}` with environment {:?}. Traceback: {}", this, tmp_vars, e))
                                ?;
                        }
                        // If the objects' comparison yields `false`, return that. If the objects aren't comparable, return the appropriate error. Otherwise, continue.
                        match if mirror {try_operation(&other_eval, &first_eval, op)} else {try_operation(&first_eval, &other_eval, op)} {
                            Ok(Object::Float(0.0)) => { return Ok(Object::Float(0.0)); }
                            Err(_) => { return Err(format!("Couldn't compare `{}` and `{}` (arising from environment {:?}).", first_eval, other_eval, env.constants)); }
                            _ => {}
                        }
                    }
                    return Ok(Object::Float(1.0)); // If nothing previous returned, then the expressions fulfill the comparison.
                }
            }
            
            // Otherwise, simply evaluate the binary operation.
            let lhs_eval = eval(lhs, extra_vars, env)?;
            // If the LHS is evaluated to zero and `op` is a multiplication, we can skip evaluating the RHS.
            // Furthermore, we actually SHOULD skip it, since this enables us to use indicator functions smartly.
            if let Object::Float(x) = &lhs_eval && approx_eq(*x, 0.0) && *op == BinaryOperation::Mul {
                return Ok(Object::Float(0.0)); // TODO: return the correct type when `Expression.type` is available
            }
            try_operation(&lhs_eval, &eval(rhs, extra_vars, env)?, op)
        },
        Expression::FoldedOperation(op, varname, from, conditions, to, inner) => {
            let mut i = eval(from, extra_vars, env)?.expect_int()?;
            // If i is more than `to` rightaway, return the default value for a folded operator over an empty range.
            if i > eval(to, &VarStack::Frame { vars: &HashMap::from([(varname, &Object::Float(i))]), parent: extra_vars }, env)?.expect_float()? {
                // TODO check type using extra function `checktype(expr)` and return 0 of the appropriate space.
                return Ok(op.if_empty());
            }
            let mut res = op.if_empty(); // TODO also change type here
            let binop = op.underlying_binop();
            'outer: while i <= eval(to, &VarStack::Frame { vars: &HashMap::from([(varname, &Object::Float(i))]), parent: extra_vars }, env)?.expect_float()? {
                // Check if all conditions are met. If not, skip this `i`.
                for cond in conditions {
                    match eval(cond, &VarStack::Frame { vars: &HashMap::from([(varname, &Object::Float(i))]), parent: extra_vars }, env)? {
                        Object::Float(1.0) => {} // Condition met; ignore
                        Object::Float(0.0) => { // Condition not met; skip `i`
                            i += 1.0;
                            continue 'outer;
                        }
                        other => return Err(format!("Expected 1 or 0 when evaluating condition, got {:?}.", other))
                    }
                }
                // At this point, all conditions are met.
                let next_term = eval(inner, &VarStack::Frame { vars: &HashMap::from([(varname, &Object::Float(i))]), parent: extra_vars }, env)?;
                res = try_operation(&res, &next_term, &binop)?;
                i += 1.0;
            }
            Ok(res)
        }
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
                    Some(FunctionRepr::Direct(f_ref)) => {
                        let (point, direction) = parse_diff_num_args(arg_exprs, extra_vars, env)?;
                        let mut mutable_version = |args: &[Object]| f_ref(args);
                        math::differentiation::numerical_directional_derivative(&mut mutable_version, point, direction)
                    }
                    Some(FunctionRepr::ByExpression(ref f_varnames, ref f_expr)) => {
                        // This is rare, but if e.g. an integral should be differentiated, then we need this case
                        // (cf. `math::differentiation::analytic_partial_derivative`, case `Expression::Integral`).
                        let (point, direction) = parse_diff_num_args(arg_exprs, extra_vars, env)?;
                        // This following closure is also the reason why we need `parse_diff_num_args` in a separate function.
                        #[allow(clippy::type_complexity)] 
                        let mut f_as_direct: Box<dyn for<'a> FnMut(&'a [Object]) -> Result<Object, String>> = Box::new(|args: &[Object]| {
                            if args.len() != f_varnames.len() {
                                Err(format!("Wrong number of arguments for {} (expected {}, got {}).", real_function_name, f_varnames.len(), args.len()))
                            } else {
                                eval(
                                    f_expr,
                                    &VarStack::Frame {
                                        vars: &(0..f_varnames.len()).map(|i| (&f_varnames[i], &args[i])).collect(),
                                        parent: extra_vars
                                    },
                                    env
                                )
                            }
                        });
                        math::differentiation::numerical_directional_derivative(&mut f_as_direct, point, direction)
                    }
                    None => Err(format!("No such function: {:?}", function_name))
                };
                if let Some(x) = rm {
                    env.functions.insert(real_function_name.to_string(), x);
                }
                res
            }

            // We're doing a little trick which is to remove the corresponding function from `functions` and reinserting it at the end.
            // This is necessary since `functions` can't be borrowed as mutable and immutable twice at the same time (caused by recursive call to `eval`).
            // By transfer of ownership, this is a very cheap operation compared to cloning a `FunctionRepr` because the latter's
            // defining expression (if present) can be highly nested.
            else if let Some(func) = env.functions.remove(function_name) {
                let ret_value = eval_function(function_name, &func, given_arg_exprs, extra_vars, env);
                env.functions.insert(function_name.clone(), func); // Reinsert the removed function
                ret_value
            }
            else {Err(format!("No such function: {:?}", function_name))}
        },
        Expression::Assignment(lhs, rhs) => {
            eval_assignment(lhs, rhs, extra_vars, env)
        }
        Expression::PartialDerivative(wrt, expr) => {
            math::differentiation::analytic_partial_derivative(expr, wrt, extra_vars, env).map(Object::LiteralExpression)
        }
        Expression::DirectionalDerivative(vars, expr, point_exprs, direction_exprs) => {
            if point_exprs.len() != direction_exprs.len() {
                return Err("Point and direction of directional derivative must have the same dimension.".to_string());
            }
            let point = point_exprs.iter()
                .map(|p| eval(p, extra_vars, env))
                .collect::<Result<Vec<_>, _>>()?;
            let direction = direction_exprs.iter()
                .map(|p| eval(p, extra_vars, env))
                .collect::<Result<Vec<_>, _>>()?;
            math::differentiation::analytic_directional_derivative(vars, expr, &point, &direction, extra_vars, env)
        }
        Expression::Integral(inner, a_expr, b_expr, wrt) => {
            let a = eval(a_expr, extra_vars, env)?.expect_float()?;
            let b = eval(b_expr, extra_vars, env)?.expect_float()?;
            math::integration::integrate(inner, a, b, wrt, extra_vars, env)
        }
        Expression::IfElse(condition, iftrue, iffalse) => {
            match eval(condition, extra_vars, env) {
                Ok(Object::Float(1.0)) => eval(iftrue, extra_vars, env),
                Ok(Object::Float(0.0)) => eval(iffalse, extra_vars, env),
                Ok(x) => Err(format!("Couldn't evaluate condition {} to 0 or 1; got {x}", &**condition)),
                other => other
            }
        }
    }
}

fn eval_function(
    function_name: &String,
    func: &FunctionRepr,
    given_arg_exprs: &[Expression],
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Object, String> {
    match func {
        FunctionRepr::ByExpression(argnames, defining_expr) => {
            if given_arg_exprs.len() != argnames.len() {
                return Err(format!("Wrong number of arguments provided for function '{}' (expected {}, got {}).", function_name, argnames.len(), given_arg_exprs.len()));
            }
            // Put all temporary variables (arguments) into a new hashmap and add it to `extra_vars`.
            let tmp_var_evals = given_arg_exprs.iter().enumerate().map(
                |(i, given_arg_expr)| {
                    eval(given_arg_expr, extra_vars, env)
                    .map_err(|e| format!("Couldn't resolve argument {} := {}. Traceback: {}", argnames[i], given_arg_expr, e))
                }
            ).collect::<Result<Vec<_>, _>>()?;
            let tmp_vars: HashMap<&String, &Object> = tmp_var_evals.iter().enumerate().map(|(i, x)| (&argnames[i], x)).collect();
            let new_stack = VarStack::Frame { vars: &tmp_vars, parent: extra_vars };
            eval(defining_expr, &new_stack, env)
        }
        FunctionRepr::Direct(f) => {
            f(&given_arg_exprs.iter()
                .map(|arg_expr| eval(arg_expr, extra_vars, env))
                .collect::<Result<Vec<_>, _>>()?)
        }
    }
}

/// Takes a vector of `Expression`s the format `point <concat> direction` and decomposes it into point and direction.
/// Also evaluates each expression.
/// 
/// Note: this needs to be a separate function to avoid passing `env` to `eval_diff_num`. This would create a conflict
/// because `env` is also captured by a closure when numerically differentiating a function defined by an expression.
fn parse_diff_num_args(arg_exprs: &[Expression], extra_vars: &VarStack, env: &mut Env) -> Result<(Vec<Object>, Vec<Object>), String> {
    // The given arguments should then have the format `point <concat> direction`, so we have to split the arguments
    // into two parts (splitting in the middle of the array which we ensured has even size).
    let point = (0..arg_exprs.len()/2)
        .map(|i| eval(&arg_exprs[i], extra_vars, env))
        .collect::<Result<Vec<_>, _>>()?;
    let direction = (arg_exprs.len()/2..arg_exprs.len())
        .map(|i| eval(&arg_exprs[i], extra_vars, env))
        .collect::<Result<Vec<_>, _>>()?;
    Ok((point, direction))
}

fn eval_assignment(
    lhs: &Expression,
    rhs: &Expression,
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Object, String> {
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
    ) -> Result<Object, String> {
        if function_name.starts_with("___") { Err("Names starting with \"___\" are forbidden".to_string()) }
        else if function_name == "D" || function_name.starts_with("D_") { Err("The name \"D\" and identifiers starting with \"D_\" are reserved for the total derivative.".to_string()) }
        else {
            // First, check that all declared arguments on the LHS are in fact just identifiers.
            let mut argnames = unparsed_args.into_iter()
                .map(|lh_arg|
                    if let Expression::Identifier(x) = lh_arg {Ok(x.clone())}
                    else {Err("Parameters in LHS of function definition must be identifiers.".to_string())}
                )
                .collect::<Result<Vec<_>, _>>()?;
            // Next, parse the RHS as explained in the documentation of `parse_function_definition`.
            let expr = parse_function_definition(rhs, &argnames, extra_vars, env)?;
            // The argument names have to be prefixed too
            argnames = argnames.into_iter().map(|x| format!("___tmp_{}", x)).collect();
            env.functions.insert(function_name.clone(), FunctionRepr::ByExpression(
                argnames,
                expr
            ));
            // The .clone() above is no problem since function definitions are rare (in the sense that performance doesn't matter for this).
            // Lastly, if there was already a function `__diff_{function_name}` present in `functions` (cf. `analytic_derivative`).
            // If so, it is now outdated, so remove it.
            env.functions.remove(&format!("___diff_num_{}", function_name));
            Ok(Object::Success)
        }
    }

    fn define_constant(
        constant_name: &String,
        value: Object,
        env: &mut Env
    ) -> Result<Object, String> {
        if constant_name.starts_with("___") {
            Err("Names starting with \"___\" are forbidden".to_string())
        } else if constant_name == "D" || constant_name.starts_with("D_") {
            Err("The name \"D\" and identifiers starting with \"D_\" are reserved for the total derivative.".to_string())
        } else {
            // The '.clone()' in below line is due to the fact that we want to save the value on one hand (within 'constants')
            // but also return it (e.g. the expression "x := 5" should not only define x as 5 but also return the value 5 so that
            // one can write "... * (x := ...)" to save intermediate results).
            env.constants.insert(constant_name.clone(), value.clone());
            Ok(value)
        }
    }

    match lhs {
        Expression::Identifier(ident)
            => define_constant(ident, eval(rhs, extra_vars, env)?, env),
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
            match eval(rhs, extra_vars, env)? {
                Object::Tuple(rhs_values) => {
                    if lhs_exprs.len() != rhs_values.len() {
                        return Err(format!("Tuples on both sides of assignment operator must be of equal length (got {}, {}).", lhs_exprs.len(), rhs_values.len()))
                    }
                    let mut rhs_values_iter = rhs_values.into_iter();
                    lhs_exprs.iter()
                    .map(|lhs_expr| {
                        if let Expression::Identifier(ident) = lhs_expr {
                            define_constant(ident, rhs_values_iter.next().unwrap(), env)
                        } else {
                            Err("All LHS entries must be identifiers.".to_string())
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
                    .map(Object::Tuple)
                }
                other => Err(format!("RHS couldn't be evaluated to a tuple (result: {}).", other))
            }
        }
        _ => Err(format!("Invalid LHS of assignment expression: {}", lhs))
    }
}