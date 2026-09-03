use std::borrow::Cow;
use std::collections::HashSet;
use std::fmt;

use crate::lang::eval;
use crate::math::objects::try_operation;
use crate::math::{Env, Expression, Object, ObjType, utils, VarStack};
use crate::status::{ExtResult, Status};
use super::binary_operations::BinaryOperation;


/// Maximum amount of warnings emitted by a call to `eval_folded_operation`.
/// 
/// Reason for having a cap: every iteration could emit at least one warning (in fact, if some iteration emits a warning,
/// this is very likely), which would saturate the console and consume huge amounts of memory for nothing.
const FOLDED_OP_WARNING_CAP: usize = 10;


/// Any operation for which an operator of the type `sum_{i=1}^n ...` is implemented.
#[derive(Clone, Debug, PartialEq)]
pub enum FoldedOperation {
    Sum,
    Product
}
impl fmt::Display for FoldedOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FoldedOperation::Sum => write!(f, "sum"),
            FoldedOperation::Product => write!(f, "prod"),
        }
    }
}

impl FoldedOperation {
    pub fn priority(&self) -> u8 {
        match self {
            FoldedOperation::Sum => BinaryOperation::Add.priority(),
            FoldedOperation::Product => BinaryOperation::Mul.priority(),
        }
    }

    pub fn underlying_binop(&self) -> BinaryOperation {
        match self {
            FoldedOperation::Sum => BinaryOperation::Add,
            FoldedOperation::Product => BinaryOperation::Mul
        }
    }

    pub fn valid_string(str: &str) -> bool {
        str == "sum" || (str.starts_with("sum") && str.chars().nth(3) == Some('_'))
        || str == "prod" || (str.starts_with("prod") && str.chars().nth(4) == Some('_')) 
    }

    pub fn from_string(str: &str) -> Option<FoldedOperation> {
        if str == "sum" || (str.starts_with("sum") && str.chars().nth(3) == Some('_')) {
            Some(FoldedOperation::Sum)
        } else if str == "prod" || (str.starts_with("prod") && str.chars().nth(4) == Some('_')) {
            Some(FoldedOperation::Product)
        } else {
            None
        }
    }

    /// Returns the value of an empty folded operation of type `self` (e.g. 0 for sums, 1 for products).
    pub fn if_empty(&self, inner_type: &ObjType) -> Object {
        match self {
            FoldedOperation::Sum => inner_type.zero(),
            FoldedOperation::Product => inner_type.one()
        }
    }
}


/// Acts as a helper for `compute_folded_operation`
#[inline]
#[allow(clippy::too_many_arguments)]
pub fn folded_operation_helper<'a, F, G>(
    op: &FoldedOperation,
    index_var: &String,
    from: &Expression,
    conditions: &'a [Expression],
    to: &Expression,
    expr_to_get_inner_assigned_to_vars: &Expression, // Not used for evaluating the inner expression
    get_inner: F,
    get_type: G,
    extra_vars: &'a VarStack,
    env: &'a mut Env
) -> ExtResult
where
    F: FnMut (         &VarStack, &mut Env) -> ExtResult,
    G: FnOnce(&Object, &VarStack, &mut Env) -> Result<ObjType, String>
{
    // Because two macros never have the same type, the following `if/else` block needs to be pulled outside of the function call.
    // Check if `to` needs to be reevaluated in every iteration. From `README.md`, the precise criterion for `sum_{i=a}^b f(i)` is:
    // The expression `b` is evaluated in every iteration iff `b` contains the identifier `i` (in any form)
    // or `f(i)` contains an assignment `x := ...` for which `b` contains the identifier `x`.
    if to.contains_identifier(index_var) || {
        let mut assigned_to_vars = HashSet::new();
        expr_to_get_inner_assigned_to_vars.get_assigned_to_variables(&mut assigned_to_vars);
        to.contains_any_of(&assigned_to_vars)
    } {
        compute_folded_operation(
            op,
            index_var,
            |_varstack, _env| eval(from, _varstack, _env),
            conditions.iter().map(|condition: &Expression| {
                |_varstack: &VarStack<'_, '_>, _env: &mut Env| eval(condition, _varstack, _env)
            }).collect(),
            |_varstack, _env| eval(to, _varstack, _env).map(|s| s.map(Cow::Owned)),
            get_inner,
            |a, b, c| get_type(a, b, c).map(|t| Status::ok(op.if_empty(&t))),
            extra_vars,
            env
        )
    } else {
        let to_eval = eval(to, extra_vars, env)?;
        compute_folded_operation(
            op,
            index_var,
            |_varstack, _env| eval(from, _varstack, _env),
            conditions.iter().map(|condition: &Expression| {
                |_varstack: &VarStack<'_, '_>, _env: &mut Env| eval(condition, _varstack, _env)
            }).collect(),
            |_, _| Ok(to_eval.clone()).map(|s| s.map(Cow::Owned)),
            get_inner,
            |a, b, c| get_type(a, b, c).map(|t| Status::ok(op.if_empty(&t))),
            extra_vars,
            env
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compute_folded_operation<'a, FFrom, FTo, FInner, FCondition, FDefaultValue>(
    op: &FoldedOperation,
    index_var: &'a String,
    get_from: FFrom,
    mut get_conditions: Vec<FCondition>,
    mut get_to: FTo,
    mut get_inner: FInner,
    get_default_value: FDefaultValue, // We have this as parameter because then, we can use `.get_type()` on an expression which is much faster than `.eval().type()`
    extra_vars: &'a VarStack<'_, '_>,
    env: &'a mut Env
) -> ExtResult 
where
    FFrom:         FnOnce(         &VarStack, &mut Env) -> ExtResult, // Only evaluated once anyway
    FTo:           FnMut (         &VarStack, &mut Env) -> Result<Status<Cow<'a, Object>>, String>,
    FInner:        FnMut (         &VarStack, &mut Env) -> ExtResult,
    FCondition:    FnMut (         &VarStack, &mut Env) -> ExtResult,
    FDefaultValue: FnOnce(&Object, &VarStack, &mut Env) -> ExtResult
{
    let mut warnings = Vec::<String>::new();
    let mut i = (&get_from(extra_vars, env)?.unpack_into(&mut warnings)).expect_int()?;
    let mut current_to_eval = get_to(
        &extra_vars.with(index_var, Cow::Owned(Object::Real(i))),
        env
    )?.unpack_into(&mut warnings).expect_float()?;
    // We initialize `res` with the appropriate identity object (e.g. zero for a sum and one for a product). To do this, we must know the correct type,
    // which we obtain using `inner.get_type`. This in turn requires us to know the type of the index var, which we can obtain by simply evaluating it once.
    // Since we need the evaluation at `to` below anyway, we choose to evalute `index_var` at this point and not (say) at `from`.
    let mut res = get_default_value(&Object::Real(current_to_eval), extra_vars, env)?.unpack_into(&mut warnings);

    // If `i` is more than `to` rightaway (i.e. more than `current_to_eval`), return the default value for a folded operator over an empty range.
    if i > current_to_eval {
        return Ok(Status{value: res, warnings});
    }

    let binop = op.underlying_binop();

    // The below condition `i + 1.0 != i` is required because for too large floats, adding 1.0 becomes a non-op and prevents the loop from ever finishing.
    'outer: while i + 1.0 != i {
        // Build varstack for current i
        let varstack = extra_vars.with(index_var, Cow::Owned(Object::Real(i)));

        // Note: the caller is responsible for modifying `get_to` if `to` doesn't need to be recomputed in every iteration (cf. `eval_folded_operation`).
        current_to_eval = get_to(&varstack, env)?.unpack_into_with_cap(&mut warnings, FOLDED_OP_WARNING_CAP).expect_float()?;
        if i > current_to_eval {
            break;
        }

        // Check if all conditions are met. If not, skip this `i`.
        for cond in get_conditions.iter_mut().map(|f| f(&varstack, env)) {
            if !cond?.unpack_into_with_cap(&mut warnings, FOLDED_OP_WARNING_CAP).expect_bool()? {
                // Condition not met; skip `i`
                i += 1.0;
                continue 'outer;
            }
        }

        // At this point, all conditions are met.
        let next_term = get_inner(&varstack, env)?.unpack_into_with_cap(&mut warnings, FOLDED_OP_WARNING_CAP);
        res = try_operation(&res, &next_term, &binop)?;
        i += 1.0;
    }
    Ok(Status{value: res, warnings})
}

/// Write `i := index_var`. Given functions `f(i)`, `f_prime(i)` and **assuming `get_to` is constant**
/// (i.e. does not contain `i` and there exists no `x` contained in it such that calling `get_f` or `get_f_prime` redefines `x`),
/// this function computes `sum_i f_prime(i) prod_{j != i} f(j)`.
/// 
/// The reason this is a separate function is that under the constancy assumption, we can compute the range `i` loops over beforehand,
/// reducing the number of `eval`-calls from `O(n²)` to `O(n)` (recall `product_derivative_helper` is a double-sum).
#[allow(clippy::too_many_arguments)]
pub fn compute_product_derivative_helper<'a, FInner, FInnerPrime, FCondition>(
    index_var: &'a String,
    from: Status<Object>,
    to: Status<Object>,
    mut get_conditions: Vec<FCondition>,
    mut get_f: FInner,
    mut get_f_prime: FInnerPrime,
    extra_vars: &'a VarStack<'_, '_>,
    env: &'a mut Env
) -> ExtResult
where
    FInner:        FnMut(&VarStack, &mut Env) -> ExtResult,
    FInnerPrime:   FnMut(&VarStack, &mut Env) -> ExtResult,
    FCondition:    FnMut(&VarStack, &mut Env) -> ExtResult
{
    let Status::<f64>{value: from_eval, mut warnings} = from.try_map(|o| o.expect_int())?;
    let to_eval = to.unpack_into(&mut warnings).expect_float()?.floor();
    if from_eval > to_eval {
        warnings.push("Couldn't infer return type from empty product, defaulated to `Real`.".to_string());
        return Ok(Status{value: Object::Real(1.0), warnings});
    }
    
    // Compute range `i` goes over
    let i_range = ((from_eval as i64)..=(to_eval as i64))
    .filter_map(|i| {
        for cond_res in get_conditions.iter_mut().map(
            |f| f(
                &extra_vars.with(index_var, Cow::Owned(Object::Real(i as f64))),
                env
            )
        ) {
            match cond_res {
                Ok(cond) => match cond.unpack_into_with_cap(&mut warnings, FOLDED_OP_WARNING_CAP).expect_bool() {
                    Ok(true) => {},
                    Ok(false) => return None,
                    Err(e) => return Some(Err(e))
                }
                Err(e) => return Some(Err(e))
            }
        }
        Some(Ok(i))
    })
    .collect::<Result<Vec<_>, _>>()?;

    let summands = i_range.iter().enumerate().map(|(i_index, i)| {
        let first_factors = i_range[..i_index].iter().map(|j| {
            get_f(&extra_vars.with(index_var, Cow::Owned(Object::Real(*j as f64))), env)
            .map(|s| s.unpack_into_with_cap(&mut warnings, FOLDED_OP_WARNING_CAP))
        });
        let mut res = if let Some(r) = utils::fold_res_obj_iter(first_factors, &BinaryOperation::Mul) {
            // If `first_factors` is non-empty, compute `(prod_{x in first_factors} x) * f'(i)`
            r.and_then(|lhs| get_f_prime(&extra_vars.with(index_var, Cow::Owned(Object::Real(*i as f64))), env)
            .and_then(|rhs| try_operation(
                &lhs,
                &rhs.unpack_into_with_cap(&mut warnings, FOLDED_OP_WARNING_CAP),
                &BinaryOperation::Mul
            )))
        } else {
            // Otherwise, this is the same as `f'(i)`
            get_f_prime(&extra_vars.with(index_var, Cow::Owned(Object::Real(*i as f64))), env)
            .map(|s| s.unpack_into_with_cap(&mut warnings, FOLDED_OP_WARNING_CAP))
        };
        // Multiply with all remaining factors
        for j in i_range[i_index+1..].iter() {
            res = res.and_then(
                |lhs|
                get_f(&extra_vars.with(index_var, Cow::Owned(Object::Real(*j as f64))), env)
                .and_then(
                    |f_j| try_operation(&lhs, &f_j.unpack_into_with_cap(&mut warnings, FOLDED_OP_WARNING_CAP), &BinaryOperation::Mul)
                )
            );
        }
        res
    });
    // Safe to unwrap below since summands keeps the length of `i_range` and if `i_range` were empty, this function would have returned already.
    Ok(Status{value: utils::fold_res_obj_iter(summands, &BinaryOperation::Add).unwrap()?, warnings})
}