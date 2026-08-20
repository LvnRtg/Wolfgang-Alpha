use std::collections::HashMap;
use std::fmt;

use crate::lang::eval;
use crate::math::objects::try_operation;
use crate::math::operations::BinaryOperation;
use crate::math::{Env, Expression, Object, ObjType, VarStack};


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

#[allow(clippy::too_many_arguments)]
pub fn eval_folded_operation(
    op: &FoldedOperation,
    index_var: &String,
    from: &Expression,
    conditions: &[Expression],
    to: &Expression,
    inner: &Expression,
    extra_vars: &VarStack,
    env: &mut Env
) -> Result<Object, String> {
    // TODO evaluate conditions once at the start if acceptable
    // TODO search for all other FoldedOperation evals and fix accordingly
    let mut i = eval(from, extra_vars, env)?.expect_int()?;
    let initial_to_eval = eval(to, &VarStack::Frame { vars: &HashMap::from([(index_var, &Object::Real(i))]), parent: extra_vars }, env)?.expect_float()?;
    // We initialize `res` with the appropriate identity object (e.g. zero for a sum and one for a product). To do this, we must know the correct type,
    // which we obtain using `inner.get_type`. This in turn requires us to know the type of the index var, which we can obtain by simply evaluating it once.
    // Since we need the evaluation at `to` below anyway, we choose to evalute `index_var` at this point and not (say) at `from`.
    let mut res = op.if_empty(&inner.get_type(
        &VarStack::Frame { vars: &HashMap::from([(index_var, &Object::Real(initial_to_eval))]), parent: extra_vars },
        env
    )?);
    // If `i` is more than `to` rightaway (i.e. more than `initial_to_eval`), return the default value for a folded operator over an empty range.
    if i > initial_to_eval {
        return Ok(res);
    }
    let binop = op.underlying_binop();
    // We will rebuild the following varstack in every iteration (I didn't find a way to make it modify itself when i changes).
    let mut i_as_obj = Object::Real(i);
    let mut varstack_top_frame = HashMap::from([(index_var, &i_as_obj)]);
    let mut varstack = VarStack::Frame { vars: &varstack_top_frame, parent: extra_vars };
    // The below condition `i + 1.0 != i` is required because for too large floats, adding 1.0 becomes a non-op and prevents the loop from ever finishing.
    'outer: while i + 1.0 != i && i <= eval(to, &varstack, env)?.expect_float()? {
        // Check if all conditions are met. If not, skip this `i`.
        for cond in conditions {
            match eval(cond, &varstack, env)? {
                Object::Real(1.0) => {} // Condition met; ignore
                Object::Real(0.0) => { // Condition not met; skip `i`
                    i += 1.0;
                    // Rebuild varstack
                    i_as_obj = Object::Real(i);
                    varstack_top_frame = HashMap::from([(index_var, &i_as_obj)]);
                    varstack = VarStack::Frame { vars: &varstack_top_frame, parent: extra_vars };
                    continue 'outer;
                }
                other => return Err(format!("Expected 1 or 0 when evaluating condition, got {:?}.", other))
            }
        }
        // At this point, all conditions are met.
        let next_term = eval(inner, &varstack, env)?;
        res = try_operation(&res, &next_term, &binop)?;
        i += 1.0;
        // Rebuild varstack
        i_as_obj = Object::Real(i);
        varstack_top_frame = HashMap::from([(index_var, &i_as_obj)]);
        varstack = VarStack::Frame { vars: &varstack_top_frame, parent: extra_vars };
    }
    Ok(res)
}