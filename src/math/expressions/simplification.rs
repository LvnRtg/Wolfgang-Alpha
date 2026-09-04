//! Contains functions that simplify expressions.

use crate::expr_binop;
use crate::math::operations::{BinaryOperation, UnaryOperation};
use super::Expression;

/// Allows to simplify literal expressions.
/// 
/// If `lhs` is zero, returns `rhs`. If `rhs` is zero, returns `lhs`. Otherwise, returns `lhs + rhs`.
pub fn simplify_add(lhs: Expression, rhs: Expression) -> Expression {
    match (lhs, rhs) {
        (Expression::Number(0.0), other) | (other, Expression::Number(0.0)) => other,
        (lhs, rhs) => Expression::BinaryOperation(Box::new(lhs), BinaryOperation::Add, Box::new(rhs))
    }
}
/// Allows to simplify literal expressions.
/// 
/// If `lhs` and `rhs` are both numbers, subtract and return the wrapped result.
/// If `lhs` is zero, returns `-rhs`. If `rhs` is zero, returns `lhs`. Otherwise, returns `lhs - rhs`.
pub fn simplify_sub(lhs: Expression, rhs: Expression) -> Expression {
    match (lhs, rhs) {
        (Expression::Number(x), Expression::Number(y)) => Expression::Number(x-y),
        (Expression::Number(0.0), rhs) => Expression::UnaryOperation(UnaryOperation::Neg, Box::new(rhs)),
        (lhs, Expression::Number(0.0)) => lhs,
        (lhs, rhs) => Expression::BinaryOperation(Box::new(lhs), BinaryOperation::Sub, Box::new(rhs))
    }
}
/// Allows to simplify literal expressions.
/// 
/// If one term is `0`, returns `0`. If one term is `1`, returns the other one. Otherwise, returns `lhs * rhs`.
pub fn simplify_mul(lhs: Expression, rhs: Expression) -> Expression {
    let (lhs, rhs) = match (lhs, rhs) { // Put the Expression::Number first if there is one
        (n @ Expression::Number(_), other) | (other, n @ Expression::Number(_)) => (n, other),
        other => other
    };
    match (lhs, rhs) {
        (Expression::Number(0.0), _) => Expression::Number(0.0),
        (Expression::Number(1.0), other) => other,
        (Expression::Number(x), Expression::Number(y)) => Expression::Number(x*y),
        (Expression::Number(x), Expression::BinaryOperation(inner_l, BinaryOperation::Mul, inner_r))
        | (Expression::BinaryOperation(inner_l, BinaryOperation::Mul, inner_r), Expression::Number(x)) => {
            match (*inner_l, *inner_r) {
                (Expression::Number(y), other) | (other, Expression::Number(y)) => expr_binop!(Expression::Number(x*y), Mul, other),
                (inner_l, inner_r) => expr_binop!(Expression::Number(x), Mul, expr_binop!(inner_l, Mul, inner_r))
            }
        }
        (lhs, rhs) => Expression::BinaryOperation(Box::new(lhs), BinaryOperation::Mul, Box::new(rhs))
    }
}
/// Allows to simplify literal expressions.
/// 
/// If `rhs` is `1`, returns `lhs`. Otherwise, returns `lhs / rhs`.
pub fn simplify_div(lhs: Expression, rhs: Expression) -> Expression {
    if let Expression::Number(1.0) = rhs {
        lhs
    }
    else {
        Expression::BinaryOperation(Box::new(lhs), BinaryOperation::Div, Box::new(rhs))
    }
}
/// Allows to simplify literal expressions.
/// 
/// If `rhs` is `1`, returns `lhs`. If `rhs` is `0` or `lhs` is `1`, returns `1`. Otherwise, returns `lhs ^ rhs`.
pub fn simplify_pow(lhs: Expression, rhs: Expression) -> Expression {
    if let Expression::Number(1.0) = rhs {
        lhs
    }
    else if let Expression::Number(0.0) = rhs {
        Expression::Number(1.0)
    }
    else if let Expression::Number(1.0) = lhs {
        lhs
    }
    else {
        Expression::BinaryOperation(Box::new(lhs), BinaryOperation::Pow(true), Box::new(rhs))
    }
}