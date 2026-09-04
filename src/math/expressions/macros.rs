//! Contains macros to simplify typing and enhance readability.

#[macro_export]
macro_rules! expr_unary_op {
    ($op:ident$(($param:expr))?, $rhs:expr) => {
        crate::math::Expression::UnaryOperation(
            crate::math::operations::UnaryOperation::$op$(($param))?,
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_binop {
    ($lhs:expr, $op:ident$(($($params:expr),*))?, $rhs:expr) => {
        crate::math::Expression::BinaryOperation(
            Box::new($lhs),
            crate::math::operations::BinaryOperation::$op$(($($params),*))?,
            Box::new($rhs)
        )
    };
    // Recursive case: fold left to right.
    ($lhs:expr, $op:ident$(($($params:expr),*))?, $rhs:expr, $($rest:expr),+ $(,)?) => {
        $crate::expr_binop!(
            $crate::expr_binop!($lhs, $op, $rhs),
            $op$(($($params),*))?,
            $($rest),+
        )
    };
}
#[macro_export]
macro_rules! expr_binop_from_iter {
    ($binop:ident, $folded_op:ident, $iter:expr) => {{
        let mut __iter = ::std::iter::IntoIterator::into_iter($iter);
        let __first = __iter.next().unwrap_or($crate::math::operations::FoldedOperation::$folded_op.if_empty(&crate::math::ObjType::Scalar).to_expression());
        __iter.fold(__first, |lhs, rhs| {
            $crate::math::Expression::BinaryOperation(
                Box::new(lhs),
                $crate::math::operations::BinaryOperation::$binop,
                Box::new(rhs),
            )
        })
    }};
}
#[macro_export]
macro_rules! expr_binop_from_enum {
    ($lhs:expr, $op:expr, $rhs:expr) => {
        crate::math::Expression::BinaryOperation(
            Box::new($lhs),
            $op,
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_compare {
    ($lhs:expr, $comparison_operator:ident, $rhs:expr) => {
        crate::math::Expression::BinaryOperation(
            Box::new($lhs),
            crate::math::operations::BinaryOperation::Comp($crate::math::operations::Comparison::$comparison_operator, None),
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_if_else {
    ($condition:expr, $iftrue:expr, $iffalse:expr) => {
        crate::math::Expression::IfElse(
            Box::new($condition),
            Box::new($iftrue),
            Box::new($iffalse)
        )
    };
}
#[macro_export]
macro_rules! expr_inv {
    ($rhs:expr) => {
        crate::math::Expression::BinaryOperation(
            Box::new(Expression::Number(1.0)),
            crate::math::operations::BinaryOperation::Div,
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_square {
    ($lhs:expr) => {
        crate::math::Expression::BinaryOperation(
            Box::new($lhs),
            crate::math::operations::BinaryOperation::Pow(true),
            Box::new(Expression::Number(2.0))
        )
    };
}
#[macro_export]
macro_rules! expr_1arg_func {
    ($name:expr, $arg:expr) => {
        crate::math::Expression::Function(
            $name.to_string(),
            vec![$arg]
        )
    };
}
