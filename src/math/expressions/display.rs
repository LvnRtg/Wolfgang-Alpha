//! Implements functions to display expressions, either in a single line or over potentially multiple lines.

use std::fmt;

use crate::math::operations::{BinaryOperation, UnaryOperation};
use super::Expression;


// Contains more parentheses than would be mathematically necessary because this is used for debugging.
// `fmt::Debug` is very verbose (e.g. `Identifier("x"` instead of `x`); `fmt::Display` is supposed to maintain
// the same level of precision while not being _as_ verbose.
impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::None => write!(f, "None"),
            Expression::Identifier(s) => write!(f, "{}", s),
            Expression::Number(x) => write!(f, "{}", x),
            Expression::Tuple(x) => write!(f, "({})", x.iter().map(|y| format!("{}", y)).collect::<Vec<String>>().join(", ")),
            Expression::Vector(x) => write!(f, "[{}]", x.iter().map(|y| format!("{}", y)).collect::<Vec<String>>().join(", ")),
            Expression::Matrix(m, n, x) => write!(f, "[{}]", (0..*m).map(|i| (0..*n).map(|j| format!("{}", x[i*n+j])).collect::<Vec<String>>().join(", ")).collect::<Vec<String>>().join("; ")),
            Expression::UnaryOperation(op, r) => {
                match op {
                    UnaryOperation::Neg => write!(f, "(-({}))", r),
                    UnaryOperation::Not => write!(f, "!({})", r),
                    UnaryOperation::Factorial => write!(f, "({})!", r),
                    UnaryOperation::Abs => write!(f, "|{}|", r),
                    UnaryOperation::Norm(opt) => write!(f, "||{}||{}", r, crate::math::operations::unary_operations::format_optional_subscript(opt)),
                }
            },
            Expression::BinaryOperation(l, op, r) => write!(f, "({} {} {})", l, op, r),
            Expression::FoldedOperation(op, ident, from, conditions, to, inner_operand)
                => write!(f, "{}_{{{}={}{}{}}}^{{{}}} {}", op, ident, from, if conditions.is_empty() {""} else {", "}, conditions.iter().map(|e| format!("{}", e)).collect::<Vec<_>>().join(", "), to, inner_operand),
            Expression::Function(name, args)
                => write!(f, "{}({})", name, args.iter().map(|x| format!("{}", x)).collect::<Vec<String>>().join(", ")),
            Expression::Assignment(lhs, rhs) => write!(f, "{} := {}", lhs, rhs),
            Expression::PartialDerivative(wrt, expr) => write!(f, "d/d{} ({})", wrt, expr),
            Expression::DirectionalDerivative(vars, expr, point, direction)
                => write!(f, "D_{{{}}} ({})({:?})[{:?}]", vars.join(", "), expr, point, direction),
            Expression::Integral(func, a, b, x)
                => write!(f, "int_{{{}}}^{{{}}} ({}) d{}", a, b, func, x),
            Expression::IfElse(condition, iftrue, iffalse)
                => write!(f, "if ({}) {{{}}} else {{{}}}", condition, iftrue, iffalse),
        }
    }
}

macro_rules! multline_vector {
    ($left_delimiter:expr, $elements:expr, $right_delimiter:expr) => {{
        // We display the vector in expanded form (i.e. one component per line) if at least one of the following holds:
        // - A component spans multiple lines
        // - A component has at least 15 chars.
        let mut multlines = $elements.iter().map(|y| y.to_multline()).collect::<Vec<Vec<String>>>();
        if multlines.iter().any(|v| v.len() > 1 || v.iter().any(|elem| elem.chars().count() >= 15)) {
            let mut result = vec![$left_delimiter.to_string()];
            multlines.iter_mut().for_each(
                |v| {
                    v.last_mut().unwrap().push(',');
                    v.iter_mut().for_each(|x| x.insert_str(0, "  "));
                }
            );
            result.reserve(multlines.iter().map(|r| r.len()).sum());
            result.extend(multlines.into_iter().flatten());
            result.push($right_delimiter.to_string());
            result
        } else {
            vec![format!(
                "{}{}{}",
                $left_delimiter,
                multlines.into_iter().map(|v| v.into_iter().next().unwrap()).collect::<Vec<String>>().join(", "),
                $right_delimiter
            )]
        }
    }};
}

impl Expression {
    /// Returns `format!("{}", self)` surrounded by braces if the expression isn't an identifier or a number.
    pub fn to_string_with_braces(&self) -> String {
        match self {
            Expression::Number(x) => x.to_string(),
            Expression::Identifier(x) => x.clone(),
            other => format!("{{{}}}", other)
        }
    }
    
    /// Formats an object to a string that may stretch over multiple lines.
    /// The lines will be returned as a vector of strings, not as a single string containing newline chars.
    /// 
    /// This function will attempt to avoid mathematically unnecessary parentheses for a more readable output.
    pub fn to_multline(&self) -> Vec<String> {
        match self {
            Expression::None => vec!["None".to_string()],
            Expression::Identifier(s) => vec![format!("{}", s)],
            Expression::Number(x) => vec![format!("{}", x)],
            Expression::Tuple(components) => multline_vector!('(', components, ')'),
            Expression::Vector(components) => multline_vector!('[', components, ']'),
            Expression::Function(name, args) => {
                let mut result = multline_vector!('(', args, ')');
                if let Some(first) = result.first_mut() {
                    first.insert_str(0, name);
                    result
                } else {
                    vec![name.clone()]
                }
            }
            Expression::Matrix(m, n, x) => {
                let values = x.iter().map(|b| b.to_multline().join(" ")).collect::<Vec<String>>();
                let column_lengths: Vec<usize> = (0..*n).map(
                    |j| (0..*m).map(
                        |i| values[i*n+j].chars().count()
                    ).max().unwrap_or(0)
                ).collect();
                let row_length = column_lengths.iter().sum::<usize>() + 2*n; // Between two columns, add 2 spaces. Before the first columns and after the last one, only 1 space.
                let mut lines = vec![format!("╭{}╮", (0..row_length).map(|_| ' ').collect::<String>())];
                for i in 0..*m {
                    lines.push(format!("│ {}│", (0..*n).map(
                        |j| format!("{:^2$} {}", values[i*n+j], if j == n-1 {""} else {" "}, column_lengths[j])
                    ).collect::<String>()));
                }
                lines.push(format!("╰{}╯", (0..row_length).map(|_| ' ').collect::<String>()));
                lines
            }
            Expression::UnaryOperation(op, r) => {
                // Here, only some types of `r` require extra parentheses around them. Specifically, if `op != Abs` and `op != Norm` (in which case no `r` needs parentheses),
                // UnaryOp(neither Abs nor op if matches!(op, Factorial|Not)), BinaryOp, Assignment, and both Derivatives
                // need extra parentheses around them.
                let mut multlined_inner = r.to_multline();
                let op_is_not_abs_or_norm = op != &UnaryOperation::Abs && !matches!(op, UnaryOperation::Norm(_));
                if op_is_not_abs_or_norm
                && matches!(&**r, Expression::BinaryOperation(..) | Expression::Assignment(..) | Expression::PartialDerivative(..) | Expression::DirectionalDerivative(..))
                || matches!(&**r, Expression::UnaryOperation(other_op, _) if op_is_not_abs_or_norm && !(other_op == op && matches!(op, UnaryOperation::Factorial | UnaryOperation::Not))) {
                    multlined_inner[0].insert(0, '(');
                    multlined_inner.last_mut().unwrap().push(')');
                }
                op.format_with_multline_expr(&mut multlined_inner);
                multlined_inner
            }
            Expression::BinaryOperation(l, op, r) => {
                // The left side needs parentheses if it is one of the following:
                // Assignment, a Derivative, a BinaryOp of strictly lower priority than `op`
                let mut multlined_left = l.to_multline();
                if matches!(&**l, Expression::Assignment(..) | Expression::PartialDerivative(..) | Expression::DirectionalDerivative(..))
                || matches!(&**l, Expression::BinaryOperation(_, other_op, _) if other_op.priority() < op.priority()) {
                    multlined_left[0].insert(0, '(');
                    multlined_left.last_mut().unwrap().push(')');
                }
                // The right side needs parentheses if it is one of the following:
                // Assignment, a Derivative, a BinaryOp of lower OR EQUAL priority to `op`
                let mut multlined_right = r.to_multline();
                if matches!(&**r, Expression::Assignment(..) | Expression::PartialDerivative(..) | Expression::DirectionalDerivative(..))
                || matches!(&**r, Expression::BinaryOperation(_, other_op, _) if other_op.priority() <= op.priority()) {
                    multlined_right[0].insert(0, '(');
                    multlined_right.last_mut().unwrap().push(')');
                }
                let mut right_iter = multlined_right.into_iter();
                multlined_left.last_mut().unwrap().push_str(format!(
                    "{}{}",
                    match op {
                        BinaryOperation::Pow(_) => op.as_str().to_string(),
                        BinaryOperation::Mul if matches!(&**l, Expression::Number(_)) && !matches!(&**r, Expression::Number(_) | Expression::IfElse(..)) => String::new(),
                        _ => format!(" {} ", op.as_str())
                    },
                    right_iter.next().unwrap()).as_str()
                );
                multlined_left.extend(right_iter);
                multlined_left
            }
            Expression::FoldedOperation(op, ident, from, conditions, to, inner_operand) => {
                let mut multlined_inner = inner_operand.to_multline();
                // The inner operand only needs extra parentheses around it if it is a BinaryOperation of lower or equal priority to `op`.
                if let Expression::BinaryOperation(_, inner_op, _) = &**inner_operand && inner_op.priority() <= op.priority() {
                    multlined_inner.first_mut().unwrap().insert(0, '(');
                    multlined_inner.last_mut().unwrap().push(')');
                }
                // Notice that for `from` and `to`, we use `fmt::Display` instead of `to_multline()` since we don't want sub- and superscripts
                // of the folded operator to span several lines.
                if multlined_inner.len() > 1 {
                    multlined_inner.insert(0, format!("{}_{{{}={}{}{}}}^{{{}}}", op, ident, from, if conditions.is_empty() {""} else {", "}, conditions.iter().map(|e| format!("{}", e)).collect::<Vec<_>>().join(", "), to));
                } else {
                    multlined_inner.first_mut().unwrap().insert_str(0, format!("{}_{{{}={}{}{}}}^{{{}}} ", op, ident, from, if conditions.is_empty() {""} else {", "}, conditions.iter().map(|e| format!("{}", e)).collect::<Vec<_>>().join(", "), to).as_str());
                }
                multlined_inner
            }
            Expression::Assignment(l, r) => {
                let mut multlined_left = l.to_multline();
                let multlined_right = r.to_multline();
                let mut right_iter = multlined_right.into_iter();
                multlined_left.last_mut().unwrap().push_str(format!(" := {}", right_iter.next().unwrap()).as_str());
                multlined_left.extend(right_iter);
                multlined_left
            }
            Expression::PartialDerivative(wrt, expr) => {
                let mut multlined = expr.to_multline();
                multlined[0].insert_str(0, format!("d/d{} (", wrt).as_str());
                multlined.last_mut().unwrap().push(')');
                multlined
            }
            Expression::DirectionalDerivative(vars, expr, point, direction) => {
                let mut multlined_expr = expr.to_multline();
                let multlined_point = point.iter().map(|x| x.to_multline()).collect::<Vec<Vec<String>>>();
                let multlined_direction = direction.iter().map(|x| x.to_multline()).collect::<Vec<Vec<String>>>();
                multlined_expr[0].insert_str(0, format!("D_{{{}}} (", vars.join(", ")).as_str());
                multlined_expr.last_mut().unwrap().push_str(format!(
                    ")({})[{}]",
                    multlined_point.into_iter().map(|v| v.join(" ")).collect::<Vec<String>>().join(", "),
                    multlined_direction.into_iter().map(|v| v.join(" ")).collect::<Vec<String>>().join(", "),
                ).as_str());
                multlined_expr
            }
            Expression::Integral(func, a, b, x) => {
                let mut multlined = func.to_multline();
                if multlined.len() == 1 {
                    multlined[0].insert_str(0, format!("int_{}^{} ", a.to_string_with_braces(), b.to_string_with_braces()).as_str());
                    multlined[0].push_str(format!(" d{}", x).as_str());
                } else {
                    multlined.iter_mut().for_each(|l| l.insert_str(0, "  "));
                    multlined.insert(0, format!("int_{}^{}", a.to_string_with_braces(), b.to_string_with_braces()));
                    multlined.push(format!(" d{}", x));
                }
                multlined
            }
            Expression::IfElse(condition, iftrue, iffalse) => {
                let mut multlined_cond = condition.to_multline();
                let mut multlined_true = iftrue.to_multline();
                let mut multlined_false = iffalse.to_multline();
                multlined_true.iter_mut().for_each(|x| x.insert_str(0, "  "));
                multlined_false.iter_mut().for_each(|x| x.insert_str(0, "  "));
                multlined_cond[0].insert_str(0, "if (");
                multlined_cond.last_mut().unwrap().push_str(") {");
                multlined_cond.extend(multlined_true);
                multlined_cond.push("} else {".to_string());
                multlined_cond.extend(multlined_false);
                multlined_cond.push("}".to_string());
                multlined_cond
            }
        }
    }
}