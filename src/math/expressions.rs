use std::fmt;
use std::collections::{HashMap, HashSet};

use crate::math::{Env, Object, VarStack};
use crate::math::operations::*;

#[derive(Debug, Clone, PartialEq)]
pub enum Expression {
    None, // Used as the LHS of unary operations
    Identifier(String),
    Number(f64),
    /// This also doubles as a container for a function's arguments when the function isn't defined yet (cf. `Assignment` block in `eval`).
    Vector(Vec<Expression>), // As for functions
    /// Dimensions of the matrix and list of entries in flattened version.
    Matrix(usize, usize, Vec<Expression>), // Same
    UnaryOperation(UnaryOperation, Box<Expression>),
    /// Comparisons are interpreted as binary operations too.
    BinaryOperation(Box<Expression>, BinaryOperation, Box<Expression>),
    /// E.g. `sum_{i=1, i != 3}^n f(i)` will become `FoldedOperation(Sum, "i", 1, [i != j], n, f(i))`.
    /// There can be as many conditions as desired, including none at all.
    FoldedOperation(FoldedOperation, String, Box<Expression>, Vec<Expression>, Box<Expression>, Box<Expression>),
    /// Respectively: function's name and list of arguments passed.
    Function(String, Vec<Expression>),
    /// A collection of comma-separated expressions between parentheses.
    Tuple(Vec<Expression>),
    /// Format: LHS := RHS
    Assignment(Box<Expression>, Box<Expression>),
    /// Compute the partial derivative of the given expression w.r.t. the given identifier. The direction to differentiate in is set to 1.0.
    PartialDerivative(String, Box<Expression>),
    /// Compute the directional derivative of `SecondArg` at point `ThirdArg` in direction `FourthArg` where the variables w.r.t. which we differentiate are `first_args`.
    DirectionalDerivative(Vec<String>, Box<Expression>, Vec<Expression>, Vec<Expression>),
    /// E.g. `int_a^b f(x) dx` gives `Integral(f(x), a, b, x)`.
    Integral(Box<Expression>, Box<Expression>, Box<Expression>, String),
    /// `if (FirstArg) { SecondArg } else { ThirdArg }`
    IfElse(Box<Expression>, Box<Expression>, Box<Expression>)
}

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
                    UnaryOperation::Norm(opt) => write!(f, "||{}||{}", r, format_optional_subscript(opt)),
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
            Expression::Tuple(components) => {
                let mut multlines = components.iter().map(|y| y.to_multline()).collect::<Vec<Vec<String>>>();
                // We display the vector in expanded form (i.e. one component per line) if at least one of the following holds:
                // A component spans multiple lines; a component has length >= 15 chars.
                if multlines.iter().any(|v| v.len() > 1 || v.iter().any(|elem| elem.len() >= 15)) {
                    let mut result = vec!["(".to_string()];
                    multlines.iter_mut().for_each( // Indent every component of the vector and add a comma at the very end
                        |v| {
                            v.last_mut().unwrap().push(',');
                            v.iter_mut().for_each(|x| x.insert_str(0, "  "));
                        }
                    );
                    result.reserve(multlines.iter().map(|r| r.len()).sum());
                    result.extend(multlines.into_iter().flatten());
                    result.push(")".to_string());
                    result
                }
                else {
                    vec![format!("[{}]", multlines.into_iter().map(|v| v.into_iter().next().unwrap()).collect::<Vec<String>>().join(", "))]
                }
            }
            Expression::Vector(components) => {
                let mut multlines = components.iter().map(|y| y.to_multline()).collect::<Vec<Vec<String>>>();
                // We display the vector in expanded form (i.e. one component per line) if at least one of the following holds:
                // A component spans multiple lines; a component has length >= 15 chars.
                if multlines.iter().any(|v| v.len() > 1 || v.iter().any(|elem| elem.len() >= 15)) {
                    let mut result = vec!["[".to_string()];
                    multlines.iter_mut().for_each( // Indent every component of the vector and add a comma at the very end
                        |v| {
                            v.last_mut().unwrap().push(',');
                            v.iter_mut().for_each(|x| x.insert_str(0, "  "));
                        }
                    );
                    result.reserve(multlines.iter().map(|r| r.len()).sum());
                    result.extend(multlines.into_iter().flatten());
                    result.push("]".to_string());
                    result
                }
                else {
                    vec![format!("[{}]", multlines.into_iter().map(|v| v.into_iter().next().unwrap()).collect::<Vec<String>>().join(", "))]
                }
            }
            Expression::Matrix(m, n, x) => {
                let values = x.iter().map(|b| b.to_multline().join(" ")).collect::<Vec<String>>();
                let column_lengths: Vec<usize> = (0..*n).map(
                    |j| (0..*m).map(
                        |i| values[i*n+j]
                        .len()
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
            Expression::Function(name, args) => {
                let mut multlines = args.iter().map(|y| y.to_multline()).collect::<Vec<Vec<String>>>();
                // We display the vector in expanded form (i.e. one component per line) if at least one of the following holds:
                // A component spans multiple lines; a component has length >= 15 chars.
                if multlines.iter().any(|v| v.len() > 1 || v.iter().any(|elem| elem.len() >= 15)) {
                    let mut result = vec![format!("{name}(")];
                    multlines.iter_mut().for_each( // Indent every component of the vector and add a comma at the very end
                        |v| {
                            v.last_mut().unwrap().push(',');
                            v.iter_mut().for_each(|x| x.insert_str(0, "  "));
                        }
                    );
                    result.reserve(multlines.iter().map(|r| r.len()).sum());
                    result.extend(multlines.into_iter().flatten());
                    result.push(")".to_string());
                    result
                }
                else {
                    vec![format!("{}({})", name, multlines.into_iter().map(|v| v.into_iter().next().unwrap()).collect::<Vec<String>>().join(", "))]
                }
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
                (Expression::Number(y), other) | (other, Expression::Number(y)) => crate::expr_mul!(Expression::Number(x*y), other),
                (inner_l, inner_r) => crate::expr_mul!(Expression::Number(x), crate::expr_mul!(inner_l, inner_r))
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


impl Expression {
    /// Parses itself recursively and replaces every encountered `ident` by `by`. Ignores the LHS of assignment operators.
    pub fn replace_identifiers_in_place(&mut self, ident: &String, by: &Expression) {
        match self {
            Expression::Identifier(x) if x == ident => {
                *self = by.clone();
            }
            Expression::Tuple(v) | Expression::Vector(v) | Expression::Matrix(.., v) | Expression::Function(_, v)
                => {v.iter_mut().for_each(|x| x.replace_identifiers_in_place(ident, by));}
            Expression::UnaryOperation(_, x) | Expression::PartialDerivative(_, x) | Expression::Assignment(_, x) // Ignore LHS of assigment operator
                => x.replace_identifiers_in_place(ident, by),
            Expression::BinaryOperation(x, _, y)
                => {x.replace_identifiers_in_place(ident, by); y.replace_identifiers_in_place(ident, by);}
            Expression::DirectionalDerivative(_, x, point, direction) => {
                x.replace_identifiers_in_place(ident, by);
                point.iter_mut().for_each(|y| y.replace_identifiers_in_place(ident, by));
                direction.iter_mut().for_each(|y| y.replace_identifiers_in_place(ident, by));
            }
            Expression::IfElse(x, y, z)
                => {x.replace_identifiers_in_place(ident, by); y.replace_identifiers_in_place(ident, by); z.replace_identifiers_in_place(ident, by);}
            _ => {}
        }
    }

    /// Clones `self` while replacing every encountered `ident` by `by`. Ignores the LHS of assignment operators.
    pub fn replace_identifiers(&self, ident: &String, by: &Expression) -> Expression {
        match self {
            Expression::None => Expression::None,
            Expression::Identifier(x) => if x == ident {by.clone()} else {Expression::Identifier(x.clone())},
            Expression::Number(x) => Expression::Number(*x),
            Expression::Tuple(v) => Expression::Tuple(v.iter().map(|x| x.replace_identifiers(ident, by)).collect()),
            Expression::Vector(v) => Expression::Vector(v.iter().map(|x| x.replace_identifiers(ident, by)).collect()),
            Expression::Matrix(m, n, v) => Expression::Matrix(*m, *n, v.iter().map(|x| x.replace_identifiers(ident, by)).collect()),
            Expression::Function(name, v) => Expression::Function(name.clone(), v.iter().map(|x| x.replace_identifiers(ident, by)).collect()),
            Expression::UnaryOperation(op, x) => Expression::UnaryOperation(op.clone(), Box::new(x.replace_identifiers(ident, by))),
            Expression::BinaryOperation(lhs, op, rhs)
                => Expression::BinaryOperation(Box::new(lhs.replace_identifiers(ident, by)), op.clone(), Box::new(rhs.replace_identifiers(ident, by))),
            Expression::FoldedOperation(op, varname, from, conditions, to, inner) => Expression::FoldedOperation(
                op.clone(),
                varname.clone(),
                Box::new(from.replace_identifiers(ident, by)),
                conditions.iter().map(|x| x.replace_identifiers(ident, by)).collect(),
                Box::new(to.replace_identifiers(ident, by)),
                Box::new(inner.replace_identifiers(ident, by))
            ),
            Expression::PartialDerivative(wrt, x) => Expression::PartialDerivative(wrt.clone(), Box::new(x.replace_identifiers(ident, by))),
            Expression::Assignment(lhs, rhs) => Expression::Assignment(lhs.clone(), Box::new(rhs.replace_identifiers(ident, by))),
            Expression::DirectionalDerivative(vars, expr, point, direction) => Expression::DirectionalDerivative(
                vars.clone(),
                Box::new(expr.replace_identifiers(ident, by)),
                point.iter().map(|x| x.replace_identifiers(ident, by)).collect(),
                direction.iter().map(|x| x.replace_identifiers(ident, by)).collect()
            ),
            Expression::IfElse(x, y, z) => Expression::IfElse(
                Box::new(x.replace_identifiers(ident, by)),
                Box::new(y.replace_identifiers(ident, by)),
                Box::new(z.replace_identifiers(ident, by))
            ),
            Expression::Integral(func, a, b, x) => Expression::Integral(
                Box::new(func.replace_identifiers(ident, by)),
                Box::new(a.replace_identifiers(ident, by)),
                Box::new(b.replace_identifiers(ident, by)),
                x.clone()
            )
        }
    }

    /// Parses the expression `expr` recursively and collects all identifiers that are neither in `constants` nor in `extra_vars` into a HashSet `modified_identifiers`.
    /// 
    /// Returns whether or not anything was modified. The parameter `modified_anything` should be set to `false` for the first call and will then be passed down recursively.
    pub fn list_unknown_identifiers(
        &self,
        extra_vars: &VarStack,
        env: &Env,
        modified_identifiers: &mut HashSet<String>
    ) {
        match self {
            Expression::None | Expression::Number(_) => {},
            Expression::Identifier(x) => {
                if !env.constants.contains_key(x) && extra_vars.lookup(x).is_none() {
                    modified_identifiers.insert(x.clone());
                }
            }
            Expression::Tuple(v) | Expression::Vector(v) | Expression::Matrix(.., v)
                => v.iter().for_each(|e| e.list_unknown_identifiers(extra_vars, env, modified_identifiers)),
            Expression::UnaryOperation(_, expr) => expr.list_unknown_identifiers(extra_vars, env, modified_identifiers),
            Expression::BinaryOperation(lhs, _, rhs) => {
                lhs.list_unknown_identifiers(extra_vars, env, modified_identifiers);
                rhs.list_unknown_identifiers(extra_vars, env, modified_identifiers);
            }
            Expression::FoldedOperation(_, varname, from, conditions, to, inner) => {
                // Important: `varname` is no longer unknown within `inner`; however, it is still unknown within `from` and `to`.
                from.list_unknown_identifiers(extra_vars, env, modified_identifiers); // Here, use old `extra_vars`
                conditions.iter().for_each(|v| v.list_unknown_identifiers(extra_vars, env, modified_identifiers));
                to.list_unknown_identifiers(extra_vars, env, modified_identifiers); // Here too
                inner.list_unknown_identifiers( // But here, declare `varname` as known (temporarily for this function call)
                    &VarStack::Frame {
                        vars: &HashMap::from([(varname, &Object::Success)]),
                        parent: extra_vars
                    },
                    env,
                    modified_identifiers
                )
            }
            Expression::Function(_, args) => {
                args.iter().for_each(|arg| arg.list_unknown_identifiers(extra_vars, env, modified_identifiers));
            }
            Expression::Assignment(_, rhs) => rhs.list_unknown_identifiers(extra_vars, env, modified_identifiers), // Do not modify the LHS of assignment expressions
            Expression::PartialDerivative(wrt, expr) => expr.list_unknown_identifiers(
                &VarStack::Frame { // Same proceeding as in `Expression::FoldedOperation`
                    vars: &HashMap::from([(wrt, &Object::Success)]),
                    parent: extra_vars
                },
                env,
                modified_identifiers
            ),
            Expression::DirectionalDerivative(vars, expr, point, direction) => {
                expr.list_unknown_identifiers(
                    &VarStack::Frame {
                        vars: &vars.iter().map(|v| (v, &Object::Success)).collect(),
                        parent: extra_vars
                    },
                    env,
                    modified_identifiers
                );
                point.iter().for_each(|v| v.list_unknown_identifiers(extra_vars, env, modified_identifiers));
                direction.iter().for_each(|v| v.list_unknown_identifiers(extra_vars, env, modified_identifiers));
            }
            Expression::IfElse(x, y, z) => {
                x.list_unknown_identifiers(extra_vars, env, modified_identifiers);
                y.list_unknown_identifiers(extra_vars, env, modified_identifiers);
                z.list_unknown_identifiers(extra_vars, env, modified_identifiers);
            }
            Expression::Integral(func, a, b, wrt) => {
                func.list_unknown_identifiers(
                    &VarStack::Frame {
                        vars: &HashMap::from([(wrt, &Object::Success)]),
                        parent: extra_vars
                    },
                    env,
                    modified_identifiers
                );
                a.list_unknown_identifiers(extra_vars, env, modified_identifiers);
                b.list_unknown_identifiers(extra_vars, env, modified_identifiers);
            }
        }
    }

    pub fn contains_identifier(&self, identifier: &String) -> bool {
        match self {
            Expression::None | Expression::Number(_) => false,
            Expression::Identifier(x) => identifier == x,
            Expression::Tuple(v) | Expression::Vector(v) | Expression::Matrix(.., v)
                => v.iter().any(|e| e.contains_identifier(identifier)),
            Expression::UnaryOperation(_, expr) => expr.contains_identifier(identifier),
            Expression::BinaryOperation(lhs, _, rhs) => {
                lhs.contains_identifier(identifier)
                || rhs.contains_identifier(identifier)
            }
            Expression::FoldedOperation(.., from, conditions, to, inner) => {
                from.contains_identifier(identifier)
                || conditions.iter().any(|c| c.contains_identifier(identifier))
                || to.contains_identifier(identifier)
                || inner.contains_identifier(identifier)
            }
            Expression::Function(_, args) => {
                args.iter().any(|arg| arg.contains_identifier(identifier))
            }
            Expression::Assignment(_, rhs) => rhs.contains_identifier(identifier),
            Expression::PartialDerivative(_, expr) => expr.contains_identifier(identifier),
            Expression::DirectionalDerivative(_, expr, point, direction) => {
                expr.contains_identifier(identifier)
                || point.iter().any(|v| v.contains_identifier(identifier))
                || direction.iter().any(|v| v.contains_identifier(identifier))
            },
            Expression::IfElse(x, y, z) => {
                x.contains_identifier(identifier)
                || y.contains_identifier(identifier)
                || z.contains_identifier(identifier)
            },
            Expression::Integral(func, a, b, wrt) => {
                a.contains_identifier(identifier)
                || b.contains_identifier(identifier)
                || (func.contains_identifier(identifier) && wrt != identifier) // Ignore presence of `identifier` in `func` if we integrate w.r.t. `identifier`
            }
        }
    }
}


// The following macros simplify typing and enhance readability by a LOT. I only add these that are actively used.
#[macro_export]
macro_rules! expr_if_else {
    ($condition:expr, $iftrue:expr, $iffalse:expr) => {
        Expression::IfElse(
            Box::new($condition),
            Box::new($iftrue),
            Box::new($iffalse)
        )
    };
}
#[macro_export]
macro_rules! expr_compare {
    ($lhs:expr, $comparison_operator:ident, $rhs:expr) => {
        Expression::BinaryOperation(
            Box::new($lhs),
            BinaryOperation::Comp($crate::math::operations::Comparison::$comparison_operator, None),
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_add {
    ($lhs:expr, $rhs:expr) => {
        Expression::BinaryOperation(
            Box::new($lhs),
            BinaryOperation::Add,
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_sub {
    ($lhs:expr, $rhs:expr) => {
        Expression::BinaryOperation(
            Box::new($lhs),
            BinaryOperation::Sub,
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_mul {
    ($lhs:expr, $rhs:expr) => {
        Expression::BinaryOperation(
            Box::new($lhs),
            BinaryOperation::Mul,
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_div {
    ($lhs:expr, $rhs:expr) => {
        Expression::BinaryOperation(
            Box::new($lhs),
            BinaryOperation::Div,
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_pow {
    ($lhs:expr, $rhs:expr) => {
        Expression::BinaryOperation(
            Box::new($lhs),
            BinaryOperation::Pow(true),
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_inv {
    ($rhs:expr) => {
        Expression::BinaryOperation(
            Box::new(Expression::Number(1.0)),
            BinaryOperation::Div,
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_square {
    ($lhs:expr) => {
        Expression::BinaryOperation(
            Box::new($lhs),
            BinaryOperation::Pow(true),
            Box::new(Expression::Number(2.0))
        )
    };
}
#[macro_export]
macro_rules! expr_and {
    ($lhs:expr, $rhs:expr) => {
        Expression::BinaryOperation(
            Box::new($lhs),
            BinaryOperation::And,
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_neg {
    ($rhs:expr) => {
        Expression::UnaryOperation(
            UnaryOperation::Neg,
            Box::new($rhs)
        )
    };
}
#[macro_export]
macro_rules! expr_1arg_func {
    ($name:expr, $arg:expr) => {
        Expression::Function(
            $name.to_string(),
            vec![$arg]
        )
    };
}
