use std::fmt;
use std::collections::{HashMap, HashSet};

use itertools::Itertools;

use crate::{expr_binop, expr_binop_from_enum};
use crate::math::{Env, Object, objects::ObjType, VarStack};
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
    /// Compute the directional derivative of `SecondArg` at point `ThirdArg` in direction `FourthArg` where the variables w.r.t. which we differentiate are `FirstArg`.
    DirectionalDerivative(Vec<String>, Box<Expression>, Vec<Expression>, Vec<Expression>),
    /// E.g. `int_a^b f(x) dx` gives `Integral(f(x), a, b, x)`.
    Integral(Box<Expression>, Box<Expression>, Box<Expression>, String),
    /// `if (FirstArg) { SecondArg } else { ThirdArg }`
    IfElse(Box<Expression>, Box<Expression>, Box<Expression>)
}

impl Default for Expression {
    fn default() -> Self {
        Expression::None
    }
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
                    UnaryOperation::Norm(opt) => write!(f, "||{}||{}", r, unary_operations::format_optional_subscript(opt)),
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

/// Constructs a match statement that calls the given function recursively on all patterns for which no behavior is specified.
/// 
/// Ignores the return value of the given function; simply calls it on every contained sub-expression.
/// 
/// See `impl Expression` below for examples.
macro_rules! fill_match {
    ($self:expr; $iter:ident; $name:ident($($args:expr),*); $( $variant:ident($($b:pat),*) $(if let $lpat:pat = $lexpr:expr,)? $(if $guard:expr)? => $body:expr ),+ $(,)?) => {
        match $self {
            $( Expression::$variant($($b),*) $(if $guard)? $(if let $lpat = $lexpr)? => $body )+
            #[allow(unreachable_patterns)]
            other => fill_match!(@default other; $iter; $name($($args),*)),
        }
    };

    (@default $self:expr; $iter:ident; $name:ident($($args:expr),*)) => {
        match $self {
            // Generally, do not process LHS of assignment
            Expression::UnaryOperation(_, x) | Expression::PartialDerivative(_, x) | Expression::Assignment(_, x) => x.$name($($args),*),
            Expression::BinaryOperation(x, _, y) => {
                x.$name($($args),*);
                y.$name($($args),*);
            }
            Expression::Integral(x, y, z, _) | Expression::IfElse(x, y, z) => {
                x.$name($($args),*);
                y.$name($($args),*);
                z.$name($($args),*);
            }
            Expression::Vector(v) | Expression::Matrix(.., v) | Expression::Function(_, v) | Expression::Tuple(v)
                => v.$iter().for_each(|x| x.$name($($args),*)),
            Expression::DirectionalDerivative(_, x, v, w) => {
                x.$name($($args),*);
                v.$iter().for_each(|y| y.$name($($args),*));
                w.$iter().for_each(|y| y.$name($($args),*));
            }
            Expression::FoldedOperation(.., x, v, y, z) => {
                x.$name($($args),*);
                y.$name($($args),*);
                z.$name($($args),*);
                v.$iter().for_each(|u| u.$name($($args),*));
            }
            _ => {}
        }
    };
}

impl Expression {
    /// Adds all identifiers `x` for which `x := ...` appears in `self` to `vars`.
    pub fn get_assigned_to_variables<'a>(&'a self, vars: &mut HashSet<&'a String>) {
        fill_match!(
            self; iter; get_assigned_to_variables(vars);
            Assignment(lhs, rhs) => {
                if let Expression::Identifier(x) = &**lhs {vars.insert(x);}
                rhs.get_assigned_to_variables(vars); // Always call recursively on the RHS
            }
        )
    }

    /// Parses itself recursively and replaces every encountered `ident` by `by`.
    /// 
    /// Ignores the LHS of assignment operators and occurrences of `ident` that would be shadowed
    /// in an evaluation (e.g. within integrals where the integration variable is exactly `ident`).
    pub fn replace_identifiers_in_place(&mut self, ident: &String, by: &Expression) {
        fill_match!(
            self; iter_mut; replace_identifiers_in_place(ident, by);
            Identifier(x) if x == ident => {
                *self = by.clone();
            },
            // In the following cases, do not parse `inner` because `wrt` will shadow `ident` in an evaluation
            FoldedOperation(_, wrt, from, conditions, to, inner) if wrt == ident => {
                // Here, do not parse `to` and `conditions` either.
                from.replace_identifiers_in_place(ident, by);
            },
            Integral(from, to, inner, wrt) if wrt == ident => {
                from.replace_identifiers_in_place(ident, by);
                to.replace_identifiers_in_place(ident, by);
            },
            PartialDerivative(wrt, inner) if wrt == ident => {},
            DirectionalDerivative(vars, inner, point, direction) if vars.contains(ident) => {
                point.iter_mut().for_each(|u| u.replace_identifiers_in_place(ident, by));
                direction.iter_mut().for_each(|u| u.replace_identifiers_in_place(ident, by));
            }
        )
    }

    /// Parses the expression `expr` recursively and collects all identifiers that are neither in
    /// `constants` nor in `extra_vars` nor bound by an outer expression (e.g. as the integration
    /// variable of an integral) into a HashSet `modified_identifiers`.
    /// 
    /// Ignores the LHS of assignment operators.
    pub fn list_unknown_identifiers(
        &self,
        extra_vars: &VarStack,
        env: &Env,
        modified_identifiers: &mut HashSet<String>
    ) {
        fill_match!(
            self; iter; list_unknown_identifiers(extra_vars, env, modified_identifiers);
            Identifier(x) => {
                if !env.constants.contains_key(x) && extra_vars.lookup(x).is_none() {
                    modified_identifiers.insert(x.clone());
                }
            },
            FoldedOperation(_, varname, from, conditions, to, inner) => {
                // Important: `varname` is no longer unknown within `conditions`, `inner` and `to`; however, it is still unknown within `from`.
                let varstack = VarStack::Frame { // Varstack where `varname` is declared as known
                    vars: &HashMap::from([(varname, &Object::Success)]),
                    parent: extra_vars
                };
                from.list_unknown_identifiers(extra_vars, env, modified_identifiers); // Here, use old `extra_vars`
                conditions.iter().for_each(|v| v.list_unknown_identifiers(&varstack, env, modified_identifiers));
                to.list_unknown_identifiers(&varstack, env, modified_identifiers); // Here too
                inner.list_unknown_identifiers(&varstack, env, modified_identifiers);
            },
            PartialDerivative(wrt, expr) => {
                // Same as above
                expr.list_unknown_identifiers(
                    &VarStack::Frame {
                        vars: &HashMap::from([(wrt, &Object::Success)]),
                        parent: extra_vars
                    },
                    env,
                    modified_identifiers
                )
            },
            DirectionalDerivative(vars, expr, point, direction) => {
                // Same again
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
            },
            Integral(func, a, b, wrt) => {
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
        )
    }

    /// Adds all identifiers among the given ones that appear in `self` to `contained_identifiers`.
    /// 
    /// Ignores the LHS of assignment operators and occurrences of an element `ident` of `identifiers` that would
    /// be shadowed in an evaluation (e.g. within integrals where the integration variable is exactly `ident`).
    pub fn add_contained_identifiers<'a>(&'a self, identifiers: &HashSet<&String>, contained_identifiers: &mut HashSet<&'a String>) {
        // Note: since in Rust, `impl<T: Hash> Hash for &T` just calls `T::hash` on the pointee, using a HashSet
        // is not only functionally correct but even more efficient than using a `Vec`.
        fill_match!(
            self; iter; add_contained_identifiers(identifiers, contained_identifiers);
            Identifier(x) if identifiers.contains(x) => {contained_identifiers.insert(x);},
            Integral(func, a, b, wrt) => {
                a.add_contained_identifiers(identifiers, contained_identifiers);
                b.add_contained_identifiers(identifiers, contained_identifiers);
                // Ignore presence of `identifier` in `func` if we integrate w.r.t. `identifier`
                let was_contained = contained_identifiers.contains(wrt);
                func.add_contained_identifiers(identifiers, contained_identifiers);
                if !was_contained {contained_identifiers.remove(wrt);}
            },
            FoldedOperation(_, wrt, from, conditions, to, inner) => {
                from.add_contained_identifiers(identifiers, contained_identifiers);
                let was_contained = contained_identifiers.contains(wrt);
                conditions.iter().for_each(|c| c.add_contained_identifiers(identifiers, contained_identifiers));
                to.add_contained_identifiers(identifiers, contained_identifiers);
                inner.add_contained_identifiers(identifiers, contained_identifiers);
                if !was_contained {contained_identifiers.remove(wrt);}
            },
            PartialDerivative(wrt, inner) => {
                let was_contained = contained_identifiers.contains(wrt);
                inner.add_contained_identifiers(identifiers, contained_identifiers);
                if !was_contained {contained_identifiers.remove(wrt);}
            },
            DirectionalDerivative(vars, inner, point, direction) => {
                point.iter().for_each(|v| v.add_contained_identifiers(identifiers, contained_identifiers));
                direction.iter().for_each(|v| v.add_contained_identifiers(identifiers, contained_identifiers));
                let not_previously_contained = vars.iter().filter(|var| !identifiers.contains(var)).collect::<Vec<&String>>();
                inner.add_contained_identifiers(identifiers, contained_identifiers);
                for var in not_previously_contained {
                    contained_identifiers.remove(var);
                }
            }
        )
    }
}

/// Constructs a match statement that calls the given function recursively on all patterns for which no behavior is specified.
/// 
/// Expects the given function to return a boolean and, in case multiple sub-expressions are contained in an expression,
/// returns the disjunction of the returned booleans.
/// 
/// See `impl Expression` below for examples.
macro_rules! fill_match_bool {
    ($self:expr; $name:ident($($args:expr),*); $( $variant:ident($($b:pat),*) $(if let $lpat:pat = $lexpr:expr,)? $(if $guard:expr)? => $body:expr ),+ $(,)?) => {
        match $self {
            $( Expression::$variant($($b),*) $(if $guard)? $(if let $lpat = $lexpr)? => $body, )+
            #[allow(unreachable_patterns)]
            other => fill_match_bool!(@default other; $name($($args),*)),
        }
    };

    (@default $self:expr; $name:ident($($args:expr),*)) => {
        match $self {
            // Generally, do not process LHS of assignment
            Expression::UnaryOperation(_, x) | Expression::PartialDerivative(_, x) | Expression::Assignment(_, x) => x.$name($($args),*),
            Expression::BinaryOperation(x, _, y) => {
                x.$name($($args),*)
                || y.$name($($args),*)
            }
            Expression::Integral(x, y, z, _) | Expression::IfElse(x, y, z) => {
                x.$name($($args),*)
                || y.$name($($args),*)
                || z.$name($($args),*)
            }
            Expression::Vector(v) | Expression::Matrix(.., v) | Expression::Function(_, v) | Expression::Tuple(v)
                => v.iter().any(|x| x.$name($($args),*)),
            Expression::DirectionalDerivative(_, x, v, w) => {
                x.$name($($args),*)
                || v.iter().any(|y| y.$name($($args),*))
                || w.iter().any(|y| y.$name($($args),*))
            }
            Expression::FoldedOperation(.., x, v, y, z) => {
                x.$name($($args),*)
                || y.$name($($args),*)
                || z.$name($($args),*)
                || v.iter().any(|u| u.$name($($args),*))
            }
            Expression::None | Expression::Identifier(_) | Expression::Number(_) => false // Default
        }
    };
}

impl Expression {
    /// Returns whether or not `self` contains an assignment operator.
    pub fn includes_assignment(&self) -> bool {
        fill_match_bool!(
            self; includes_assignment();
            Assignment(..) => true
        )
    }

    /// Returns whether or not `self` contains the given identifier.
    /// 
    /// Ignores the LHS of assignment operators and occurrences of `ident` that would be shadowed
    /// in an evaluation (e.g. within integrals where the integration variable is exactly `ident`).
    pub fn contains_identifier(&self, ident: &String) -> bool {
        fill_match_bool!(
            self; contains_identifier(ident);
            Identifier(x) => x == ident,
            FoldedOperation(_, wrt, from, ..) if wrt == ident => from.contains_identifier(ident),
            Integral(_, from, to, wrt) if wrt == ident => {
                from.contains_identifier(ident) || to.contains_identifier(ident)
            },
            PartialDerivative(wrt, _) if wrt == ident => false,
            DirectionalDerivative(vars, _, point, direction) if vars.contains(ident) => {
                point.iter().any(|x| x.contains_identifier(ident))
                || direction.iter().any(|x| x.contains_identifier(ident))
            }
        )
    }

    /// Returns whether `self` contains any identifier from `identifiers`.
    /// 
    /// Ignores the LHS of assignment operators and occurrences of an element `ident` of `identifiers` that would
    /// be shadowed in an evaluation (e.g. within integrals where the integration variable is exactly `ident`).
    pub fn contains_any_of(&self, identifiers: &HashSet<&String>) -> bool {
        fill_match_bool!(
            self; contains_any_of(identifiers);
            Identifier(x) => identifiers.contains(x),
            // Typically, we search for less identifiers than will realistically occur in an expression.
            // More precisely, if we search for `n` identifiers, then it is reasonable to assume that `self`
            // could have even more than `n` identifiers in it. Therefore, in the following cases, it is generally
            // cheaper to copy the hashset `identifiers`, remove `wrt` and call `.contains_identifer(new_hashset)`
            // (the hashset only contains references anyway) than to call `.add_contained_identifiers`
            // and check if an element of `identifiers` other than `wrt` is in the result.
            FoldedOperation(_, wrt, from, conditions, to, inner) if identifiers.contains(wrt) => {
                let mut new_hashset = identifiers.clone();
                new_hashset.remove(wrt);
                from.contains_any_of(identifiers)
                || conditions.iter().any(|x| x.contains_any_of(&new_hashset))
                || to.contains_any_of(&new_hashset)
                || inner.contains_any_of(&new_hashset)
            },
            Integral(inner, from, to, wrt) if identifiers.contains(wrt) => {
                let mut new_hashset = identifiers.clone();
                new_hashset.remove(wrt);
                from.contains_any_of(identifiers)
                || to.contains_any_of(identifiers)
                || inner.contains_any_of(&new_hashset)
            },
            PartialDerivative(wrt, inner) if identifiers.contains(wrt) => {
                let mut new_hashset = identifiers.clone();
                new_hashset.remove(wrt);
                inner.contains_any_of(&new_hashset)
            },
            DirectionalDerivative(vars, inner, point, direction) if vars.iter().any(|var| identifiers.contains(var)) => {
                let mut new_hashset = identifiers.clone();
                for var in vars {
                    new_hashset.remove(var);
                }
                point.iter().any(|x| x.contains_any_of(identifiers))
                || direction.iter().any(|x| x.contains_any_of(identifiers))
                || inner.contains_any_of(&new_hashset)
            }
        )
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

    pub fn expect_ident(&self) -> Result<&String, String> {
        match self {
            Expression::Identifier(id) => Ok(id),
            other => Err(format!("Expected identifier, found `{:?}`.", other))
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
                let index_type_repr = from.get_type(extra_vars, env)?.representative();
                inner.get_type(&VarStack::Frame { vars: &HashMap::from([(index_var_name, &index_type_repr)]), parent: extra_vars }, env)
            }
            Expression::Function(name, args) => {
                match env.functions.get(name) {
                    Some(super::FunctionRepr::ByExpression(varnames, expr)) => {
                        let h = varnames.iter().zip(args)
                            .map(|(v, a)| a.get_type(extra_vars, env).map(|t| (v, t.representative())))
                            .collect::<Result<HashMap<_, _>, _>>()?;
                        expr.get_type(
                            &VarStack::Frame {
                                vars: &h.iter().map(|(v, r)| (*v, r)).collect(),
                                parent: extra_vars
                            },
                            env
                        )
                    }
                    Some(super::FunctionRepr::Direct(_, (m, n, b))) => {
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
                let h = vars.iter().zip(point)
                    .map(|(v, a)| a.get_type(extra_vars, env).map(|t| (v, t.representative())))
                    .collect::<Result<HashMap<_, _>, _>>()?;
                expr.get_type(
                    &VarStack::Frame {
                        vars: &h.iter().map(|(v, r)| (*v, r)).collect(),
                        parent: extra_vars
                    },
                    env
                )
            }
            Expression::Integral(func, .., wrt) => {
                // This time, we can assume truly w.l.o.g. that `func` always returns the same type,
                // otherwise the integral wouldn't be defined.
                // The integration variable has to be real.
                func.get_type(&VarStack::Frame { vars: &HashMap::from([(wrt, &Object::Real(1.0))]), parent: extra_vars }, env)
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
                    &VarStack::Frame { vars: &HashMap::from([(index_var_name, &Object::Real(1.0))]), parent: extra_vars },
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
                    &VarStack::Frame { vars: &HashMap::from([(index_var_name, &Object::Real(1.0))]), parent: extra_vars },
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
                    Some(super::FunctionRepr::ByExpression(varnames, defining_expr)) => {
                        // If this is a `FunctionRepr::ByExpression`:
                        // Idea: make the type of `defining_expr` top-level and then simply replace `f(x)` by `defining_expr`.
                        // Evidently, this requires us to replace `varnames` within `defining_expr` by the corresponding given argument in `args`.
                        let h = varnames.iter().zip(args)
                            .map(|(v, a)| a.get_type(extra_vars, env).map(|t| (v, t.representative())))
                            .collect::<Result<HashMap<_, _>, _>>()?;
                        let (mut iexpr, itype) = defining_expr.make_type_top_level(
                            &VarStack::Frame {
                                vars: &h.iter().map(|(v, r)| (*v, r)).collect(),
                                parent: extra_vars
                            },
                            env
                        )?;
                        for (varname, arg) in varnames.iter().zip(args) {
                            iexpr.replace_identifiers_in_place(varname, arg);
                        }
                        Ok((iexpr, itype))
                    }
                    Some(super::FunctionRepr::Direct(_, (m, n, b))) => {
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
                    &VarStack::Frame { vars: &HashMap::from([(wrt, &Object::Real(1.0))]), parent: extra_vars },
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
                let h = vars.iter().zip(point)
                    .map(|(v, a)| a.get_type(extra_vars, env).map(|t| (v, t.representative())))
                    .collect::<Result<HashMap<_, _>, _>>()?;
                let varstack = VarStack::Frame {
                    vars: &h.iter().map(|(v, r)| (*v, r)).collect(),
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
                    &VarStack::Frame { vars: &HashMap::from([(wrt, &Object::Real(1.0))]), parent: extra_vars },
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


impl Expression {
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

    /// Returns the set of all identifiers among the given ones that appear in `self` to `contained_identifiers`.
    pub fn get_contained_identifiers<'a>(&'a self, identifiers: &HashSet<&String>) -> HashSet<&'a String> {
        let mut set = HashSet::new();
        self.add_contained_identifiers(identifiers, &mut set);
        set
    }

    /// Returns the first identifier of the form `prefixNumber` (e.g. `x2` if `prefix` is `x`) which
    /// is contained nowhere inside `self`.
    /// 
    /// This can be used to create an `Expression::Integral` for which
    /// the integration variable doesn't clash with any variable inside the integrand.
    pub fn get_new_free_identifier(&self, prefix: &str) -> String {
        format!("{}{}", prefix, self.get_new_free_identifier_recursive(prefix, 0))
    }
    /// Returns the first identifier of the form `prefixNumber` (e.g. `x2` if `prefix` is `x`) which
    /// is contained in none of the given expressions.
    /// 
    /// This can be used to create an `Expression::Integral` for which
    /// the integration variable doesn't clash with any variable inside the integrand.
    pub fn get_new_free_identifier_in_none_of(prefix: &str, exprs: &[Expression]) -> String {
        format!("{}{}", prefix, exprs.iter().map(|e| e.get_new_free_identifier_recursive(prefix, 0)).max().unwrap_or(0))
    }
    /// Returns an integer `j` (not necessarily the smallest one) such that `{prefix}{j}` is not contained in `self`.
    /// Returning the smallest one is not very useful for `get_new_free_identifier` but would increase computation time.
    /// 
    /// If `{prefix}{i}` is not contained in `self` (for the given parameter `i`), then `i` is returned as is.
    fn get_new_free_identifier_recursive(&self, prefix: &str, i: usize) -> usize {
        // The below function `check_id` does the following.
        // If `id` is of the form `{prefix}{j}` for some `j >= i`, return `j+1`, otherwise `i`.
        // This ensures that whenever we reach the end of the expression `self`, the integer this function returns is contained nowhere.
        let check_id = |id: &String| {
            if let Some(suffix) = id.strip_prefix(prefix) && let Ok(j) = suffix.parse::<usize>() && j >= i {
                j+1
            } else {
                i
            }
        };
        match self {
            Expression::None | Expression::Number(_) => i, // `i` is still valid then
            Expression::Identifier(id) => check_id(id),
            Expression::Tuple(v) | Expression::Vector(v) | Expression::Matrix(.., v) =>
                v.iter()
                .map(|e: &Expression| e.get_new_free_identifier_recursive(prefix, i))
                .max()
                .unwrap_or(i),
            Expression::UnaryOperation(_, expr) | Expression::Assignment(_, expr) | Expression::PartialDerivative(_, expr) =>
                expr.get_new_free_identifier_recursive(prefix, i),
            Expression::BinaryOperation(lhs, _, rhs) =>
                lhs.get_new_free_identifier_recursive(prefix, i)
                .max(rhs.get_new_free_identifier_recursive(prefix, i)),
            Expression::FoldedOperation(_, loop_var, from, conditions, to, inner) =>
                from.get_new_free_identifier_recursive(prefix, i)
                .max(conditions.iter().map(|c| c.get_new_free_identifier_recursive(prefix, i)).max().unwrap_or(i))
                .max(to.get_new_free_identifier_recursive(prefix, i))
                .max(inner.get_new_free_identifier_recursive(prefix, i))
                .max(check_id(loop_var)),
            Expression::Function(name, args) =>
                args.iter().map(|arg| arg.get_new_free_identifier_recursive(prefix, i)).max().unwrap_or(i)
                .max(check_id(name)),
            Expression::DirectionalDerivative(vars, expr, point, direction) =>
                expr.get_new_free_identifier_recursive(prefix, i)
                .max(point.iter().map(|v| v.get_new_free_identifier_recursive(prefix, i)).max().unwrap_or(i))
                .max(direction.iter().map(|v| v.get_new_free_identifier_recursive(prefix, i)).max().unwrap_or(i))
                .max(vars.iter().map(check_id).max().unwrap_or(i)),
            Expression::IfElse(x, y, z) =>
                x.get_new_free_identifier_recursive(prefix, i)
                .max(y.get_new_free_identifier_recursive(prefix, i))
                .max(z.get_new_free_identifier_recursive(prefix, i)),
            Expression::Integral(func, a, b, wrt) =>
                a.get_new_free_identifier_recursive(prefix, i)
                .max(b.get_new_free_identifier_recursive(prefix, i))
                .max(func.get_new_free_identifier_recursive(prefix, i))
                .max(check_id(wrt))
        }
    }
}


// The following macros simplify typing and enhance readability by a LOT.
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
