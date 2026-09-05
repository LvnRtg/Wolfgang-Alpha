//! Contains functions that obtain basic information from an expression by traversing it.
//! These functions are generally quite simple, only the recursive structure can make them somewhat long.

use std::borrow::Cow;
use std::collections::HashSet;

use crate::math::{Env, Object, VarStack, VarStackLookup};
use super::Expression;

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
                let varstack = extra_vars.with(varname, Cow::Owned(Object::Success)); // Varstack where `varname` is declared as known
                from.list_unknown_identifiers(extra_vars, env, modified_identifiers); // Here, use old `extra_vars`
                conditions.iter().for_each(|v| v.list_unknown_identifiers(&varstack, env, modified_identifiers));
                to.list_unknown_identifiers(&varstack, env, modified_identifiers); // Here too
                inner.list_unknown_identifiers(&varstack, env, modified_identifiers);
            },
            PartialDerivative(wrt, expr) => {
                // Same as above
                expr.list_unknown_identifiers(
                    &extra_vars.with(wrt, Cow::Owned(Object::Success)),
                    env,
                    modified_identifiers
                )
            },
            DirectionalDerivative(vars, expr, point, direction) => {
                // Same again
                expr.list_unknown_identifiers(
                    &extra_vars.with_multiple(vars.iter(), std::iter::repeat_n(&Object::Success, vars.len())),
                    env,
                    modified_identifiers
                );
                point.iter().for_each(|v| v.list_unknown_identifiers(extra_vars, env, modified_identifiers));
                direction.iter().for_each(|v| v.list_unknown_identifiers(extra_vars, env, modified_identifiers));
            },
            Integral(func, a, b, wrt) => {
                func.list_unknown_identifiers(
                    &extra_vars.with(wrt, Cow::Owned(Object::Success)),
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

    /// Returns the set of all identifiers among the given ones that appear in `self` to `contained_identifiers`.
    pub fn get_contained_identifiers<'a>(&'a self, identifiers: &HashSet<&String>) -> HashSet<&'a String> {
        let mut set = HashSet::new();
        self.add_contained_identifiers(identifiers, &mut set);
        set
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
}

impl Expression {
    /// Returns the first identifier of the form `prefixNumber` (e.g. `x2` if `prefix` is `x`) which
    /// is contained nowhere inside `self`.
    /// 
    /// This can be used to create an `Expression::Integral` for which
    /// the integration variable doesn't clash with any variable inside the integrand.
    pub fn get_new_free_identifier(&self, prefix: &str) -> String {
        let i = self.get_new_free_identifier_recursive(prefix, 0);
        if i == 0 {
            prefix.to_string()
        } else {
            format!("{}{}", prefix, i)
        }
    }
    /// Returns the first identifier of the form `prefixNumber` (e.g. `x2` if `prefix` is `x`) which
    /// is contained in none of the given expressions.
    /// 
    /// This can be used to create an `Expression::Integral` for which
    /// the integration variable doesn't clash with any variable inside the integrand.
    pub fn get_new_free_identifier_in_none_of<'a>(prefix: String, exprs: impl Iterator<Item = &'a Expression>) -> String {
        let i = exprs.map(|e| e.get_new_free_identifier_recursive(&prefix, 0)).max().unwrap_or(0);
        if i == 0 {
            prefix
        } else {
            format!("{}{}", prefix, i)
        }
    }
    /// Returns an integer `j` (not necessarily the smallest one) such that `{prefix}{j}` is not contained in `self`.
    /// Returning the smallest one is not very useful for `get_new_free_identifier` but would increase computation time.
    /// 
    /// Returns `0` iff `prefix` itself is not contained in `self`.
    /// 
    /// If `{prefix}{i}` is not contained in `self` (for the given parameter `i`), then `i` is returned as is.
    fn get_new_free_identifier_recursive(&self, prefix: &str, i: usize) -> usize {
        // The below function `check_id` does the following.
        // If `id` is of the form `{prefix}{j}` for some `j >= i`, return `j+1`, otherwise `i`.
        // This ensures that whenever we reach the end of the expression `self`, the integer this function returns is contained nowhere.
        let check_id = |id: &String| {
            if let Some(suffix) = id.strip_prefix(prefix) {
                if suffix.is_empty() { // `id == prefix`
                    1
                } else if let Ok(j) = suffix.parse::<usize>() && j >= i { // `id == {prefix}{j}`
                    j+1
                } else {
                    i
                }
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