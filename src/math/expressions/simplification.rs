//! Contains functions that simplify expressions.
//! 
//! Importantly, these assume commutativity of multiplication, so they are not generally
//! valid for matrices. For instance, `xy + yx` would be simplified to `2xy`, which is
//! not necessarily the same if `x`, `y` are matrices.
//! 
//! However, there is a way to circumvent this; simply wrap any expression containing
//! matrices in `Decomposable::Indecomposable(expr)`. This shields the expression from
//! any further decomposition, but will still allow simplifications like `xy + xy`
//! -> `2xy`. Under the hood, any expression ran through `Decomposable::from` will
//! be decomposed into individual terms/factors, so e.g. `xy` will become `Product(x, y)`.
//! Two products, for example, are fused together iff their factors are identical
//! up to reordering. Having the matrix expression wrapped in `Indecomposable` treats
//! it as a single factor, so factorization will only take place for expressions
//! that match exactly.
//! 
//! It is always preferred to factor out a negation sign, e.g. `-(x*y)` is preferred
//! over `(-x) * y` and `x * (-y)`; moreover, `-(x+y)` is preferred over `(-x) - y`
//! and `(-x) + (-y)`.
//! 
//! Note on design choice: all of these functions do not return errors they may find
//! in the structure of expressions. For instance, the function `simplify_add` might
//! find both sides to be vectors of different sizes, which makes the operation `+`
//! invalid. However, since these functions don't find _all_ errors (unless extra
//! code is added) and their intended purpose is not the static analysis of
//! expressions, I decided they should not report any errors at all.

use paste::paste;
use std::borrow::Cow;

use crate::math::utils;
use crate::{defaults, expr_binop, expr_binop_from_enum, expr_binop_from_iter, expr_if_else, expr_unary_op};
use crate::math::operations::{BinaryOperation, Comparison, FoldedOperation, UnaryOperation};
use super::Expression;


#[derive(Debug, Clone, PartialEq)]
pub struct Product {
    /// `+1` or `-1`
    sign: f64,
    leading_float: f64,
    /// Expressions in the numerator
    num: Vec<Decomposed>,
    /// Expressions in the denominator
    denom: Vec<Decomposed>
}
impl Product {
    #[inline]
    pub fn unscaled(num: Vec<Decomposed>, denom: Vec<Decomposed>) -> Self {
        Product {
            sign: 1.0,
            leading_float: 1.0,
            num,
            denom
        }
    }

    pub fn from_f64(x: f64) -> Self {
        if x >= 0.0 {
            Product {
                sign: 1.0,
                leading_float: x,
                num: Vec::new(),
                denom: Vec::new()
            }
        } else {
            Product {
                sign: -1.0,
                leading_float: -x,
                num: Vec::new(),
                denom: Vec::new()
            }
        }
    }

    pub fn to_expr(self) -> Expression {
        let unsigned = match (self.leading_float == 1.0, self.num.is_empty(), self.denom.is_empty()) {
            (_, true, true) => return Expression::Number(self.sign * self.leading_float), // Here, the sign is applied directly
            (_, true, false) => expr_binop!(
                Expression::Number(self.leading_float),
                Div,
                expr_binop_from_iter!(Mul, Product, self.denom.into_iter().map(|f| f.to_expr()))
            ),
            (true, false, true) => expr_binop_from_iter!(Mul, Product, self.num.into_iter().map(|f| f.to_expr())),
            (false, false, true) => expr_binop!(
                Expression::Number(self.leading_float),
                Mul,
                expr_binop_from_iter!(Mul, Product, self.num.into_iter().map(|f| f.to_expr()))
            ),
            (true, false, false) => expr_binop!(
                expr_binop_from_iter!(Mul, Product, self.num.into_iter().map(|f| f.to_expr())),
                Div,
                expr_binop_from_iter!(Mul, Product, self.denom.into_iter().map(|f| f.to_expr()))
            ),
            (false, false, false) => expr_binop!(
                expr_binop!(
                    Expression::Number(self.leading_float),
                    Mul,
                    expr_binop_from_iter!(Mul, Product, self.num.into_iter().map(|f| f.to_expr()))
                ),
                Div,
                expr_binop_from_iter!(Mul, Product, self.denom.into_iter().map(|f| f.to_expr()))
            )
        };
        if self.sign == 1.0 {
            unsigned
        } else {
            expr_unary_op!(Neg, unsigned)
        }
    }

    #[inline]
    pub fn is_neg(&self) -> bool {
        self.sign == -1.0
    }

    pub fn with_sign(mut self, sign: f64) -> Self {
        self.sign = sign;
        self
    }
    pub fn without_scaling(mut self) -> Self {
        self.sign = 1.0;
        self.leading_float = 1.0;
        self
    }
    pub fn negated(mut self) -> Self {
        self.sign = -self.sign;
        self
    }
}

/// All binary operations for which it is relevant to decompose them in some way
/// for simplification purposes. This excludes binary operations like `+` and `*`
/// for which it makes sense to collect all operands of a chain like `x + y + z`
/// in a single vector instead of keeping the nested structure `(x+y) + z`.
#[derive(Debug, Clone, PartialEq)]
pub enum DecomposableBinop {
    Pow,
    Comp(Comparison),
    And,
    Or
}
macro_rules! decomp_binop {
    ($lhs:expr, $op:ident$(($($param:expr),*))?, $rhs:expr) => {
        Decomposed::BinaryOperation(
            Box::new($lhs),
            DecomposableBinop::$op$(($($param),*))?,
            Box::new($rhs)
        )
    };
}
/// The unary pendant of `DecomposableBinop`.
#[derive(Debug, Clone, PartialEq)]
pub enum DecomposableUnaryOp {
    Not,
    Abs
}
macro_rules! decomp_unop {
    ($op:ident$(($($param:expr),*))?, $rhs:expr) => {
        Decomposed::UnaryOperation(
            DecomposableUnaryOp::$op$(($($param),*))?,
            Box::new($rhs)
        )
    };
}

macro_rules! decomp_if_else {
    ($a:expr, $b:expr, $c:expr) => {
        Decomposed::IfElse(
            Box::new($a),
            Box::new($b),
            Box::new($c)
        )
    };
}

#[derive(Debug, Clone, PartialEq)]
pub enum Decomposed {
    Indecomposable(Expression),
    /// `Sum(v, ops)` is to be interpreted as `op[0] v[0] op[1] v[1] ... op[n] v[n]`
    /// where `op[i] = true` signifies `+` and `false` signifies `-`.
    Sum(Vec<Decomposed>, Vec<bool>),
    Product(Product),
    BinaryOperation(Box<Decomposed>, DecomposableBinop, Box<Decomposed>),
    UnaryOperation(DecomposableUnaryOp, Box<Decomposed>),
    Vector(Vec<Decomposed>),
    Matrix(usize, usize, Vec<Decomposed>),
    Function(String, Vec<Decomposed>),
    IfElse(Box<Decomposed>, Box<Decomposed>, Box<Decomposed>)
}

impl Default for Decomposed {
    fn default() -> Self {
        Decomposed::Indecomposable(Expression::None)
    }
}

impl Decomposed {
    /// Parses the given expression top-down and decomposes it into individual factors.
    /// Does _not_ simplify anything while decomposing. To simplify, call `simplify` instead.
    pub fn from(expr: Expression) -> Self {
        match expr {
            Expression::Number(x) => Decomposed::from_f64(x),
            Expression::BinaryOperation(lhs, BinaryOperation::Add, rhs) => {
                let (mut res, mut ops) = Decomposed::from(*lhs).as_sum();
                let (r_terms, r_ops) = Decomposed::from(*rhs).as_sum();
                res.extend(r_terms);
                ops.extend(r_ops);
                Decomposed::Sum(res, ops)
            }
            Expression::BinaryOperation(lhs, BinaryOperation::Sub, rhs) => {
                let (mut res, mut ops) = Decomposed::from(*lhs).as_sum();
                let (r_terms, r_ops) = Decomposed::from(*rhs).as_sum();
                res.extend(r_terms);
                ops.extend(r_ops.into_iter().map(|x| !x));
                Decomposed::Sum(res, ops)
            }
            Expression::BinaryOperation(lhs, BinaryOperation::Mul, rhs) => {
                match (Decomposed::from(*lhs), Decomposed::from(*rhs)) {
                    (Decomposed::Product(mut l_p), Decomposed::Product(r_p)) => {
                        l_p.sign *= r_p.sign;
                        l_p.leading_float *= r_p.leading_float;
                        l_p.num.extend(r_p.num);
                        l_p.denom.extend(r_p.denom);
                        Decomposed::Product(l_p)
                    }
                    (Decomposed::Product(mut l_p), other_r) => {
                        l_p.num.push(other_r);
                        Decomposed::Product(l_p)
                    }
                    (other_l, Decomposed::Product(mut r_p)) => {
                        r_p.num.insert(0, other_l);
                        Decomposed::Product(r_p)
                    }
                    (other_l, other_r) => {
                        Decomposed::Product(Product::unscaled(vec![other_l, other_r], Vec::new()))
                    }
                }
            }
            Expression::BinaryOperation(lhs, BinaryOperation::Div, rhs) => {
                match (Decomposed::from(*lhs), Decomposed::from(*rhs)) {
                    (Decomposed::Product(mut l_p), Decomposed::Product(r_p)) => {
                        l_p.sign *= r_p.sign;
                        l_p.leading_float *= r_p.leading_float;
                        l_p.num.extend(r_p.denom);
                        l_p.denom.extend(r_p.num);
                        Decomposed::Product(l_p)
                    }
                    (Decomposed::Product(mut l_p), other_r) => {
                        // Insert `other_r` at the start of the denominator to keep as much compatibility as
                        // possible with non-commutative multiplication:
                        // `(A/B) / C = A B^(-1) C^(-1) = A / (CB)` but `A / (BC) = A (BC)^(-1) = A C^(-1) B^(-1)`
                        l_p.denom.insert(0, other_r);
                        Decomposed::Product(l_p)
                    }
                    (other_l, Decomposed::Product(r_p)) => {
                        // A / (BC/DE) = A * (BC E^(-1) D^(-1))^(-1) = A * DE * C^(-1) B^(-1) = A * DE / BC
                        let mut p = Product {
                            sign: r_p.sign,
                            leading_float: 1.0 / r_p.leading_float,
                            num: r_p.denom,
                            denom: r_p.num
                        };
                        p.num.insert(0, other_l);
                        Decomposed::Product(p)
                    }
                    (other_l, other_r) => {
                        Decomposed::Product(Product::unscaled(vec![other_l], vec![other_r]))
                    }
                }
            }
            Expression::BinaryOperation(lhs, BinaryOperation::Pow(_), rhs) => {
                decomp_binop!(Decomposed::from(*lhs), Pow, Decomposed::from(*rhs))
            }
            Expression::BinaryOperation(lhs, BinaryOperation::Comp(c, None), rhs) => {
                decomp_binop!(Decomposed::from(*lhs), Comp(c), Decomposed::from(*rhs))
            }
            Expression::BinaryOperation(lhs, BinaryOperation::And, rhs) => {
                decomp_binop!(Decomposed::from(*lhs), And, Decomposed::from(*rhs))
            }
            Expression::BinaryOperation(lhs, BinaryOperation::Or, rhs) => {
                decomp_binop!(Decomposed::from(*lhs), Or, Decomposed::from(*rhs))
            }
            Expression::UnaryOperation(UnaryOperation::Not, inner) => {
                decomp_unop!(Not, Decomposed::from(*inner))
            }
            Expression::UnaryOperation(UnaryOperation::Neg, inner) => {
                Decomposed::from(*inner).negated()
            }
            Expression::Vector(v) => Decomposed::Vector(
                v.into_iter().map(|e| Decomposed::from(e)).collect()
            ),
            Expression::Matrix(m, n, v) => Decomposed::Matrix(
                m, n,
                v.into_iter().map(|e| Decomposed::from(e)).collect()
            ),
            Expression::Function(name, args) => Decomposed::Function(
                name,
                args.into_iter().map(|arg| Decomposed::from(arg)).collect()
            ),
            Expression::IfElse(a, b, c) => decomp_if_else!(
                Decomposed::from(*a),
                Decomposed::from(*b),
                Decomposed::from(*c)
            ),
            other => Decomposed::Indecomposable(other)
        }
    }
    /// Creates a `Decomposed` from `expr` without splitting the individual factors of
    /// multiplications/divisions apart. Only checks for potential leading negation signs.
    /// 
    /// This is useful for expressions of which it is known that they don't contain any
    /// multiplications/divisions, as well as for e.g. matrices where we sometimes want
    /// to avoid factorization.
    pub fn from_indecomposable(expr: Expression) -> Self {
        match expr {
            Expression::Number(x) => Decomposed::from_f64(x),
            Expression::UnaryOperation(UnaryOperation::Neg, inner) => {
                Decomposed::Product(Product {
                    sign: -1.0,
                    leading_float: 1.0,
                    num: vec![Decomposed::Indecomposable(*inner)],
                    denom: Vec::new()
                })
            }
            Expression::Vector(v) => Decomposed::Vector(
                v.into_iter().map(|e| Decomposed::from_indecomposable(e)).collect()
            ),
            Expression::Matrix(m, n, v) => Decomposed::Matrix(
                m, n,
                v.into_iter().map(|e| Decomposed::from_indecomposable(e)).collect()
            ),
            other => Decomposed::Indecomposable(other)
        }
    }
    #[inline]
    pub fn from_f64(x: f64) -> Self {
        Decomposed::Product(Product::from_f64(x))
    }
    /// Merges all factors in numerator and denominator together so that methods like
    /// `simplify_add` can be called while avoiding any factorization.
    /// 
    /// This is useful for objects where multiplication is not commutative (e.g. matrices),
    /// because then, factorizations like `xvy + xwy` -> `(v+w) * xy` would be incorrect.
    #[inline]
    pub fn to_indecomposable(self) -> Self {
        match self {
            Decomposed::Product(p) => Decomposed::Product(Product {
                sign: p.sign,
                leading_float: p.leading_float,
                num: vec![Decomposed::Indecomposable(p.without_scaling().to_expr())],
                denom: Vec::new()
            }),
            other => Decomposed::Indecomposable(other.to_expr())
        }
    }
    /// Just to enhance readability, e.g. in `differentiation.rs`.
    #[inline]
    pub fn to_indecomposable_if(self, b: bool) -> Self {
        if b {
            self.to_indecomposable()
        } else {
            self
        }
    }
    /// If `self = Decomposed::Sum(v, op)`, returns `(v, op)`, otherwise `([self], [])`.
    pub fn as_sum(self) -> (Vec<Decomposed>, Vec<bool>) {
        if let Decomposed::Sum(v, ops) = self {
            (v, ops)
        } else if self.is_neg() {
            (vec![self.negated()], vec![false])
        } else {
            (vec![self], vec![true])
        }
    }
    pub fn as_powers(self) -> (f64, Vec<(Decomposed, Decomposed)>) {
        match self {
            Decomposed::BinaryOperation(lhs, DecomposableBinop::Pow, rhs) => (1.0, vec![(*lhs, *rhs)]),
            Decomposed::Product(Product { sign, leading_float, num, denom }) => (
                sign * leading_float,
                {
                    let mut powers = num.into_iter().map(|f| (f, Decomposed::from_f64(1.0))).collect::<Vec<_>>();
                    powers.extend(denom.into_iter().map(|f| (f, Decomposed::from_f64(-1.0))));
                    powers
                }
            ),
            Decomposed::Function(name, v) if name == "sqrt" => (1.0, vec![(v.into_iter().next().unwrap(), Decomposed::from_f64(0.5))]),
            other => (1.0, vec![(other, Decomposed::from_f64(1.0))])
        }
    }

    pub fn to_expr(self) -> Expression {
        match self {
            Decomposed::Indecomposable(e) => e,
            Decomposed::Product(p) => p.to_expr(),
            Decomposed::Sum(terms, ops) => {
                if terms.is_empty() {
                    Expression::Number(1.0)
                } else {
                    let (mut terms, mut ops) = (terms.into_iter(), ops.into_iter());
                    let mut res = if ops.next().unwrap() {
                        terms.next().unwrap().to_expr()
                    } else {
                        terms.next().unwrap().negated().to_expr()
                    };
                    for (mut term, op) in terms.zip(ops) {
                        let neg = if let Decomposed::Product(p) = &mut term {
                            if p.is_neg() {
                                p.sign = 1.0;
                                true
                            } else {
                                false
                            }
                        } else {
                            false
                        };
                        if op ^ neg {
                            res = expr_binop!(res, Add, term.to_expr());
                        } else {
                            res = expr_binop!(res, Sub, term.to_expr());
                        }
                    }
                    res
                }
            },
            Decomposed::BinaryOperation(lhs, DecomposableBinop::Pow, rhs) => expr_binop!(
                lhs.to_expr(),
                Pow(true),
                rhs.to_expr()
            ),
            Decomposed::BinaryOperation(lhs, DecomposableBinop::Comp(c), rhs) => expr_binop!(
                lhs.to_expr(),
                Comp(c, None),
                rhs.to_expr()
            ),
            Decomposed::BinaryOperation(lhs, DecomposableBinop::And, rhs) => expr_binop!(
                lhs.to_expr(),
                And,
                rhs.to_expr()
            ),
            Decomposed::BinaryOperation(lhs, DecomposableBinop::Or, rhs) => expr_binop!(
                lhs.to_expr(),
                Or,
                rhs.to_expr()
            ),
            Decomposed::UnaryOperation(DecomposableUnaryOp::Not, inner) => expr_unary_op!(Not, inner.to_expr()),
            Decomposed::UnaryOperation(DecomposableUnaryOp::Abs, inner) => expr_unary_op!(Abs, inner.to_expr()),
            Decomposed::Vector(v) => Expression::Vector(v.into_iter().map(|f| f.to_expr()).collect()),
            Decomposed::Matrix(m, n, v) => Expression::Matrix(m, n, v.into_iter().map(|f| f.to_expr()).collect()),
            Decomposed::Function(name, args) => Expression::Function(name, args.into_iter().map(|arg| arg.to_expr()).collect()),
            Decomposed::IfElse(a, b, c) => expr_if_else!(a.to_expr(), b.to_expr(), c.to_expr())
        }
    }

    #[inline]
    pub fn is_neg(&self) -> bool {
        if let Decomposed::Product(p) = self && p.is_neg() {
            true
        } else {
            false
        }
    }
    pub fn negated(self) -> Self {
        match self {
            Decomposed::Product(mut p) => {
                p.sign *= -1.0;
                Decomposed::Product(p)
            },
            Decomposed::Sum(v, ops) => Decomposed::Sum(
                v,
                ops.into_iter().map(|x| !x).collect()
            ),
            other => Decomposed::Product(Product {
                sign: -1.0,
                leading_float: 1.0,
                num: vec![other],
                denom: Vec::new()
            })
        }
    }
    pub fn inverted(self) -> Self {
        match self {
            Decomposed::Product(Product { sign, leading_float, num, denom }) => Decomposed::Product(Product {
                sign,
                leading_float: 1.0 / leading_float,
                num: denom,
                denom: num
            }),
            Decomposed::BinaryOperation(base, DecomposableBinop::Pow, pow) => Decomposed::BinaryOperation(
                base,
                DecomposableBinop::Pow,
                Box::new(pow.negated())
            ),
            other => Decomposed::Product(Product {
                sign: 1.0,
                leading_float: 1.0,
                num: Vec::new(),
                denom: vec![other]
            })
        }
    }

    fn try_vec_to_floats(v: &Vec<Decomposed>) -> Option<f64> {
        let mut res = 1.0;
        for d in v.iter() {
            res *= d.try_to_float()?;
        }
        Some(res)
    }
    /// Returns
    /// - `x` iff `self` is `Expression::Number(x)`
    /// - `sign * leading_float` if `self` is a product with empty `num`, `denom`
    /// - `sign * leading_float * prod(num) / prod(denom)` if `self` is a product where `num` and `denom` only contain floats
    /// - `None` otherwise
    pub fn try_to_float(&self) -> Option<f64> {
        if let Decomposed::Indecomposable(Expression::Number(x)) = self {
            Some(*x)
        } else if let Decomposed::Product(
            Product {sign, leading_float, num, denom}
        ) = self {
            if utils::approx_eq(*leading_float, 0.0) {
                // If the leading float is already zero, it doesn't matter what `num` and `denom` contain.
                Some(0.0)
            } else if let Some(num_f) = Decomposed::try_vec_to_floats(num) && let Some(denom_f) = Decomposed::try_vec_to_floats(denom) {
                Some(sign * leading_float * num_f / denom_f)
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn is_vec(&self) -> Option<usize> {
        if let Decomposed::Vector(v) = self {
            Some(v.len())
        } else if let Decomposed::Product(Product { sign: _, leading_float: _, num, denom }) = self
        && num.len() == 1 && denom.is_empty()
        && let Some(n) = num[0].is_vec() {
            Some(n)
        } else {
            None
        }
    }
    pub fn is_mat(&self) -> Option<(usize, usize)> {
        if let Decomposed::Matrix(m, n, _) = self {
            Some((*m, *n))
        } else if let Decomposed::Product(Product { sign: _, leading_float: _, num, denom }) = self
        && num.len() == 1 && denom.is_empty()
        && let Some((m, n)) = num[0].is_mat() {
            Some((m, n))
        } else {
            None
        }
    }
    /// ## Panics
    /// (or returns garbage) if `self` isn't actually a vector in the sense of `is_vec()`.
    pub(crate) fn expect_vec(self) -> Box<dyn Iterator<Item=Decomposed>> {
        match self {
            Decomposed::Vector(v) => Box::new(v.into_iter()),
            Decomposed::Product(Product { sign, leading_float, num, denom: _ }) => Box::new(
                num.into_iter().next().unwrap()
                .expect_vec()
                .map(move |f| Decomposed::Product(Product {
                    sign,
                    leading_float,
                    num: vec![f],
                    denom: Vec::new()
                }))
            ),
            _ => unreachable!()
        }
    }
    /// ## Panics
    /// (or returns garbage) if `self` isn't actually a matrix in the sense of `is_mat()`.
    pub(crate) fn expect_mat(self) -> Box<dyn Iterator<Item=Decomposed>> {
        match self {
            Decomposed::Matrix(.., v) => Box::new(v.into_iter()),
            Decomposed::Product(Product { sign, leading_float, num, denom: _ }) => Box::new(
                num.into_iter().next().unwrap()
                .expect_mat()
                .map(move |f| Decomposed::Product(Product {
                    sign,
                    leading_float,
                    num: vec![f],
                    denom: Vec::new()
                }))
            ),
            _ => unreachable!()
        }
    }

    pub fn leading_float(&self) -> f64 {
        match self {
            Decomposed::Product(Product { sign: _, leading_float, num: _, denom: _ }) => *leading_float,
            _ => 1.0
        }
    }
    pub fn with_leading_float(self, x: f64) -> Self {
        if utils::approx_eq(x, 0.0) {
            Decomposed::from_f64(0.0)
        } else if let Decomposed::Product(Product { sign, leading_float: _, num, denom }) = self {
            Decomposed::Product(Product { sign, leading_float: x, num, denom })
        } else if x != 1.0 {
            Decomposed::Product(Product { sign: 1.0, leading_float: x, num: vec![self], denom: Vec::new() })
        } else {
            self
        }
    }
    pub fn factors(self) -> (Vec<Decomposed>, Vec<Decomposed>) {
        match self {
            Decomposed::Product(Product { sign: _, leading_float: _, num, denom }) => (num, denom),
            other => (vec![other], Vec::new())
        }
    }
    pub fn borrow_factors<'a>(&'a self) -> (Cow<'a, [Decomposed]>, Cow<'a, [Decomposed]>) {
        match self {
            Decomposed::Product(Product { sign: _, leading_float: _, num, denom }) => (
                Cow::Borrowed(num),
                Cow::Borrowed(denom)
            ),
            other => (
                Cow::Borrowed(std::slice::from_ref(other)),
                Cow::Owned(Vec::new())
            )
        }
    }

    pub fn is_certainly_real(&self) -> bool {
        match self {
            Decomposed::Product(Product { sign: _, leading_float: _, num, denom }) => {
                num.iter().all(|x| x.is_certainly_real()) && denom.iter().all(|x| x.is_certainly_real())
            }
            Decomposed::BinaryOperation(_, DecomposableBinop::And | DecomposableBinop::Comp(_) | DecomposableBinop::Or, _) => true,
            Decomposed::BinaryOperation(lhs, DecomposableBinop::Pow, rhs) => lhs.is_certainly_real() && rhs.is_certainly_real(),
            Decomposed::UnaryOperation(DecomposableUnaryOp::Abs | DecomposableUnaryOp::Not, _) => true,
            Decomposed::Indecomposable(Expression::Identifier(s)) if let Some((_, (_, 0.0))) = defaults::DEFAULT_CONSTANTS.iter().find(|(_s, _)| _s == s) => true,
            Decomposed::Indecomposable(Expression::Number(_)) => true,
            Decomposed::Function(name, args) => match name.as_str() {
                "sign" => true,
                "exp" | "ln" | "cos" | "acos" | "cosh" | "acosh" | "sin" | "asin" | "sinh" | "asinh" | "tan" | "atan" | "tanh" | "atanh" => args[0].is_certainly_real(),
                "log" => args[0].is_certainly_real() && args[1].is_certainly_real(),
                "sqrt" => matches!(args[0].has_sign_with_certainty(), Some(1.0)),
                _ => false
            },
            _ => false
        }
    }

    /// Returns `Some(x)` if `self` can be determined to certainly have sign `x` where
    /// `x` is `+1` or `-1`. Returns `None` otherwise, including when `self` might be a complex number.
    pub fn has_sign_with_certainty(&self) -> Option<f64> {
        match self {
            Decomposed::Product(Product { sign, leading_float: _, num, denom }) => {
                let mut res = *sign;
                for d in num.iter().chain(denom.iter()) {
                    res *= d.has_sign_with_certainty()?;
                }
                Some(res)
            }
            Decomposed::Sum(terms, ops) => {
                if terms.is_empty() {
                    Some(1.0) // Empty sum is interpreted as 0.0
                } else {
                    let mut it = terms.iter().zip(ops.iter());
                    let all_signs_must_be = {
                        let (a, b) = it.next().unwrap();
                        if *b {
                            a.has_sign_with_certainty()?
                        } else {
                            -a.has_sign_with_certainty()?
                        }
                    };
                    for (a, b) in it {
                        if {
                            if *b {
                                a.has_sign_with_certainty()?
                            } else {
                                -a.has_sign_with_certainty()?
                            }
                        } != all_signs_must_be {
                            return None;
                        }
                    }
                    Some(all_signs_must_be)
                }
            }
            Decomposed::UnaryOperation(DecomposableUnaryOp::Abs | DecomposableUnaryOp::Not, _)
            | Decomposed::BinaryOperation(_, DecomposableBinop::And | DecomposableBinop::Comp(_) | DecomposableBinop::Or, _)
            => Some(1.0),
            Decomposed::BinaryOperation(lhs, DecomposableBinop::Pow, rhs) => {
                if let Some(pow) = rhs.try_to_float() {
                    if let Some(i) = utils::expect_int::<i64>(pow) && i.rem_euclid(2) != 0 {
                        lhs.has_sign_with_certainty()
                    } else {
                        Some(1.0)
                    }
                } else {
                    // If `lhs >= 0`, then `lhs ^ rhs >= 0`. If `lhs < 0`, then `lhs ^ rhs` is only valid when
                    // `rhs` is an odd integer, and then, `lhs ^ rhs < 0` too.
                    lhs.has_sign_with_certainty()
                }
            }
            Decomposed::Indecomposable(Expression::Identifier(s)) if let Some((_, (c, 0.0))) = defaults::DEFAULT_CONSTANTS.iter().find(|(_s, _)| _s == s) => {
                Some(if *c >= 0.0 {1.0} else {-1.0})
            }
            Decomposed::Indecomposable(Expression::Number(c)) => {
                Some(if *c >= 0.0 {1.0} else {-1.0})
            }
            Decomposed::Function(name, args) => match name.as_str() {
                "sign" | "tan" | "atan" | "tanh" | "atanh" => args[0].has_sign_with_certainty(),
                "exp" => if args[0].is_certainly_real() {Some(1.0)} else {None},
                _ => None
            },
            _ => None
        }
    }
}

macro_rules! simplify_match_statement {
    (
        $expr:expr;
        $($binop:ident$(($($binop_param:pat),*))?),*;
        $($comp_to_zero:ident),*;
        $($unop:ident$(($($unop_param:pat),*))?),*;
        $($folded_op:ident),*;
        $($unary_function:ident),*
    ) => {
        paste! {
            match $expr {
                e @ (Expression::Identifier(_) | Expression::None) => Decomposed::Indecomposable(e),
                Expression::Number(x) => Decomposed::from_f64(x),
                Expression::Vector(v) => simplify_vec(v.into_iter().map(simplify).collect()),
                Expression::Matrix(m, n, v) => simplify_mat(m, n, v.into_iter().map(simplify).collect()),
                Expression::Tuple(v) => Decomposed::Indecomposable(Expression::Tuple(v.into_iter().map(|e| simplify(e).to_expr()).collect())),
                $(
                    Expression::BinaryOperation(lhs, BinaryOperation::$binop$(($($binop_param),*))?, rhs)
                    => [<simplify_ $binop:lower>](simplify(*lhs), simplify(*rhs))
                ),*,
                $(
                    Expression::BinaryOperation(lhs, BinaryOperation::Comp(Comparison::$comp_to_zero, None), rhs)
                    if matches!(*rhs, Expression::Number(0.0))
                    => [<simplify_ $comp_to_zero:lower _zero>](simplify(*lhs))
                ),*,
                Expression::BinaryOperation(lhs, other_binop, rhs) => Decomposed::Indecomposable(
                    expr_binop_from_enum!(
                        simplify(*lhs).to_expr(),
                        other_binop,
                        simplify(*rhs).to_expr()
                    )
                ),
                $(
                    Expression::UnaryOperation(UnaryOperation::$unop$(($($unop_param),*))?, inner)
                    => [<simplify_ $unop:lower>](simplify(*inner))
                ),*,
                Expression::UnaryOperation(other_unop, inner) => Decomposed::Indecomposable(
                    Expression::UnaryOperation(
                        other_unop,
                        Box::new(simplify(*inner).to_expr())
                    )
                ),
                $(
                    Expression::FoldedOperation(FoldedOperation::$folded_op, index_var, from, conditions, to, inner)
                    => [<simplify_ $folded_op:lower>](
                        index_var,
                        simplify(*from),
                        conditions.into_iter().map(simplify).collect(),
                        simplify(*to),
                        simplify(*inner)
                    )
                ),*,
                $(
                    Expression::Function(name, args)
                    if &name == stringify!($unary_function) && args.len() == 1
                    => [<simplify_ $unary_function>](simplify(args.into_iter().next().unwrap()))
                ),*,
                Expression::Function(other_func, args) => Decomposed::Function(
                    other_func,
                    args.into_iter().map(simplify).collect()
                ),
                Expression::Assignment(lhs, rhs) => Decomposed::Indecomposable(
                    Expression::Assignment(
                        Box::new(simplify(*lhs).to_expr()),
                        Box::new(simplify(*rhs).to_expr())
                    )
                ),
                Expression::PartialDerivative(vars, inner) => Decomposed::Indecomposable(
                    Expression::PartialDerivative(
                        vars,
                        Box::new(simplify(*inner).to_expr())
                    )
                ),
                Expression::DirectionalDerivative(vars, inner, point, direction) => Decomposed::Indecomposable(
                    Expression::DirectionalDerivative(
                        vars,
                        Box::new(simplify(*inner).to_expr()),
                        point.into_iter().map(|x| simplify(x).to_expr()).collect(),
                        direction.into_iter().map(|x| simplify(x).to_expr()).collect()
                    )
                ),
                Expression::Integral(inner, from, to, wrt) => Decomposed::Indecomposable(
                    Expression::Integral(
                        Box::new(simplify(*inner).to_expr()),
                        Box::new(simplify(*from).to_expr()),
                        Box::new(simplify(*to).to_expr()),
                        wrt
                    )
                ),
                Expression::IfElse(condition, iftrue, iffalse) => Decomposed::Indecomposable(
                    simplify_if_else(
                        simplify(*condition),
                        simplify(*iftrue),
                        simplify(*iffalse)
                    )
                    .to_expr()
                )
            }
        }
    };
}

/// Simplifies an expression bottom-up.
/// 
/// Call `.to_expr()` on the result to convert it to a regular `Expression`.
pub fn simplify(expr: Expression) -> Decomposed {
    let res = simplify_match_statement!(
        expr;
        Add, Sub, Mul, Div, Pow(_), And, Or;
        Eq, Gt, Lt;
        Neg, Abs, Not;
        Sum, Product;
        exp, ln, sqrt, sign,
        cos, sin, cosh, sinh, tanh
    );
    res
}

/// Constructs a `Decomposed::Sum` while grouping together all terms that are made up
/// of the same factors.
/// 
/// Assumes that all potential `sign` fields are `+1`. This is guaranteed by the callers
/// `simplify_add_unsigned` and `simplify_sub_unsigned`, which in turn are only called
/// on terms made to have sign `+1` if they were products with sign `-1` before.
fn group_sum(terms: Vec<Decomposed>, operations: Vec<bool>) -> Decomposed {
    if terms.is_empty() {
        return Decomposed::from_f64(0.0);
    }
    let (mut new_terms, mut new_ops) = (Vec::<Decomposed>::new(), Vec::<bool>::new());
    for (r_term, r_sign) in terms.into_iter().zip(operations) {
        let (inum, idenom) = r_term.borrow_factors();
        if let Some((l_term, l_sign)) = new_terms
        .iter_mut()
        .zip(new_ops.iter_mut())
        .find(|(l_term, _)| {
            let (jnum, jdenom) = l_term.borrow_factors();
            utils::is_permutation_of(&inum, &jnum) && utils::is_permutation_of(&idenom, &jdenom)
        }) {
            let new_f = match (&l_sign, &r_sign) {
                (true, true) => l_term.leading_float() + r_term.leading_float(),
                (false, true) => r_term.leading_float() - l_term.leading_float(),
                (true, false) => l_term.leading_float() - r_term.leading_float(),
                (false, false) => -l_term.leading_float() - r_term.leading_float()
            };
            if utils::approx_eq(new_f, 0.0) {
                *l_term = Decomposed::from_f64(0.0);
            } else {
                *l_sign = new_f >= 0.0;
                let old = std::mem::take(l_term);
                *l_term = old.with_leading_float(new_f.abs());
            }
        } else {
            new_terms.push(r_term);
            new_ops.push(r_sign);
        }
    }
    (new_terms, new_ops) = new_terms.into_iter()
    .zip(new_ops)
    .filter(|(x, _)| x.leading_float() != 0.0)
    .collect();
    match new_terms.len() {
        0 => Decomposed::from_f64(0.0),
        1 => if new_ops[0] {
            new_terms.into_iter().next().unwrap()
        } else {
            new_terms.into_iter().next().unwrap().negated()
        },
        _ => Decomposed::Sum(new_terms, new_ops)
    }
}

macro_rules! impl_simplify_binop {
    ($binop:ident: ($lhs:ident, $rhs:ident) => $body:expr) => {
        paste! {
            /// **Only works if the given expressions `lhs`, `rhs` are already simplified.**
            pub fn [<simplify_ $binop>](lhs: Decomposed, rhs: Decomposed) -> Decomposed {
                match (lhs, rhs) {
                    (Decomposed::IfElse(lcondition, liftrue, liffalse), Decomposed::IfElse(rcondition, riftrue, riffalse))
                    if matches!(*liffalse, Decomposed::Indecomposable(Expression::None))
                    && matches!(*riffalse, Decomposed::Indecomposable(Expression::None))
                    => Decomposed::IfElse(
                        Box::new(simplify_and(*lcondition, *rcondition)),
                        Box::new([<simplify_ $binop>](*liftrue, *riftrue)),
                        liffalse
                    ),
                    (Decomposed::IfElse(condition, iftrue, iffalse), rhs)
                    if matches!(*iffalse, Decomposed::Indecomposable(Expression::None)) => Decomposed::IfElse(
                        condition,
                        Box::new([<simplify_ $binop>](*iftrue, rhs)),
                        iffalse
                    ),
                    (lhs, Decomposed::IfElse(condition, iftrue, iffalse))
                    if matches!(*iffalse, Decomposed::Indecomposable(Expression::None)) => Decomposed::IfElse(
                        condition,
                        Box::new([<simplify_ $binop>](lhs, *iftrue)),
                        iffalse
                    ),
                    ($lhs, $rhs) => $body
                }
            }
        }
    }
}

impl_simplify_binop!(add: (lhs, rhs) => match (lhs.is_neg(), rhs.is_neg()) {
    // (-x) + (-y) => -(x+y)
    (true, true) => simplify_add_unsigned(lhs.negated(), rhs.negated()).negated(),
    // (-x) + y => y - x
    (true, false) => simplify_sub_unsigned(rhs, lhs.negated()),
    // x + (-y) => x - y
    (false, true) => simplify_sub_unsigned(lhs, rhs.negated()),
    _ => simplify_add_unsigned(lhs, rhs)
});
/// Acts like `simplify_add` but does not perform the initial sign checks.
fn simplify_add_unsigned(lhs: Decomposed, rhs: Decomposed) -> Decomposed {
    match (lhs.try_to_float(), rhs.try_to_float()) {
        (Some(x), _) if utils::approx_eq(x, 0.0) => return rhs,
        (_, Some(x)) if utils::approx_eq(x, 0.0) => return lhs,
        (Some(x), Some(y)) => return Decomposed::from_f64(x + y),
        _ => {}
    }
    // If both sides are vectors of the same length, make the sum of vectors a vector of sums.
    // This allows us to further simplify the individual components of the vector.
    if let Some(m) = lhs.is_vec() && let Some(n) = rhs.is_vec() && m == n {
        return simplify_vec(
            lhs.expect_vec()
            .zip(rhs.expect_vec())
            .map(
                |(x, y)| simplify_add(x, y)
            )
            .collect()
        );
    }
    // Same for matrices
    if let Some((m, n)) = lhs.is_mat() && let Some((_m, _n)) = rhs.is_mat() && m == _m && n == _n {
        return simplify_mat(
            m, n,
            lhs.expect_mat()
            .zip(rhs.expect_mat())
            .map(
                |(x, y)| simplify_add(x, y)
            )
            .collect()
        );
    }
    // Otherwise, interpret both sides as sums, concatenate them and group identical terms.
    let (mut lterms, mut lops) = lhs.as_sum();
    let (rterms, rops) = rhs.as_sum();
    lterms.extend(rterms);
    lops.extend(rops);
    group_sum(lterms, lops)
}

impl_simplify_binop!(sub: (lhs, rhs) => match (lhs.is_neg(), rhs.is_neg()) {
    // (-x) - (-y) => y - x
    (true, true) => simplify_sub_unsigned(rhs.negated(), lhs.negated()),
    // (-x) - y => -(x+y)
    (true, false) => simplify_add_unsigned(lhs.negated(), rhs).negated(),
    // x - (-y) => x + y
    (false, true) => simplify_add_unsigned(lhs, rhs.negated()),
    _ => simplify_sub_unsigned(lhs, rhs)
});
/// Acts like `simplify_add` but does not perform the initial sign checks.
fn simplify_sub_unsigned(lhs: Decomposed, rhs: Decomposed) -> Decomposed {
    match (lhs.try_to_float(), rhs.try_to_float()) {
        (Some(x), _) if utils::approx_eq(x, 0.0) => return rhs.negated(),
        (_, Some(x)) if utils::approx_eq(x, 0.0) => return lhs,
        (Some(x), Some(y)) => return Decomposed::from_f64(x - y),
        _ => {}
    }
    if let Some(m) = lhs.is_vec() && let Some(n) = rhs.is_vec() && m == n {
        return simplify_vec(
            lhs.expect_vec()
            .zip(rhs.expect_vec())
            .map(
                |(x, y)| simplify_sub(x, y)
            )
            .collect()
        );
    }
    if let Some((m, n)) = lhs.is_mat() && let Some((_m, _n)) = rhs.is_mat() && m == _m && n == _n {
        return simplify_mat(
            m, n,
            lhs.expect_mat()
            .zip(rhs.expect_mat())
            .map(
                |(x, y)| simplify_sub(x, y)
            )
            .collect()
        );
    }
    let (mut lterms, mut lops) = lhs.as_sum();
    let (rterms, rops) = rhs.as_sum();
    lterms.extend(rterms);
    // Since we subtract the RHS, all signs on the RHS must be flipped.
    lops.extend(rops.into_iter().map(|x| !x));
    group_sum(lterms, lops)
}


/// Constructs a `Decomposed::Product` with the numerator consisting of the factors `num`
/// and the denominated of the factors `denom` while grouping together all identical factors.
/// Multiplies the result by `float`.
/// 
/// Assumes that all potential `sign` fields are `+1`. This is guaranteed by the callers
/// `simplify_mul_unsigned` and `simplify_div_unsigned`, which in turn are only called
/// on factors made to have sign `+1` if they were products with sign `-1` before.
fn group_product(num: Vec<Decomposed>, denom: Vec<Decomposed>, float: f64) -> Decomposed {
    if num.is_empty() && denom.is_empty() {
        return Decomposed::from_f64(1.0);
    }
    let mut leading_float = float;
    // `new_factors` contains tuples `(factor, exponent)`.
    let mut new_factors = Vec::<(Decomposed, Decomposed)>::new();
    for new_factor in num {
        let (new_float, powers) = new_factor.as_powers();
        leading_float *= new_float;
        for (new_base, new_pow) in powers {
            if let Some((_, old_pow)) = new_factors.iter_mut().find(|(old_base, _)| *old_base == new_base) {
                let old = std::mem::take(old_pow);
                *old_pow = simplify_add(old, new_pow);
            } else if let Some(x) = new_base.try_to_float() && utils::approx_eq(x, 0.0)
            && let Some(y) = new_pow.try_to_float() && !utils::approx_eq(y, 0.0) {
                return Decomposed::from_f64(0.0);
            } else {
                new_factors.push((new_base, new_pow));
            }
        }
    }
    for new_factor in denom {
        let (new_float, powers) = new_factor.as_powers();
        leading_float *= new_float;
        for (new_base, new_pow) in powers {
            if let Some((_, old_pow)) = new_factors.iter_mut().find(|(old_base, _)| *old_base == new_base) {
                let old = std::mem::take(old_pow);
                *old_pow = simplify_sub(old, new_pow);
            } else {
                new_factors.push((new_base, new_pow));
            }
        }
    }
    let mut num = Vec::new();
    let mut denom = Vec::new();
    for (base, pow) in new_factors {
        if let Some(x) = pow.try_to_float() && utils::approx_eq(x, 0.0) {
            continue;
        } else if pow.is_neg() {
            denom.push(simplify_pow(base, pow.negated()));
        } else {
            num.push(simplify_pow(base, pow));
        }
    }
    if num.is_empty() && denom.is_empty() {
        Decomposed::from_f64(leading_float)
    } else if num.len() == 1 && denom.is_empty() {
        let res = num.into_iter().next().unwrap();
        let f = res.leading_float();
        res.with_leading_float(f * leading_float)
    } else {
        Decomposed::Product(Product {
            sign: 1.0,
            leading_float,
            num,
            denom
        })
    }
}

impl_simplify_binop!(mul: (lhs, rhs) => match (lhs.is_neg(), rhs.is_neg()) {
    // (-x) * (-y) => x * y
    (true, true) => simplify_mul_unsigned(lhs.negated(), rhs.negated()),
    // (-x) * y => -(x * y)
    (true, false) => simplify_mul_unsigned(lhs.negated(), rhs).negated(),
    // x * (-y) => -(x * y)
    (false, true) => simplify_mul_unsigned(lhs, rhs.negated()).negated(),
    _ => simplify_mul_unsigned(lhs, rhs)
});
/// Acts like `simplify_mul` but does not perform the initial sign checks.
fn simplify_mul_unsigned(lhs: Decomposed, rhs: Decomposed) -> Decomposed {
    match (lhs.try_to_float(), rhs.try_to_float()) {
        (Some(x), _) | (_, Some(x)) if utils::approx_eq(x, 0.0) => return Decomposed::from_f64(0.0),
        (Some(x), _) if utils::approx_eq(x, 1.0) => return rhs,
        (_, Some(x)) if utils::approx_eq(x, 1.0) => return lhs,
        (Some(x), Some(y)) => return Decomposed::from_f64(x * y),
        _ => {}
    }
    match (lhs, rhs) {
        // sign(x) * |x| = |x| * sign(x) = x
        (
            Decomposed::Function(name, v),
            Decomposed::UnaryOperation(DecomposableUnaryOp::Abs, x)
        ) | (
            Decomposed::UnaryOperation(DecomposableUnaryOp::Abs, x),
            Decomposed::Function(name, v)
        ) if name == "sign" && v[0] == *x => *x,
        // sign(x) * x = x * sign(x) = |x|
        (
            Decomposed::Function(name, v),
            x
        ) | (
            x,
            Decomposed::Function(name, v)
        ) if name == "sign" && v[0] == x => simplify_abs(x),
        // Default case
        (lhs, rhs) => {
            let float = lhs.leading_float() * rhs.leading_float();
            let (mut lnum, mut ldenom) = lhs.factors();
            let (rnum, rdenom) = rhs.factors();
            lnum.extend(rnum);
            ldenom.extend(rdenom);
            group_product(lnum, ldenom, float)
        }
    }
}

impl_simplify_binop!(div: (lhs, rhs) => match (lhs.is_neg(), rhs.is_neg()) {
    // (-x) / (-y) => x / y
    (true, true) => simplify_div_unsigned(lhs.negated(), rhs.negated()),
    // (-x) / y => -(x / y)
    (true, false) => simplify_div_unsigned(lhs.negated(), rhs).negated(),
    // x / (-y) => -(x / y)
    (false, true) => simplify_div_unsigned(lhs, rhs.negated()).negated(),
    _ => simplify_div_unsigned(lhs, rhs)
});
/// Acts like `simplify_div` but does not perform the initial sign checks.
fn simplify_div_unsigned(lhs: Decomposed, rhs: Decomposed) -> Decomposed {
    match (lhs.try_to_float(), rhs.try_to_float()) {
        (Some(x), _) if utils::approx_eq(x, 0.0) => return Decomposed::from_f64(0.0),
        (_, Some(x)) if utils::approx_eq(x, 0.0) => panic!("attempt to divide by zero"),
        (Some(x), _) if utils::approx_eq(x, 1.0) => return rhs.inverted(),
        (_, Some(x)) if utils::approx_eq(x, 1.0) => return lhs,
        (Some(x), Some(y)) => return Decomposed::from_f64(x / y),
        _ => {}
    }
    let float = lhs.leading_float() / rhs.leading_float();
    let (mut lnum, mut ldenom) = lhs.factors();
    let (rnum, rdenom) = rhs.factors();
    lnum.extend(rdenom);
    ldenom.extend(rnum);
    group_product(lnum, ldenom, float)
}

impl_simplify_binop!(pow: (lhs, rhs) => {
    match (lhs.try_to_float(), rhs.try_to_float()) {
        (Some(l), Some(r)) => return Decomposed::from_f64(l.powf(r)),
        (None, Some(r)) if utils::approx_eq(r, 1.0) => return lhs,
        (None, Some(r)) if utils::approx_eq(r, 0.0) => return Decomposed::from_f64(1.0),
        (Some(l), None) if utils::approx_eq(l, 0.0) => return Decomposed::Function(
            "1".to_string(),
            vec![decomp_binop!(
                rhs,
                Comp(Comparison::Eq),
                Decomposed::from_f64(0.0)
            )]
        ),
        _ => {}
    }
    match lhs {
        // (x^y)^z = x^(yz)
        Decomposed::BinaryOperation(inner_l, DecomposableBinop::Pow, inner_r) => Decomposed::BinaryOperation(
            inner_l,
            DecomposableBinop::Pow,
            Box::new(simplify_mul(*inner_r, rhs))
        ),
        // sqrt(x)^y = x^(y/2)
        Decomposed::Function(name, v) if let Some(x) = rhs.try_to_float() && name == "sqrt" => simplify_pow(
            v.into_iter().next().unwrap(),
            Decomposed::from_f64(x / 2.0)
        ),
        lhs => decomp_binop!(lhs, Pow, rhs)
    }
});

pub fn simplify_and(lhs: Decomposed, rhs: Decomposed) -> Decomposed {
    match (lhs.try_to_float(), rhs.try_to_float()) {
        (Some(l), Some(r)) => Decomposed::from_f64(if utils::approx_eq(l, 1.0) && utils::approx_eq(r, 1.0) {1.0} else {0.0}),
        (None, Some(r)) => if utils::approx_eq(r, 1.0) {lhs} else {Decomposed::from_f64(0.0)},
        (Some(l), None) => if utils::approx_eq(l, 1.0) {rhs} else {Decomposed::from_f64(0.0)},
        (None, None) => {
            if lhs == rhs {
                lhs
            } else {
                decomp_binop!(lhs, And, rhs)
            }
        }
    }
}
pub fn simplify_or(lhs: Decomposed, rhs: Decomposed) -> Decomposed {
    match (lhs.try_to_float(), rhs.try_to_float()) {
        (Some(l), Some(r)) => Decomposed::from_f64(if utils::approx_eq(l, 1.0) || utils::approx_eq(r, 1.0) {1.0} else {0.0}),
        (None, Some(r)) => if utils::approx_eq(r, 1.0) {Decomposed::from_f64(1.0)} else {lhs},
        (Some(l), None) => if utils::approx_eq(l, 1.0) {Decomposed::from_f64(1.0)} else {rhs},
        (None, None) => {
            if lhs == rhs {
                lhs
            } else {
                decomp_binop!(lhs, Or, rhs)
            }
        }
    }
}
pub fn simplify_not(operand: Decomposed) -> Decomposed {
    match operand.try_to_float() {
        Some(x) => Decomposed::from_f64(if utils::approx_eq(x, 1.0) {0.0} else {1.0}),
        None => decomp_unop!(Not, operand)
    }
}

#[inline]
pub fn simplify_neg(operand: Decomposed) -> Decomposed {
    operand.negated()
}
pub fn simplify_abs(operand: Decomposed) -> Decomposed {
    if let Some(x) = operand.try_to_float() {
        Decomposed::from_f64(x.abs())
    } else if let Some(sign) = operand.has_sign_with_certainty() {
        if sign == 1.0 {
            operand
        } else {
            operand.negated()
        }
    } else {
        Decomposed::UnaryOperation(DecomposableUnaryOp::Abs, Box::new(operand))
    }
}

pub fn simplify_exp(arg: Decomposed) -> Decomposed {
    if let Some(x) = arg.try_to_float() {
        return Decomposed::from_f64(x.exp());
    }
    match arg {
        Decomposed::Function(s, v) if s == "ln" => v.into_iter().next().unwrap(),
        Decomposed::Function(s, v) if s == "log" => {
            // exp(log_a(b)) = exp(ln(b) / ln(a)) = b^(1/ln(a))
            let mut it = v.into_iter();
            let (a, b) = (it.next().unwrap(), it.next().unwrap());
            simplify_pow(
                b,
                simplify_div(
                    Decomposed::from_f64(1.0),
                    simplify_ln(a)
                )
            )
        }
        other => Decomposed::Function("exp".to_string(), vec![other])
    }
}
pub fn simplify_ln(arg: Decomposed) -> Decomposed {
    if let Some(x) = arg.try_to_float() {
        return Decomposed::from_f64(x.ln());
    }
    match arg {
        Decomposed::Function(s, v) if s == "exp" => v.into_iter().next().unwrap(),
        // N.b.: we're not splitting up general products because:
        // - `ln(ab)` is already more concise than `ln(a) + ln(b)`
        // - In `ln(ab)`, it is possible that `a` and `b` are both negative at times and both positive
        //   at other times, so `ln(a) + ln(b)` is not always valid.
        Decomposed::Product(Product { sign, leading_float, num, denom })
        if num.is_empty() => {
            // ln(c / xyz) = -ln((1/c) * xyz)
            Decomposed::Function(
                "ln".to_string(),
                vec![Decomposed::Product(Product {
                    sign,
                    leading_float: 1.0 / leading_float,
                    num: denom,
                    denom: num
                })]
            )
            .negated()
        }
        Decomposed::BinaryOperation(lhs, DecomposableBinop::Pow, rhs) => {
            // ln(a^b) = b ln(a)
            simplify_mul(
                *rhs,
                simplify_ln(*lhs)
            )
        }
        other => Decomposed::Function("exp".to_string(), vec![other])
    }
}
pub fn simplify_sqrt(arg: Decomposed) -> Decomposed {
    if let Some(x) = arg.try_to_float() {
        return Decomposed::from_f64(x.sqrt());
    }
    match arg {
        Decomposed::BinaryOperation(lhs, DecomposableBinop::Pow, rhs)
        if let Some(pow) = rhs.try_to_float() => {
            Decomposed::BinaryOperation(
                lhs,
                DecomposableBinop::Pow,
                Box::new(Decomposed::from_f64(pow / 2.0))
            )
        }
        other => Decomposed::Function("sqrt".to_string(), vec![other])
    }
}
pub fn simplify_sign(arg: Decomposed) -> Decomposed {
    if let Some(x) = arg.try_to_float() {
        Decomposed::from_f64(if x >= 0.0 {1.0} else {-1.0})
    } else if let Some(sign) = arg.has_sign_with_certainty() {
        Decomposed::from_f64(sign)
    } else {
        Decomposed::Function("sign".to_string(), vec![arg])
    }
}

macro_rules! simplify_trig_fn {
    ($func:ident) => {
        paste! {
            pub fn [<simplify_ $func>](arg: Decomposed) -> Decomposed {
                if let Some(x) = arg.try_to_float() {
                    return Decomposed::from_f64(x.$func());
                }
                match arg {
                    Decomposed::Function(s, v) if s == stringify!([<a $func>]) => v.into_iter().next().unwrap(),
                    other => Decomposed::Function(stringify!($func).to_string(), vec![other])
                }
            }
            pub fn [<simplify_a $func>](arg: Decomposed) -> Decomposed {
                if let Some(x) = arg.try_to_float() {
                    return Decomposed::from_f64(x.[<a $func>]());
                }
                match arg {
                    Decomposed::Function(s, v) if s == stringify!($func) => v.into_iter().next().unwrap(),
                    other => Decomposed::Function(stringify!([<a $func>]).to_string(), vec![other])
                }
            }
        }
    }
}
simplify_trig_fn!(cos);
simplify_trig_fn!(sin);
simplify_trig_fn!(tan);
simplify_trig_fn!(cosh);
simplify_trig_fn!(sinh);
simplify_trig_fn!(tanh);

pub fn simplify_if_else(condition: Decomposed, iftrue: Decomposed, iffalse: Decomposed) -> Decomposed {
    if iftrue == iffalse {
        iftrue
    } else {
        Decomposed::Indecomposable(expr_if_else!(
            condition.to_expr(),
            iftrue.to_expr(),
            iffalse.to_expr()
        ))
    }
}

pub fn simplify_eq_zero(operand: Decomposed) -> Decomposed {
    if let Some(x) = operand.try_to_float() {
        Decomposed::from_f64(if x == 0.0 {1.0} else {0.0})
    } else {
        decomp_binop!(
            operand,
            Comp(Comparison::Eq),
            Decomposed::from_f64(0.0)
        )
    }
}
pub fn simplify_gt_zero(operand: Decomposed) -> Decomposed {
    if let Some(x) = operand.try_to_float() {
        Decomposed::from_f64(if x > 0.0 {1.0} else {0.0})
    } else {
        decomp_binop!(
            operand,
            Comp(Comparison::Gt),
            Decomposed::from_f64(0.0)
        )
    }
}
pub fn simplify_lt_zero(operand: Decomposed) -> Decomposed {
    if let Some(x) = operand.try_to_float() {
        Decomposed::from_f64(if x < 0.0 {1.0} else {0.0})
    } else {
        decomp_binop!(
            operand,
            Comp(Comparison::Lt),
            Decomposed::from_f64(0.0)
        )
    }
}

pub fn simplify_sum(
    index_var: String,
    from: Decomposed,
    conditions: Vec<Decomposed>,
    to: Decomposed,
    inner: Decomposed
) -> Decomposed {
    if inner.is_neg() {
        Decomposed::Indecomposable(Expression::FoldedOperation(
            FoldedOperation::Sum,
            index_var,
            Box::new(from.to_expr()),
            conditions.into_iter().map(|x| x.to_expr()).collect(),
            Box::new(to.to_expr()),
            Box::new(inner.negated().to_expr())
        ))
        .negated()
    } else {
        Decomposed::Indecomposable(Expression::FoldedOperation(
            FoldedOperation::Sum,
            index_var,
            Box::new(from.to_expr()),
            conditions.into_iter().map(|x| x.to_expr()).collect(),
            Box::new(to.to_expr()),
            Box::new(inner.to_expr())
        ))
    }
}
pub fn simplify_product(
    index_var: String,
    from: Decomposed,
    conditions: Vec<Decomposed>,
    to: Decomposed,
    inner: Decomposed
) -> Decomposed {
    Decomposed::Indecomposable(Expression::FoldedOperation(
        FoldedOperation::Product,
        index_var,
        Box::new(from.to_expr()),
        conditions.into_iter().map(|x| x.to_expr()).collect(),
        Box::new(to.to_expr()),
        Box::new(inner.to_expr())
    ))
}

/// Constructs an `Expression::Vector` from the given components.
/// 
/// If strictly more than half of all given components are found to be negative,
/// pulls out the negation sign.
pub fn simplify_vec(components: Vec<Decomposed>) -> Decomposed {
    if components.iter().map(|c| c.is_neg() as usize).sum::<usize>() <= components.len() / 2 {
        // Leave the vector as is
        Decomposed::Vector(components)
    } else {
        // Pull the negation sign out, which means negating all inner components
        Decomposed::Vector(components.into_iter().map(|x| x.negated()).collect())
    }
}
pub fn simplify_mat(m: usize, n: usize, components: Vec<Decomposed>) -> Decomposed {
    if components.iter().map(|c| c.is_neg() as usize).sum::<usize>() <= components.len() / 2 {
        Decomposed::Matrix(m, n, components)
    } else {
        Decomposed::Matrix(m, n, components.into_iter().map(|x| x.negated()).collect())
    }
}