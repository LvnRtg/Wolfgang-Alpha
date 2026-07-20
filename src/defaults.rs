use std::collections::HashMap;
use std::f64::consts;
use std::sync::LazyLock;

use crate::math::expressions;
use crate::math::{Complex, DirectFunction, Expression, Matrix, Object, ObjType, FunctionRepr};
use crate::math::operations::{UnaryOperation, BinaryOperation};
use crate::{expr_if_else, expr_and, expr_compare, expr_add, expr_sub, expr_mul, expr_div, expr_square, expr_neg, expr_1arg_func};

/// Wrapped in a function because const hashmaps aren't available yet.
pub fn default_constants() -> HashMap<String, Object> {
    HashMap::<String, Object>::from([
        ("e".to_string(), Object::Real(consts::E)),
        ("pi".to_string(), Object::Real(consts::PI)),
        ("i".to_string(), Object::Complex(Complex { real: 0.0, imag: 1.0 })),
    ])
}

/// Takes a function name `name`, e.g. `exp`, and returns the tuple consisting of
/// 1. Stringified name of the function
/// 2. `FunctionRepr::Direct`: expect exactly one `f64` as argument; if so, return `Ok(x.name())`, otherwise, the appropriate `Err`.
macro_rules! float_1_function {
    ($name:ident) => {
        Box::new(|args| {
            if args.len() != 1 {
                Err(format!(
                    "Wrong number of arguments provided for function '{}' (expected 1, got {}).",
                    stringify!($name),
                    args.len()
                ))
            } else {
                match args[0] {
                    Object::Real(x) => Ok(Object::Real(x.$name())),
                    _ => Err(format!(
                        "Wrong type of argument provided for function '{}' (expected float).",
                        stringify!($name)
                    )),
                }
            }
        })
    };
}

/// Takes a function name `name` (e.g. `log`), a number `n` and an expression `expr`. Returns the tuple consisting of
/// 1. Stringified name of the function
/// 2. `FunctionRepr::Direct`: expect exactly `n` arguments; if so, return `expr(args)`, otherwise, the appropriate `Err`.
macro_rules! expect_n_args {
    ($name:ident, $n:expr, $e:expr) => {
        Box::new(|args: &[Object]| {
            if args.len() != $n {
                Err(format!(
                    "Wrong number of arguments provided for function '{}' (expected {}, got {}).",
                    stringify!($name),
                    $n,
                    args.len()
                ))
            } else {
                $e(args)
            }
        })
    };
}

/// For examples, see the use of the macro in `default_functions`.
macro_rules! apply_matrix_fn {
    ($name:ident, $e:expr) => {
        Box::new(|args| {
            if args.len() != 1 {
                Err(format!(
                    "Wrong number of arguments provided for function '{}' (expected 1, got {}).",
                    stringify!($name),
                    args.len()
                ))
            } else {
                if let Object::Matrix(mat) = &args[0] {
                    $e(mat.$name(), &mat)
                }
                else { Err(format!("Wrong type for argument of function '{}' (expected Matrix).", stringify!($name))) }
            }
        })
    };
}

/// This approach is needed because direct functions cannot be cloned (and we need to clone the environment in some scenarios).
/// Therefore, we want to use `&DirectFunction` instead of `DirectFunction` in `Env`; but this requires the direct functions
/// to be permanently stored at a fixed location. This location is here.
/// 
/// Note that the user can't create new direct functions, so this approach works.
pub static DEFAULT_DIRECT_FUNCTIONS: LazyLock<[DirectFunction; 23]> = LazyLock::new(|| [
    expect_n_args!(sign, 1, |args: &[Object]| {
        match &args[0] {
            Object::Real(x) => Ok(Object::Real(if *x >= 0.0 {1.0} else {-1.0})),
            Object::Vector(v) => Ok(Object::Vector(v.transform(|x| if x >= 0.0 {1.0} else {-1.0}))),
            Object::Matrix(m) => Ok(Object::Matrix(m.transform(|x| if x >= 0.0 {1.0} else {-1.0}))),
            other => Err(format!("Undefined operation `sign` for operand {:?}.", other))
        }
    }),

    float_1_function!(exp),
    float_1_function!(ln),
    expect_n_args!(log, 2, |args: &[Object]| {
        if let Object::Real(base) = args[1] {
            match args[0] {
                Object::Real(x) => Ok(Object::Real(x.log(base))),
                _ => Err("Wrong type for first argument (value) of function 'log' (expected float).".to_string())
            }
        }
        else { Err("Wrong type for second argument (base) of function 'log' (expected float).".to_string()) }
    }),
    float_1_function!(sqrt),

    float_1_function!(cos), float_1_function!(cosh), float_1_function!(acos), float_1_function!(acosh),
    float_1_function!(sin), float_1_function!(sinh), float_1_function!(asin), float_1_function!(asinh),
    float_1_function!(tan), float_1_function!(tanh), float_1_function!(atan), float_1_function!(atanh),

    expect_n_args!(eig, 1, |args: &[Object]| {
        if let Object::Matrix(mat) = &args[0] {
            match mat.eigenvalues() {
                Some(eig) => Ok(Object::Tuple(eig)),
                None => Err(format!("Matrix must be quadratic (got size {}x{}).", mat.m(), mat.n()))
            }
        }
        else { Err("Wrong type for argument of function 'eig' (expected Matrix).".to_string()) }
    }),
    apply_matrix_fn!(det, |r, mat: &Matrix| match r {
        Some(res) => Ok(Object::Real(res)),
        None => Err(format!("Matrix must be quadratic (got size {}x{}).", mat.m(), mat.n()))
    }),
    apply_matrix_fn!(adj, |r, mat: &Matrix| match r {
        Some(res) => Ok(Object::Matrix(res)),
        None => Err(format!("Matrix must be quadratic (got size {}x{}).", mat.m(), mat.n()))
    }),
    apply_matrix_fn!(tr, |r: Result<f64, String>, _| {r.map(Object::Real)}),
    apply_matrix_fn!(transpose, |r: Matrix, _| {Ok(Object::Matrix(r))}),
    Box::new(|args|
        if args.len() == 2
        && let (Object::Vector(x), Object::Vector(y)) = (&args[0], &args[1])
        && x.len() == y.len() {
            let n = x.len();
            Ok(Object::Real((0..n).map(|i|
                x[i]
                * if i > 0 {(0..i).map(|j| y[j]).product()} else {1.0}
                * if i < n-1 {(i+1..n).map(|j| y[j]).product()} else {1.0}
            ).sum()))
        } else {
            Err("Arguments to `___helper_prod_rule` must be two vectors of equal length.".to_string())
        }
    )
]);

/// Wrapped in a function because const hashmaps aren't available yet.
pub fn default_functions() -> HashMap<String, FunctionRepr> {
    // Just collect all elements in `DEFAULT_DIRECT_FUNCTIONS` into a hashmap along with the appropriate function names
    let mut res: HashMap<String, FunctionRepr> = vec![
        "sign", "exp", "ln", "log", "sqrt",
        "cos", "cosh", "acos", "acosh",
        "sin", "sinh", "asin", "asinh",
        "tan", "tanh", "atan", "atanh",
        "eig", "det", "adj", "tr", "transpose",
        "___helper_prod_rule"
    ].into_iter().enumerate().map(
        |(i, n)|
        (n.to_string(), FunctionRepr::Direct(&DEFAULT_DIRECT_FUNCTIONS[i]))
    ).collect();
    res.insert("1".to_string(), FunctionRepr::ByExpression(
        vec!["___tmp_x".to_string()],
        expr_if_else!(
            Expression::Identifier("___tmp_x".to_string()),
            Expression::Number(1.0),
            Expression::Number(0.0)
        )
    ));
    res
}

/// Given the name of a default function and the types of the given arguments,
/// returns the corresponding output type.
pub fn get_default_fn_type(name: &str, arg_types: &[ObjType]) -> Result<ObjType, String> {
    match (name, arg_types) {
        ("eig", [ObjType::Matrix(m, n)]) if m == n => Ok(ObjType::Vector(*n)),
        ("det", [ObjType::Matrix(m, n)]) | ("tr", [ObjType::Matrix(m, n)]) if m == n => Ok(ObjType::Scalar),
        ("adj", [ObjType::Matrix(m, n)]) if m == n => Ok(ObjType::Matrix(*n, *n)),
        ("transpose", [ObjType::Matrix(m, n)]) => Ok(ObjType::Matrix(*n, *m)),
        ("___helper_prod_rule", [ObjType::Vector(m), ObjType::Vector(n)]) if m == n => Ok(ObjType::Scalar),
        ("log", [ObjType::Scalar, ObjType::Scalar]) => Ok(ObjType::Scalar),
        (_, [ObjType::Scalar]) => Ok(ObjType::Scalar),
        _ => Err(format!("No function \"{}\" accepting arguments of type {:?}.", name, arg_types))
    }
}

pub const FUNCTIONS_WITH_PROVIDED_DERIVATIVE: [&str; 20] = [
    "exp", "ln", "log",
    "sign", "sqrt",
    "cos", "cosh", "acos", "acosh",
    "sin", "sinh", "asin", "asinh",
    "tan", "tanh", "atan", "atanh",
    "det", "tr", "transpose"
];

/// Ensures that both `point` and `direction` have length `n`.
macro_rules! assert_length {
    ($n:expr, $name:ident, $point:expr, $direction:expr, $and_then:expr) => {
        if $point.len() != $n || $direction.len() != $n {
            Err(format!(
                "Wrong number of arguments provided for derivative of '{}' (expected both point and direction of length {}, got ({}, {})).",
                stringify!($name),
                $n,
                $point.len(),
                $direction.len()
            ))
        } else {
            Ok($and_then)
        }
    };
}

/// Example: `(exp, point) => Ok(Expression::Function("exp", point[0].clone()))` if `point` and `direction` both have length 1, otherwise `Err`.
macro_rules! apply_to_first_arg {
    ($name:ident, $point:expr, $direction:expr) => {
        assert_length!(1, $name, $point, $direction,
            expressions::simplify_mul(Expression::Function(
                stringify!($name).to_string(),
                vec![$point[0].clone()]
            ), $direction[0].clone())
        )
    };
}

/// If `function_name` is among the default functions, returns its derivative at point `point` in direction `direction` (provided it exists; if it simply doesn't exist, returns `Expression::None`).
/// If there is a greater error, e.g. no such default function or wrong number of arguments given, returns the corresponding `Err`.
/// 
/// Acts like a HashMap, but initializing a hashmap for this would be overkill since we do not need to modify it.
/// 
/// N.b.: we return an expression and not e.g. a `FunctionRepr` for the sake of simplicity in the application.
pub fn get_default_derivative(function_name: &str, point: &[Expression], direction: &[Expression]) -> Result<Expression, String> {
    match function_name {
        "exp" => apply_to_first_arg!(exp, point, direction),
        "ln" => assert_length!(1, ln, point, direction,
            expr_if_else!(
                expr_compare!(point[0].clone(), Gt, Expression::Number(0.0)),
                expr_div!(direction[0].clone(), point[0].clone()),
                Expression::None
            )
        ),
        "log" => assert_length!(2, log, point, direction,
            // D log(x, b)[s, t] = s \partial_x log(x, b) + t \partial_b log(x, b) = s/(x*ln(y)) - (t*ln(x))/(b*ln(b)²)     for x, b > 0expr_if_else!(
            expr_if_else!(
                expr_and!(expr_compare!(point[0].clone(), Gt, Expression::Number(0.0)), expr_compare!(point[1].clone(), Gt, Expression::Number(0.0))),
                expr_sub!(
                    expr_div!(
                        direction[0].clone(),
                        expr_mul!(
                            point[0].clone(),
                            expr_1arg_func!("ln", point[1].clone())
                        )
                    ),
                    expr_div!(
                        expr_mul!(
                            direction[1].clone(),
                            expr_1arg_func!("ln", point[0].clone())
                        ),
                        expr_mul!(
                            point[1].clone(),
                            expr_square!(expr_1arg_func!("ln", point[1].clone()))
                        )
                    )
                ),
                Expression::None
            )
        ),
        "sign" => assert_length!(1, sign, point, direction,
            expr_if_else!(
                expr_compare!(point[0].clone(), Eq, Expression::Number(0.0)),
                Expression::None,
                Expression::Number(0.0)
            )
        ),
        "sqrt" => assert_length!(1, sqrt, point, direction,
            expr_if_else!(
                expr_compare!(point[0].clone(), Gt, Expression::Number(0.0)),
                expr_div!(
                    direction[0].clone(),
                    expr_mul!(Expression::Number(2.0), expr_1arg_func!("sqrt", point[0].clone()))
                ),
                Expression::None
            )
        ),
        "cos" => assert_length!(1, cos, point, direction,
            expr_neg!(apply_to_first_arg!(sin, point, direction)?)
        ),
        "sin" => assert_length!(1, sin, point, direction,
            apply_to_first_arg!(cos, point, direction)?
        ),
        "tan" => assert_length!(1, tan, point, direction,
            expr_div!(
                direction[0].clone(),
                expr_square!(Expression::Function(
                    "cos".to_string(),
                    vec![point[0].clone()]
                ))
            )
        ),
        "acos" => assert_length!(1, acos, point, direction,
            expr_div!(
                expr_neg!(direction[0].clone()),
                expr_1arg_func!(
                    "sqrt",
                    expr_sub!(
                        Expression::Number(1.0),
                        expr_square!(point[0].clone())
                    )
                )
            )
        ),
        "asin" => assert_length!(1, asin, point, direction,
            expr_div!(
                direction[0].clone(),
                expr_1arg_func!(
                    "sqrt",
                    expr_sub!(
                        Expression::Number(1.0),
                        expr_square!(point[0].clone())
                    )
                )
            )
        ),
        "atan" => assert_length!(1, atan, point, direction,
            expr_div!(
                direction[0].clone(),
                expr_add!(
                    Expression::Number(1.0),
                    expr_square!(point[0].clone())
                )
            )
        ),
        "cosh" => apply_to_first_arg!(sinh, point, direction),
        "sinh" => apply_to_first_arg!(cosh, point, direction),
        "tanh" => assert_length!(1, tanh, point, direction,
            expr_mul!(
                direction[0].clone(),
                expr_sub!(
                    Expression::Number(1.0),
                    expr_square!(
                        Expression::Function(
                            "tanh".to_string(),
                            vec![point[0].clone()]
                        )
                    )
                )
            )
        ),
        "acosh" => assert_length!(1, acosh, point, direction,
            expr_div!(
                direction[0].clone(),
                expr_1arg_func!(
                    "sqrt",
                    expr_sub!(
                        expr_square!(point[0].clone()),
                        Expression::Number(1.0)
                    )
                )
            )
        ),
        "asinh" => assert_length!(1, asinh, point, direction,
            expr_div!(
                direction[0].clone(),
                expr_1arg_func!(
                    "sqrt",
                    expr_add!(
                        expr_square!(point[0].clone()),
                        Expression::Number(1.0)
                    )
                )
            )
        ),
        "atanh" => assert_length!(1, atanh, point, direction,
            expr_div!(
                direction[0].clone(),
                expr_sub!(
                    Expression::Number(1.0),
                    expr_square!(point[0].clone())
                )
            )
        ),
        // Jacobi's formula states `d/dt det A(t) = tr(adj(A(t)) * d/dt A(t))`.
        // Here, `A(t) = point[0]` and `d/dt A(t) = direction[0]`.
        "det" => assert_length!(1, det, point, direction,
            expr_1arg_func!(
                "tr",
                expr_mul!(
                    Expression::Function("adj".to_string(), vec![point[0].clone()]),
                    direction[0].clone()
                )
            )
        ),
        // `tr` is linear and thus commutes with the derivative.
        "tr" => assert_length!(1, tr, point, direction,
            expr_1arg_func!(
                "tr",
                direction[0].clone()
            )
        ),
        // `transpose` is linear and thus commutes with the derivative.
        "transpose" => assert_length!(1, transpose, point, direction,
            expr_1arg_func!(
                "transpose",
                direction[0].clone()
            )
        ),
        _ => Err(format!("No derivative provided for '{function_name}'."))
    }
}