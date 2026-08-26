use itertools::Itertools;
use std::collections::HashMap;
use std::f64::consts;
use std::sync::LazyLock;

use crate::{expr_if_else, expr_and, expr_compare, expr_add, expr_sub, expr_mul, expr_div, expr_square, expr_neg, expr_1arg_func};
use crate::lang::eval;
use crate::math::expressions;
use crate::math::{Complex, DirectFunction, Expression, FunctionRepr, Matrix, Object, ObjType, VarStack};
use crate::status::Status;

/// Wrapped in a function because const hashmaps aren't available yet.
pub fn default_constants() -> HashMap<String, Object> {
    HashMap::<String, Object>::from([
        ("e".to_string(), Object::Real(consts::E)),
        ("pi".to_string(), Object::Real(consts::PI)),
        ("π".to_string(), Object::Real(consts::PI)),
        ("i".to_string(), Object::Complex(Complex { real: 0.0, imag: 1.0 })),
        ("inf".to_string(), Object::Real(f64::INFINITY)),
        ("∞".to_string(), Object::Real(f64::INFINITY))
    ])
}

/// Takes a function name `name`, e.g. `exp`, and returns a `FunctionRepr::Direct` (along with its mask) which expects
/// exactly one `f64` as argument; if such an argument is given, it returns `Ok(x.name())`, otherwise, the appropriate `Err`.
/// 
/// Note: "`FunctionRepr::Direct` which expects exactly one `f64`-arg" implies that in reality, three args are expected:
/// the `f64`, the varstack and the environment.
macro_rules! float_1_function {
    ($name:ident) => {
        (
            Box::new(|evaluated_args, unevaluated_args, _| {
                let warnings = if unevaluated_args.is_empty() {
                    vec![]
                } else {
                    vec![format!(
                        "Provided {} unevaluated arguments although none are expected.",
                        unevaluated_args.len()
                    )]
                };
                if evaluated_args.len() != 1 {
                    Err(format!(
                        "Wrong number of evaluated arguments provided for function '{}' (expected 1, got {}).",
                        stringify!($name),
                        evaluated_args.len()
                    ))
                } else {
                    match &evaluated_args[0] {
                        Object::Real(x) => Ok(Status{value: Object::Real(x.$name()), warnings}),
                        other => Err(format!(
                            "Wrong type of argument provided for function '{}' (expected float, got {}).",
                            stringify!($name),
                            other.get_type()
                        )),
                    }
                }
            }),
            (1, 0, false)
        )
    };
}

/// Takes a function name `name` (e.g. `log`), a number `n` and an expression `expr`. Returns a `FunctionRepr::Direct`
/// (along with its mask) which expects exactly `n` arguments that can be evaluated to `Object`; if such arguments are given,
/// it returns `expr(args.map(eval))`, otherwise, the appropriate `Err`.
/// 
/// Note: "`FunctionRepr::Direct` which expects exactly `n` args" implies that in reality, `n+2` args are expected:
/// the `n` args, the varstack and the environment.
macro_rules! expect_n_objs {
    ($name:ident, $n:expr, $e:expr) => {
        (
            Box::new(|evaluated_args, unevaluated_args, _| {
                let warnings = if unevaluated_args.is_empty() {
                    vec![]
                } else {
                    vec![format!(
                        "Provided {} unevaluated arguments although none are expected.",
                        unevaluated_args.len()
                    )]
                };
                if evaluated_args.len() != $n {
                    Err(format!(
                        "Wrong number of evaluated arguments provided for function '{}' (expected {}, got {}).",
                        stringify!($name),
                        $n,
                        evaluated_args.len()
                    ))
                } else {
                    $e(evaluated_args).map(|value| Status{value, warnings})
                }
            }),
            ($n, 0, false)
        )
    };
}

/// For examples, see the use of the macro in `default_functions`.
macro_rules! apply_matrix_fn {
    ($name:ident, $e:expr) => {
        (
            Box::new(|evaluated_args, unevaluated_args, _| {
                let warnings = if unevaluated_args.is_empty() {
                    vec![]
                } else {
                    vec![format!(
                        "Provided {} unevaluated arguments although none are expected.",
                        unevaluated_args.len()
                    )]
                };
                if evaluated_args.len() != 1 {
                    Err(format!(
                        "Wrong number of evaluated arguments provided for function '{}' (expected 1, got {}).",
                        stringify!($name),
                        evaluated_args.len()
                    ))
                } else {
                    if let Object::Matrix(mat) = &evaluated_args[0] {
                        $e(mat.$name(), &mat).map(|value| Status{value, warnings})
                    }
                    else { Err(format!("Wrong type for argument of function '{}' (expected Matrix).", stringify!($name))) }
                }
            }),
            (1, 0, false)
        )
    };
}

/// This approach is needed because direct functions cannot be cloned (and we need to clone the environment in some scenarios).
/// Therefore, we want to use `&DirectFunction` instead of `DirectFunction` in `Env`; but this requires the direct functions
/// to be permanently stored at a fixed location. This location is here.
/// 
/// Note that the user can't create new direct functions, so this approach works.
#[allow(clippy::type_complexity)]
pub static DEFAULT_DIRECT_FUNCTIONS: LazyLock<[(DirectFunction, (usize, usize, bool)); 24]> = LazyLock::new(|| [
    expect_n_objs!(sign, 1, |args: &[Object]| {
        match &args[0] {
            Object::Real(x) => Ok(Object::Real(if *x >= 0.0 {1.0} else {-1.0})),
            Object::Vector(v) => Ok(Object::Vector(v.transform(|x| if x >= 0.0 {1.0} else {-1.0}))),
            Object::Matrix(m) => Ok(Object::Matrix(m.transform(|x| if x >= 0.0 {1.0} else {-1.0}))),
            other => Err(format!("Undefined operation `sign` for operand {:?}.", other))
        }
    }),

    float_1_function!(exp),
    float_1_function!(ln),
    expect_n_objs!(log, 2, |args: &[Object]| {
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

    expect_n_objs!(eig, 1, |args: &[Object]| {
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

    // ___helper_prod_rule
    // Takes an object `x_val`, expressions `x`, `i`, `a(x)`, `b(x)`, `f(i,x)` and `f'(i,x)`, a `&mut Env env` and a `&VarStack`.
    // Afterwards, there can be an arbitrary additional amount of expressions: these will be considered as conditions.
    // Then, returns `\sum_{i=a(x), all_conditions(i)}^{b(x)} f'(i,x) * \prod_{j=a(x), j!=i, all_conditions(j)}^{b(x)} f(j,x)`.
    (
        Box::new(|evaluated_args, unevaluated_args, context| {
            if evaluated_args.len() != 1 || unevaluated_args.len() < 6 {
                return Err(format!("Wrong number of arguments provided for function '___helper_prod_rule' (expected ==1 evaluated and >=6 unevaluated, got {}, {} respectively).", evaluated_args.len(), unevaluated_args.len()));
            }
            if context.is_none() {
                return Err("Function '___helper_prod_rule' needs `VarStack` and `Env`.".to_string());
            }
            let (base_stack, env) = context.unwrap(); // Safe
            let (x, i) = (unevaluated_args[0].expect_ident()?, unevaluated_args[1].expect_ident()?);
            let [a_x, b_x, f, f_prime] = &unevaluated_args[2..6] else {unreachable!()};
            let conditions = &unevaluated_args[6..];

            let mut warnings = Vec::<String>::new();
            let varstack = VarStack::Frame { vars: &HashMap::from([(x, &evaluated_args[0])]), parent: base_stack };
            // `b` can be non-integer but, to the best of my knowledge, there is no canonical understanding for non-integer `a`.
            let (a, b) = (
                eval(a_x, &varstack, env)?.unpack_into(&mut warnings).expect_int::<i64>()?,
                eval(b_x, &varstack, env)?.unpack_into(&mut warnings).expect_float()?.floor() as i64
            );
            if a > b {return Ok(Status{value: Object::Real(0.0), warnings});} // Empty sum
            let mut sum = 0.0;
            'outer: for i_val in a..=b {
                let i_as_obj = Object::Real(i_val as f64);
                let vs = VarStack::Frame {
                    vars: &HashMap::from([(i, &i_as_obj)]),
                    parent: &varstack
                };
                // Check if all conditions are met (o/w skip)
                for cond in conditions {
                    // We put a cap on the number of warnings emitted in order to not saturate the console
                    match eval(cond, &vs, env)?.unpack_into_with_cap(&mut warnings, 10) {
                        Object::Real(1.0) => {}
                        Object::Real(0.0) => {continue 'outer;}
                        other => return Err(format!("Expected 1 or 0 when evaluating condition, got {:?}.", other))
                    }
                }
                // Start with f'(i, x)
                let mut product = eval(f_prime, &vs, env)?.unpack_into_with_cap(&mut warnings, 10).expect_float()?;
                // Multiply with all f(j, x)
                'middle: for j_val in a..=b {
                    if j_val == i_val {continue;}
                    let j_as_obj = Object::Real(j_val as f64);
                    let vsj = VarStack::Frame {
                        vars: &HashMap::from([(i, &j_as_obj)]),
                        parent: &varstack
                    };
                    for cond in conditions {
                        match eval(cond, &vsj, env)?.unpack_into_with_cap(&mut warnings, 10) {
                            Object::Real(1.0) => {}
                            Object::Real(0.0) => {continue 'middle;}
                            other => return Err(format!("Expected 1 or 0 when evaluating condition, got {:?}.", other))
                        }
                    }
                    product *= eval(f, &vsj, env)?.unpack_into_with_cap(&mut warnings, 10).expect_float()?;
                }
                sum += product;
            }
            Ok(Status{value: Object::Real(sum), warnings})
        }),
        (1, 6, false)
    ),

    // del
    // Arbitrary amount of args (technically including 0), all not evaluated. Only accepts identifiers.
    // Warns if an argument is passed that is not an identifier.
    (
        Box::new(|evaluated_args, unevaluated_args, context| {
            if context.is_none() {
                return Err("Function 'del' needs `Env`.".to_string());
            }
            let warnings = if evaluated_args.is_empty() {
                vec![]
            } else {
                vec![format!(
                    "Provided {} evaluated arguments although none are expected.",
                    evaluated_args.len()
                )]
            };
            let env = context.unwrap().1;
            let mut unknown_identifiers = Vec::<&String>::new();
            let mut none_identifiers = Vec::<&Expression>::new();
            for arg in unevaluated_args {
                match arg {
                    Expression::Identifier(id) => {
                        // No bugs with short-circuiting here because if `constants.remove(id)` is `Some`, then `id` was in `constants`,
                        // so it can't be in `functions` too.
                        if env.constants.remove(id).is_none() && env.functions.remove(id).is_none() {
                            unknown_identifiers.push(id);
                        }
                    }
                    other => none_identifiers.push(other)
                }
            }
            if unknown_identifiers.is_empty() && none_identifiers.is_empty() {
                Ok(Status{value: Object::Success, warnings})
            } else {
                let mut err_str = "Some arguments couldn't be deleted from the environment.".to_string();
                if !unknown_identifiers.is_empty() {
                    err_str.push_str("\nNot present in environment: ");
                    err_str.push_str(unknown_identifiers.into_iter().join(", ").as_str());
                    err_str.push('.');
                }
                if !none_identifiers.is_empty() {
                    err_str.push_str("\nNot identifiers: ");
                    err_str.push_str( none_identifiers.into_iter().join(", ").as_str());
                    err_str.push('.');
                }
                Err(err_str)
            }
        }),
        (0, 0, false)
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
        "___helper_prod_rule", "del"
    ].into_iter().enumerate().map(
        |(i, n)|
        (n.to_string(), FunctionRepr::Direct(&DEFAULT_DIRECT_FUNCTIONS[i].0, DEFAULT_DIRECT_FUNCTIONS[i].1))
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