use crate::lang;
use crate::math;

/// Shared evaluation engine: tokenizes, parses, and evaluates `input` against `env`.
/// 
/// Returns the display lines to show to the user as well as a bool that is `true` iff an error occurred.
pub fn eval_line(input: &str, env: &mut math::Env) -> (Vec<String>, bool) {
    let tokens = match lang::tokenize(input) {
        Ok(x) => x,
        Err(e) => return (vec![e], false),
    };
    let mut parser = lang::Parser::from(tokens);
    let mut output = Vec::<String>::new();
    let mut is_error = false;
    while let Some(res) = parser.parse_next(env) {
        match res {
            Ok(expr) => {
                if expr == math::Expression::Identifier("debug".to_string()) {
                    output.push(format!("Constants: {:?}", env.constants));
                    output.push(format!("Functions: {:?}", env.functions));
                } else {
                    match lang::eval(&expr, &math::VarStack::Empty, env) {
                        Ok(status) => {
                            output.append(&mut status.into_multline());
                        }
                        Err(e) => {
                            is_error = true;
                            output.push(e);
                        }
                    }
                }
            }
            Err(e) => {
                is_error = true;
                output.push(e);
            }
        }
    }
    (output, is_error)
}
