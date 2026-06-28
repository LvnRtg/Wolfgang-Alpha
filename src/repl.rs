use crate::lang;
use crate::math;

/// Shared evaluation engine: tokenizes, parses, and evaluates `input` against `env`.
/// Returns the display lines to show to the user.
pub fn eval_line(input: &str, env: &mut math::Env) -> Vec<String> {
    let tokens = match lang::tokenize(input) {
        Ok(x) => x,
        Err(e) => return vec![format!("[ERROR] {e}")],
    };
    let mut parser = lang::Parser::from(tokens);
    let mut output = Vec::<String>::new();
    match parser.parse(env) {
        Ok(expressions) => {
            for expr in expressions {
                if expr == math::Expression::Identifier("debug".to_string()) {
                    output.push(format!("Constants: {:?}", env.constants));
                    output.push(format!("Functions: {:?}", env.functions));
                } else {
                    match lang::eval(&expr, &math::VarStack::Empty, env) {
                        Ok(obj) => {
                            output = obj.to_multline();
                        }
                        Err(e) => {
                            output.push(format!("[ERROR] {}", e));
                        }
                    }
                }
            }
        }
        Err(e) => {
            output.push(format!("[ERROR] {}", e));
        }
    }
    output
}
