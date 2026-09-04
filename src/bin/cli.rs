use wolfgang_alpha::{math, defaults, repl};
use std::io::{self, Write, BufRead};

fn main() {
    let mut env = math::Env {
        constants: defaults::default_constants(),
        functions: defaults::default_functions(),
    };

    println!("Wolfgang Alpha REPL");
    println!("Type 'exit' or 'quit' to exit.");
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    print!("> ");
    let _ = stdout.flush();
    for line in stdin.lock().lines() {
        match line {
            Ok(input) => {
                let input = input.trim().to_string();
                if input.is_empty() {
                    print!("> ");
                    let _ = stdout.flush();
                    continue;
                }
                if input == "exit" || input == "quit" {
                    break;
                }
                let (output, _) = repl::eval_line(&input, &mut env);
                for line in output {
                    println!("{}", line);
                }
                print!("> ");
                let _ = stdout.flush();
            }
            Err(e) => {
                eprintln!("Error reading input: {}", e);
                break;
            }
        }
    }
}
