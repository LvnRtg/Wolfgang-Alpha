//! The basic syntax is the natural one with usual operator precedence. A few special features are the following.
//! - Matrices can be initialized by typing `[1, 2, 3 \ 4, 5, 6 \ 7, 8, 9]` where the rows will be `[1,2,3]`, `[4,5,6]` and `[7,8,9]` respectively.
//!   The backslash can be used interchangeably with a semicolon `;`, even within the same matrix.
//! - Vectors can be initialized by typing either `(1, 2, 3)` (particular to vectors) or `[1; 2; 3]` or `[1 \ 2 \ 3]` (as one would initialize a matrix with only one column).
//! - Definition of constants: `identifier := expr`, where `expr` can be any expression that can be evaluated at the time of the definition.<br/>
//!   This returns the evaluation of `expr`, so one can write e.g. `(x := 2) + 1` to obtain `3` as output and define `x` simultaneously.<br/>
//!   If `identifier` is already a defined constant, this will re-define it and permanently suppress the old value.
//! - Definition of functions: `f(x, y) := 2x + y`. If e.g. `x` already exists as a constant/function, this will be ignored for the sake of the function's definition.
//!   The `x` on the RHS of the definition will always be the `x` passed as argument, not the constant.<br/>
//!   If one wants to include a constant from the current environment, simply type `f(y) := 2x + y` where `x` is a pre-defined constant. Note that the
//!   current value of `x` will be captured at the time of the definition; if you change `x` later on, `f` will still use its old value.
//! - Test if two values are equal: `expr = other_expr` where both expressions must be evaluable to an `Object`. Very small errors are tolerated.
//! - The same works for `<`, `<=`, `>` and `>=`. The strict comparison signs do _not_ tolerate small errors.
//!   As for equality, two vectors/matrices of the same size satisfy a comparison iff all of their components satisfy it.
//! - Test if two functions are equal by evaluating at `n` random points: `f(x) ={n} g(x)` where `n` can be any expression evaluable to a float (will then be rounded to the nearest integer).
//!   The same works for `<`, `<=`, `>` and `>=`.
//! - Partially differentiate: `d/dx (x^3 + 2x + 1)` returns `3x^2 + 2` as expression. The parentheses are not needed when differentiating e.g. a monome.<br/>
//!   The output can be stored in a function: `f(x) := d/dx ...`.<br/>
//!   Differentiating a function with a matrix/vector as output will differentiate component-wise and return the corresponding matrix/vector-valued function.<br/>
//!   If the differentiated function `f(x)` outputs a vector/matrix, the output will be the function `p \mapsto D_x f(p)[1]`, that is, the direction to differentiate in will be set to 1.0 by default.
//!   This means the syntax is still accepted although not recommended.
//! - Directionally differentiate: multiple syntaxes:
//!     - `D_x <expr1> (expr2)[expr3]` leads to `point := {x: expr2}` and `direction := {x: expr3}`.
//!     - `D_{x, y} <expr1> (expr2x, expr2y)[expr3x, expr3y]` leads to `point := {x: expr2x, y expr2y}` and analogously for `direction`. Analogously for any higher number of variables.
//!     - `D f(4)[2]`: free variables are set to be the argnames of `f` (these will be the keys of the hashmap, cf. implementation).
//!     - `D <expr> (expr_1, ..., expr_n)[expr'_1, ..., expr'_m]`: collect all unknown identifiers within `expr` into a vector in ascending alphabetic order `x_1, ..., x_l`.
//!       If `l=m=n`, infer that these should be the keys of the hashmaps (cf. implementation). Otherwise, return `Err`.
//! - Notice that the token `!` acts as both the `not` operator and the factorial operator. In context, one can always differentiate between the two, with one minor downside:
//!   the syntax `x * (!y)` cannot be shortened to `x !y` (since these spaces disappear while tokenizing, one would not be able to differentiate this with `(x!) * y`).

use std::iter::Peekable;
use std::collections::HashMap;
use std::collections::HashSet;
use std::str::Chars;
use rand::Rng;

use crate::math;
use crate::math::{Comparison, BinaryOperation, UnaryOperation, Object, Expression, FunctionRepr}; // Common types that will be used several times


const DEFAULT_TESTEQ_REPETITIONS: i64 = 100;


#[derive(Debug)]
pub enum VarStack<'a> {
    Empty,
    Frame {
        vars: &'a HashMap<&'a String, &'a Object>,
        parent: &'a VarStack<'a>,
    },
}

impl<'a> VarStack<'a> {
    pub fn lookup(&self, key: &String) -> Option<&Object> {
        match self {
            VarStack::Empty => None,
            VarStack::Frame { vars, parent } => {
                vars.get(key).copied().or_else(|| parent.lookup(key))
            }
        }
    }
}


/// The tokens appearing in the grammar used by the calculator.
/// 
/// The goal is for the tokens to be entirely context free. Therefore, e.g. functions, matrices and vectors aren't tokens, they can only be later crafted as `Expression`.
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Identifier(String),
    /// To be understood as an explicit sequence of digits (without sign, possibly including a '.').
    Number(f64),
    Plus,
    Minus,
    Asterisk,
    Slash,
    /// Quotient
    DoubleSlash,
    /// Remainder
    Percent,
    /// Exponentiation
    Circumflex,
    /// Contains a vector of tokens if this uses the syntax `f(x) ={expr} g(x)` (where the tokens are the tokenized expression `expr`), otherwise `None`.
    Comparison(Comparison, Option<Vec<Token>>), // TODO: allow to restrict the domain of the free variables
    /// :=
    Assign,
    LParenthesis,
    RParenthesis,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    /// Separation of arguments in function call / vector / matrix
    Comma,
    /// Separation of rows in matrix or separation of commands
    Semicolon,
    /// Alternative row separator in matrix
    Backslash,
    /// |
    Pipe,
    /// The keyword "if"
    If,
    /// The keyword "else"
    Else,
    /// &
    Ampersand,
    /// &&
    DoubleAmpersand,
    /// ||
    DoublePipe,
    /// !
    ExclamationMark,
    EOF,
}

#[derive(Debug)]
pub struct Parser {
    pub tokens: Vec<Token>,
    pub pos: usize,
}

/// Tokenizes a given input string recursively using the internal function 'tokenize_recursive'.
pub fn tokenize(input: &str) -> Vec<Token> {
    let mut chars = input.chars().peekable();
    tokenize_recursive(&mut chars, Vec::<char>::new())
}

/// Substitute for if-let chains
fn char_peek_equals(chars: &mut Peekable<Chars>, ch: char, consume_if_true: bool) -> bool {
    match chars.peek() {
        Some(&c) if c == ch => {
            if consume_if_true {
                chars.next();
            }
            true
        }
        _ => false
    }
}

/// Called when a comparison token was just parsed.
/// 
/// If it exists, returns the parameter between brackets in tokenized form. Otherwise, returns `None`.
fn parse_comparison_parameter(chars: &mut Peekable<Chars>) -> Option<Vec<Token>> {
    if char_peek_equals(chars, '{', true) {
        let toks = tokenize_recursive(chars, vec!['}']);
        chars.next(); // Consume right bracket
        Some(toks)
    }
    else {
        None // No parameter given
    }
}

fn identifier_to_token(ident: String) -> Token {
    match ident.as_str() {
        "if" => Token::If,
        "else" => Token::Else,
        _ => Token::Identifier(ident)
    }
}

fn tokenize_recursive(chars: &mut Peekable<Chars>, return_early: Vec<char>) -> Vec<Token> {
    let mut tokens = Vec::new();
    while let Some(&ch) = chars.peek() {
        if return_early.contains(&ch) {
            // Do NOT consume this char, it is important to know what it was.
            return tokens;
        }
        match ch {
            c if c.is_whitespace() => { chars.next(); }
            '+' => { chars.next(); tokens.push(Token::Plus); }
            '-' => { chars.next(); tokens.push(Token::Minus); }
            '*' => { chars.next(); tokens.push(Token::Asterisk); }
            '/' => {
                chars.next();
                tokens.push(if char_peek_equals(chars, '/', true) { Token::DoubleSlash } else { Token::Slash });
            }
            '%' => { chars.next(); tokens.push(Token::Percent); }
            '^' => { chars.next(); tokens.push(Token::Circumflex); }
            ':' => {
                chars.next();
                if char_peek_equals(chars, '=', true) {
                    tokens.push(Token::Assign);
                }
                else {
                    panic!("Unexpected lone colon (must be followed by '=').");
                }
            }
            '=' => {
                chars.next();
                tokens.push(Token::Comparison(Comparison::Eq, parse_comparison_parameter(chars)));
            }
            '<' => {
                chars.next();
                tokens.push(Token::Comparison(
                    if char_peek_equals(chars, '=', true) {Comparison::Le} else {Comparison::Lt},
                    parse_comparison_parameter(chars))
                );
            }
            '>' => {
                chars.next();
                tokens.push(Token::Comparison(
                    if char_peek_equals(chars, '=', true) {Comparison::Ge} else {Comparison::Gt},
                    parse_comparison_parameter(chars))
                );
            }
            '(' => { chars.next(); tokens.push(Token::LParenthesis); }
            ')' => { chars.next(); tokens.push(Token::RParenthesis); }
            '[' => { chars.next(); tokens.push(Token::LBracket); }
            ']' => { chars.next(); tokens.push(Token::RBracket); }
            '{' => { chars.next(); tokens.push(Token::LBrace); }
            '}' => { chars.next(); tokens.push(Token::RBrace); }
            ',' => { chars.next(); tokens.push(Token::Comma); }
            ';' => { chars.next(); tokens.push(Token::Semicolon); }
            '!' => { chars.next(); tokens.push(Token::ExclamationMark); }
            '\\' => { chars.next(); tokens.push(Token::Backslash); }
            '&' => {
                chars.next();
                match chars.peek() {
                    Some('&') => {chars.next(); tokens.push(Token::DoubleAmpersand);}
                    Some(_) => {tokens.push(Token::Ampersand);}
                    _ => {}
                }
            }
            '|' => {
                chars.next();
                match chars.peek() {
                    Some('|') => {chars.next(); tokens.push(Token::DoublePipe);}
                    Some(_) => {tokens.push(Token::Pipe);}
                    _ => {}
                }
            }
            c if c.is_alphanumeric() || c == '_' => {
                // Encountered first character of a "word" (potential identifier, but maybe of the form "2x").
                // First, parse all leading digits (which may include a point), then the rest of the word.
                let mut digits = String::new();
                let mut had_point = false; // Only one point allowed per constant
                while let Some(&nc) = chars.peek() {
                    if nc.is_ascii_digit() {
                        digits.push(nc);
                        chars.next();
                    }
                    else if !had_point && nc == '.' {
                        had_point = true;
                        digits.push(nc);
                        chars.next();
                    }
                    else {break;}
                }
                let mut ident = String::new();
                while let Some(&nc) = chars.peek() {
                    if nc.is_alphabetic() || nc == '_' {
                        ident.push(nc);
                        chars.next();
                    } else {break}
                };
                // Currently, accepted syntaxes are "numberIDENTIFIER" and "IDENTIFIER" only.
                if !digits.is_empty() {
                    tokens.push(Token::Number(digits.parse::<f64>().unwrap()));
                    if !ident.is_empty() {
                        tokens.push(Token::Asterisk); // Since "2x" is to be parsed as "2*x"
                        tokens.push(identifier_to_token(ident));
                    }
                }
                else { // Note that if this case of the match block is even called, either 'digits' or 'ident' has to be non-trivial.
                    tokens.push(identifier_to_token(ident));
                }
            }
            other => {
                panic!("Unexpected character in input: {:?}.", other);
            }
        }
    }
    tokens.push(Token::EOF);
    tokens
}

fn get_unknown_identifiers(expr: &Expression, identifiers: &mut HashSet<String>, constants: &HashMap<String, Object>) {
    match expr {
        Expression::None | Expression::Number(_) => {}
        Expression::Identifier(x) => {
            if !constants.contains_key(x) {
                identifiers.insert(x.clone());
            }
        }
        Expression::Vector(v) | Expression::Matrix(.., v) | Expression::Function(_, v) => {
            for x in v {
                get_unknown_identifiers(x, identifiers, constants);
            }
        }
        Expression::UnaryOperation(_, x) => get_unknown_identifiers(x, identifiers, constants),
        Expression::BinaryOperation(x, _, y) | Expression::Assignment(x, y) => {
            get_unknown_identifiers(x, identifiers, constants);
            get_unknown_identifiers(y, identifiers, constants);
        }
        Expression::PartialDerivative(wrt, x) => {
            let b = identifiers.contains(wrt);
            get_unknown_identifiers(x, identifiers, constants);
            if !b {identifiers.remove(wrt);}
        }
        Expression::DirectionalDerivative(vars, x, ..) => {
            let rm: Vec<&String> = vars.iter().filter(|v| !identifiers.contains(*v)).collect(); // Need to collect to call next line
            get_unknown_identifiers(x, identifiers, constants);
            for v in rm {identifiers.remove(v);}
        }
        Expression::IfElse(x, y, z) => {
            get_unknown_identifiers(x, identifiers, constants);
            get_unknown_identifiers(y, identifiers, constants);
            get_unknown_identifiers(z, identifiers, constants);
        }
    }
}

impl Parser {
    pub fn new(input: &str) -> Self {
        Parser { tokens: tokenize(input), pos: 0 }
    }

    fn peek(&self) -> &Token {
        &self.tokens[self.pos]
    }
    fn next(&mut self) -> Token {
        let t = self.tokens[self.pos].clone();
        if self.pos < self.tokens.len() { self.pos += 1; }
        t
    }

    /// Uses the following functions to parse expressions separated by commas until the token `closer` follows an expression.
    /// 
    /// Panics if an unexpected token is encountered.
    fn parse_comma_expression(&mut self, closer: &Token, constants: &mut HashMap<String, Object>, functions: &mut HashMap<String, FunctionRepr>) -> Vec<Expression> {
        let mut exprs = Vec::<Expression>::new();
        loop {
            exprs.push(self.parse_expression(0, constants, functions));
            match self.next() {
                Token::Comma => {},
                some if &some == closer => {break;},
                other => panic!("Expected ')', found {:?}", other),
            }
        }
        exprs
    }

    /// Allows to recursively parse vectors of tokens.
    fn parse_expression(&mut self, min_precedence: u8, constants: &mut HashMap<String, Object>, functions: &mut HashMap<String, FunctionRepr>) -> Expression {
        // First, determine the LHS of the next operation to execute.
        // This is either an identifier, a number or a further expression between parentheses.
        let mut lhs = match self.next() {
            Token::Minus => Expression::UnaryOperation(UnaryOperation::Neg, Box::new(self.parse_expression(5, constants, functions))),
            Token::ExclamationMark // An exclamation mark before an expected expression signifies a `not` operator
                => Expression::UnaryOperation(UnaryOperation::Not, Box::new(self.parse_expression(3, constants, functions))),
            Token::Identifier(id) if id == "D" || id == "D_" => { // Total derivative
                // Expected tokens: ("D" | "D_{...}") <FunctionExpr> (<point>) [<direction>].
                // For a list of all accepted syntaxes, see the documentation of the program's syntax.
                let mut argnames = Vec::<String>::new();
                if id == "D_" { // Then, parse argnames now. Otherwise, we need knowledge of `function_expr` for this.
                    match self.next() {
                        Token::LBrace => {
                            loop {
                                match self.next() {
                                    Token::Identifier(ident) => {argnames.push(ident);}
                                    _ => panic!("Expected variable name in `D_{{...}}`.")
                                }
                                match self.next() {
                                    Token::Comma => {},
                                    Token::RBrace => {break;},
                                    other => panic!("Expected ',' or '}}', found {:?}", other),
                                }
                            }
                        }
                        Token::Identifier(ident) => {argnames.push(ident);}
                        other => panic!("Expected '{{' or identifier after `D_`, got {:?}", other)
                    }
                }
                let mut function_expr = self.parse_expression(8, constants, functions);
                // At this point, the next token can either be a parenthesis or a bracket.
                let point = match (self.peek(), &mut function_expr) {
                    // This case means the point is yet to parse.
                    (Token::LParenthesis, _) => {
                        if id == "D" { // Then, only parse arguments now (since we need to know `function_expr` for this)
                            let mut identifiers = HashSet::<String>::new();
                            get_unknown_identifiers(&function_expr, &mut identifiers, constants);
                            argnames = identifiers.into_iter().collect::<Vec<String>>();
                            argnames.sort_unstable();
                        }
                        self.next();
                        self.parse_comma_expression(&Token::RParenthesis, constants, functions)
                    }
                    // The following case is only valid if `function_expr` is a `Expression::Function(f, x)`, in which case `x` is the actual point
                    // and the true arguments given to `f` should be its argnames in order (if f has direct representation, use x_1, ..., x_n
                    // instead where n is the length of `x`). The syntax `D f` should be used here (and not e.g. `D_x f`);
                    // if `D_x f` was used, it is treated as `D` (formally, the previous value of `argnames` (here, `["x"]`) is overwritten).
                    (Token::LBracket, Expression::Function(name, args)) if functions.contains_key(name) => {
                        argnames = match functions.get(name).unwrap() {
                            FunctionRepr::ByExpression(argnames, _) => argnames.clone(),
                            FunctionRepr::Direct(_) => (0..args.len()).map(|i| format!("x_{}", i)).collect()
                        };
                        std::mem::replace(args, argnames.iter().map(|x| Expression::Identifier(x.clone())).collect())
                    }
                    _ => panic!("Missing point to differentiate at in total derivative expression.")

                };
                match self.next() {
                    Token::LBracket => {}
                    other => {panic!("Expected '[', found {:?}", other);}
                };
                let direction = self.parse_comma_expression(&Token::RBracket, constants, functions);
                Expression::DirectionalDerivative(argnames, Box::new(function_expr), point, direction)
            }
            Token::Identifier(x) => {
                // We have to check whether this will be a function call: we judge this to be the case iff the next token is an LParenthesis and
                // either the identifier `x` is contained in `functions` or we are on the LHS of an assignment operator. There is no efficient
                // way to know yet whether there will be an assignment operator on the same precedence level as we currently are. Therefore,
                // this case will be handled afterwards by 'eval'. So, we only have to check the case:
                match self.peek() {
                    Token::LParenthesis if functions.contains_key(&x) || x.starts_with("___diff_num_") => {
                        self.next();
                        Expression::Function(x, self.parse_comma_expression(&Token::RParenthesis, constants, functions))
                    }
                    _ => Expression::Identifier(x)
                }
            }
            Token::Number(x) => Expression::Number(x),
            Token::LParenthesis => {
                // Parse expression between parentheses recursively. It could just be a single expression of multiple entries separated by commas.
                let mut entries = Vec::<Expression>::new();
                loop {
                    entries.push(self.parse_expression(0, constants, functions));
                    match self.next() {
                        Token::Comma => {},
                        Token::RParenthesis => {break;},
                        other => panic!("Expected ')', found {:?}", other),
                    }
                }
                match entries.len() {
                    0 => Expression::None,
                    1 => entries.pop().unwrap(), // I decided to not box the elements rightaway since the case `entries.len() == 1` is more common.
                    _ => Expression::Vector(entries.into_iter().collect())
                }
            }
            Token::LBracket => {
                let mut entries = Vec::<Expression>::new();
                let mut m: usize = 1;
                let mut n: usize = 0;
                loop {
                    entries.push(self.parse_expression(0, constants, functions));
                    match self.next() {
                        Token::Comma => {},
                        Token::Semicolon | Token::Backslash => {
                            if n == 0 { // If n has not been set yet
                                n = entries.len(); // Set n
                            }
                            m += 1;
                        }
                        Token::RBracket => {
                            if n == 0 { // If n has not been set yet
                                n = entries.len(); // Set n
                            }
                            break;
                        },
                        other => panic!("Expected ')', found {:?}", other),
                    }
                }
                if n == 1 {
                    Expression::Vector(entries)
                }
                else {
                    Expression::Matrix(m, n, entries)
                }
            }
            Token::Pipe => { // As for parentheses
                let inner = self.parse_expression(0, constants, functions);
                match self.next() {
                    Token::Pipe => Expression::UnaryOperation(UnaryOperation::Abs, Box::new(inner)),
                    other => panic!("Expected closing '|', found {:?}", other),
                }
            }
            Token::If => {
                let condition = self.parse_expression(0, constants, functions); // Will return wenn LBrace is encountered.
                assert!(matches!(self.next(), Token::LBrace), "Expected '{{' after condition {condition}.");
                let iftrue = self.parse_expression(0, constants, functions);
                assert!(matches!((self.next(), self.next(), self.next()), (Token::RBrace, Token::Else, Token::LBrace)), "Expected '}} else {{' after first `if` case.");
                let iffalse = self.parse_expression(0, constants, functions);
                assert!(matches!(self.next(), Token::RBrace), "Expected '}}' after `else` case.");
                Expression::IfElse(Box::new(condition), Box::new(iftrue), Box::new(iffalse))
            }
            other => panic!("Unexpected token where expression expected: {:?}", other)
        };

        // Then, parse the RHS recursively.
        loop {
            let (op, prec, consume) = match self.peek() {
                Token::Plus => (BinaryOperation::Add, 5, true),
                Token::Minus => (BinaryOperation::Sub, 5, true),
                Token::Asterisk => (BinaryOperation::Mul, 6, true),
                Token::LParenthesis => (BinaryOperation::Mul, 6, false), // Expressions such as "2(x+1)" are parsed as "2*(x+1)".
                Token::LBracket => (BinaryOperation::Mul, 6, false), // Same if a vector/matrix follows.
                Token::Identifier(_) => (BinaryOperation::Mul, 6, false), // Expressions such as "(x+1)y" are parsed as "(x+1)*y".
                Token::Slash => (BinaryOperation::Div, 6, true),
                Token::DoubleSlash => (BinaryOperation::Quo, 6, true),
                Token::Percent => (BinaryOperation::Rem, 6, true),
                Token::Circumflex => (BinaryOperation::Pow, 7, true),
                Token::Assign => (BinaryOperation::Add, 0, true), // We don't need any operation here, so 'Add' is just a placeholder to simplify notation
                Token::Comparison(..) => {
                    // Slightly longer code here because we have to consume the token right now in order to avoid copying
                    if let Token::Comparison(c, param) = self.next() { // Will always succeed
                        (BinaryOperation::Comp(
                            c,
                            // `unwrap` in the following line is acceptable since the `map` at the beginning ensures that `param` is `Some`
                            param.map(|p| Box::new(Parser{tokens: p, pos: 0}.parse(constants, functions).into_iter().next().unwrap()))
                        ), 4, false) // `false` since the token was already consumed
                    }
                    else { // Will never happen anyway
                        (BinaryOperation::Add, 0, false)
                    }
                },
                Token::DoublePipe => (BinaryOperation::Or, 1, true),
                Token::DoubleAmpersand => (BinaryOperation::And, 2, true),

                Token::ExclamationMark => { // An exclamation mark after an expression signifies a factorial operation
                    lhs = Expression::UnaryOperation(UnaryOperation::Factorial, Box::new(lhs));
                    continue;
                }

                // I wrote the following cases down explicitely so adding new tokens requires reviewing this code.
                // Note that in this context, Token::Pipe is the closing pipe, since the opening one would have been consumed in the definition of `lhs`.
                // Note also that the opening brace is tied to specific syntaxes (e.g. `if else` blocks) and thus cannot be found "freely".
                Token::Number(_) | Token::Comma | Token::Semicolon | Token::Backslash | Token::LBrace | Token::Ampersand
                | Token::If | Token::Else | Token::EOF
                | Token::RParenthesis | Token::RBracket | Token::RBrace | Token::Pipe
                    => { break; }
            };

            if prec < min_precedence { // If we encountered an operator of lower precedence, the current expression ends here.
                break;
            }
            if consume { self.next(); } // Implicit operators, e.g. left parentheses interpreted as "*(", do not lead to the consumption of the next token.

            // The RHS can only contain operators of strictly larger precedence, so we parse it with parameter 'prec+1'.
            let rhs = self.parse_expression(prec + 1, constants, functions);

            lhs = if prec == 0 { // Assignment operator
                Expression::Assignment(Box::new(lhs), Box::new(rhs))
            }
            else {
                // Check whether the expression is of the form `d/dx`, which would signify a derivative.
                // The check requires that op == Div, lhs = Identifier("d") and rhs is an Identifier starting with "d".
                match (&op, &lhs, &rhs) {
                    (
                        BinaryOperation::Div,
                        Expression::Identifier(lhs_ident),
                        Expression::Identifier(ident)
                    ) if lhs_ident == "d" && ident.len() > 1 => {
                        // Parse the function to differentiate recursively.
                        Expression::PartialDerivative(ident[1..].to_string(), Box::new(self.parse_expression(8, constants, functions)))
                    }
                    _ => Expression::BinaryOperation(Box::new(lhs), op, Box::new(rhs))
                }
            };
        }

        lhs
    }

    /// Parse the entire given vector of tokens recursively while consuming it.
    /// 
    /// Note that we need knowledge of the environment here, since e.g. "x(y+1)" can be interpreted either as function call or as multiplication;
    /// knowledge of the environment resolves such ambiguities.
    /// 
    /// Note that in the following table, there may be gaps between numbers, which allow to insert to operators more easily.
    /// 
    /// <table>
    /// <tr> <th>Operators<th/> <th>Precedence<th/> </tr>
    /// <trd> <td>:=<td/> <td>0<td/> </tr>
    /// <trd> <td>||<td/> <td>1<td/> </tr>
    /// <trd> <td>&&<td/> <td>2<td/> </tr>
    /// <trd> <td>! (not)<td/> <td>3<td/> </tr>
    /// <trd> <td><, >, <=, >=, ==<td/> <td>4<td/> </tr>
    /// <tr> <td>+, -<td/> <td>5<td/> </tr>
    /// <tr> <td>*, /, //, %<td/> <td>6<td/> </tr>
    /// <tr> <td>^<td/> <td>7<td/> </tr>
    /// <tr> <td>d/dx, D<td/> <td>8<td/> </tr>
    /// </table>
    pub fn parse(&mut self, constants: &mut HashMap<String, Object>, functions: &mut HashMap<String, FunctionRepr>) -> Vec<Expression> {
        let mut exprs = Vec::<Expression>::new();
        loop {
            let expr = self.parse_expression(0, constants, functions);
            exprs.push(expr);
            match self.peek() {
                Token::EOF => {return exprs;},
                Token::Semicolon => {self.next(); continue;}
                other => panic!("Unexpected trailing token: {:?}", other),
            }
        }
    }
}

/// When an function definition is encountered, the expression on the RHS is processed in a special way.
/// Generally, it has to be cloned (cleanest way to work with the 'eval' function below), which is the main action this function performs.
/// 
/// Parsing the expression recursively, every identifier that is NOT declared as an argument of the function is replaced
/// by the constant it represents in the current environment. Identifiers can be declared as arguments even if they exist in the environment;
/// the environmental value will then be ignored. Moreover, every identifier that is declared as an argument is prefixed with three underscores
/// (this will be needed for evaluation). For example, if 'constants = {"x": 1, "y": 2}', the RHS of the literal expression
/// "f(y, z) := x + 3*y + z" will become "1 + 3*___tmp_y + ___tmp_z".
/// 
/// I have decided that if the definition depends on another function (say, "f(x, y) = g(x) + y"), the other function shall
/// not be replaced by its literal expression. It makes sense to me to capture the current values of free variables because
/// if this were not intended, one could simply include them as parameters, but this solution isn't available for functions
/// (yet), hence the decision.
/// 
/// This cannot avoid cloning objects (e.g. matrices) because if a variable (say, "x" in above example) is changed later, we
/// still want the function to behave the same, so we'd have to keep the old value stored somewhere anyway.
/// However, is doesn't matter if this function is relatively expensive to call since function definitions are rare.
/// 
/// Note: for the definition of constants, this is not necessary, since constants have to be computable at the moment they are defined.
pub fn parse_function_definition(
    expr: &Expression,
    argument_names: &Vec<String>,
    constants: &HashMap<String, Object>
) -> Expression {
    match expr {
        Expression::None => Expression::None,
        Expression::Identifier(x) => {
            if argument_names.contains(x) {
                Expression::Identifier(format!("___tmp_{}", x))
            } else if let Some(y) = constants.get(x) {
                match y { // As discussed in this function's documentation, clone will be necessary here
                    Object::Success | Object::Undefined => Expression::None, // This would be a syntax error
                    Object::Float(x) => Expression::Number(*x),
                    Object::Vector(v) => Expression::Vector(v.values.iter().map(|entry| Expression::Number(*entry)).collect()),
                    Object::Matrix(x) => Expression::Matrix(
                        x.m, x.n,
                        x.iter_values().map(|entry| Expression::Number(*entry)).collect()
                    ),
                    Object::LiteralExpression(e) => e.clone()
                }
            }
            else {
                Expression::Identifier(x.clone())
            }
        },
        Expression::Number(x) => Expression::Number(*x),
        Expression::Vector(x) => Expression::Vector(x.iter().map(|x| parse_function_definition(x, argument_names, constants)).collect()),
        Expression::Matrix(m, n, x) => Expression::Matrix(*m, *n, x.iter().map(|x| parse_function_definition(x, argument_names, constants)).collect()),
        Expression::UnaryOperation(op, rhs) => Expression::UnaryOperation(
            *op,
            Box::new(parse_function_definition(rhs, argument_names, constants))
        ),
        Expression::BinaryOperation(lhs, op, rhs) => Expression::BinaryOperation(
            Box::new(parse_function_definition(lhs, argument_names, constants)),
            op.clone(),
            Box::new(parse_function_definition(rhs, argument_names, constants))
        ),
        Expression::Function(function_name, args) => Expression::Function(
            function_name.clone(),
            args.iter().map(|x| parse_function_definition(x, argument_names, constants)).collect()
        ),
        Expression::Assignment(lhs, rhs) => Expression::Assignment(
            Box::new(parse_function_definition(lhs, argument_names, constants)),
            Box::new(parse_function_definition(rhs, argument_names, constants))
        ),
        Expression::PartialDerivative(wrt, expr) => Expression::PartialDerivative(
            wrt.clone(),
            Box::new(parse_function_definition(expr, argument_names, constants))
        ),
        Expression::DirectionalDerivative(vars, expr, point, direction) => Expression::DirectionalDerivative(
            vars.clone(),
            Box::new(parse_function_definition(expr, argument_names, constants)),
            point.iter().map(|x| parse_function_definition(x, argument_names, constants)).collect(),
            direction.iter().map(|x| parse_function_definition(x, argument_names, constants)).collect()
        ),
        Expression::IfElse(x, y, z) => Expression::IfElse(
            Box::new(parse_function_definition(x, argument_names, constants)),
            Box::new(parse_function_definition(y, argument_names, constants)),
            Box::new(parse_function_definition(z, argument_names, constants)),
        )
    }
}

/// Parses the expression `expr` recursively and collects all identifiers that are neither in `constants` nor in `extra_vars` into a HashSet `modified_identifiers`.
/// 
/// Returns whether or not anything was modified. The parameter `modified_anything` should be set to `false` for the first call and will then be passed down recursively.
fn list_unknown_identifiers(
    expr: &Expression,
    extra_vars: &VarStack,
    constants: &HashMap<String, Object>,
    modified_identifiers: &mut HashSet<String>,
    modified_anything: bool
) -> bool {
    match expr {
        Expression::None | Expression::Number(_) | Expression::Vector(_) | Expression::Matrix(..) => modified_anything,
        Expression::Identifier(x) => {
            if !constants.contains_key(x) && extra_vars.lookup(x).is_none() {
                modified_identifiers.insert(x.clone());
                true
            }
            else { modified_anything }
        }
        Expression::UnaryOperation(_, expr) => list_unknown_identifiers(expr, extra_vars, constants, modified_identifiers, modified_anything),
        Expression::BinaryOperation(lhs, _, rhs) => {
            // This will modify something iff at least either LHS or RHS is modified.
            list_unknown_identifiers(lhs, extra_vars, constants, modified_identifiers, modified_anything)
            || list_unknown_identifiers(rhs, extra_vars, constants, modified_identifiers, modified_anything)
        }
        Expression::Function(_, args) => {
            args.iter().map(|arg| list_unknown_identifiers(arg, extra_vars, constants, modified_identifiers, modified_anything)).collect::<Vec<_>>().iter().any(|x| *x)
        }
        Expression::Assignment(_, rhs) => list_unknown_identifiers(rhs, extra_vars, constants, modified_identifiers, modified_anything), // Do not modify the LHS of assignment expressions
        Expression::PartialDerivative(_, expr) => list_unknown_identifiers(expr, extra_vars, constants, modified_identifiers, modified_anything),
        Expression::DirectionalDerivative(_, expr, point, direction) => {
            list_unknown_identifiers(expr, extra_vars, constants, modified_identifiers, modified_anything)
            || point.iter().map(|v| list_unknown_identifiers(v, extra_vars, constants, modified_identifiers, modified_anything)).collect::<Vec<_>>().iter().any(|x| *x)
            || direction.iter().map(|v| list_unknown_identifiers(v, extra_vars, constants, modified_identifiers, modified_anything)).collect::<Vec<_>>().iter().any(|x| *x)
        },
        Expression::IfElse(x, y, z) => {
            // This will modify something iff at least either LHS or RHS is modified.
            list_unknown_identifiers(x, extra_vars, constants, modified_identifiers, modified_anything)
            || list_unknown_identifiers(y, extra_vars, constants, modified_identifiers, modified_anything)
            || list_unknown_identifiers(z, extra_vars, constants, modified_identifiers, modified_anything)
        }
    }
}



/// Evaluates a given expression and returns the computed value (as reference, see below).
/// Requires knowledge of the environment, i.e. the hashmaps 'constants' and 'functions'.
/// 1. If the expression can be computed directly (e.g. "2+3" or "5*x" where constants.contains("x")), returns its value as type 'Object'.
/// 2. If the expression is a valid definition (e.g. "x := 7" or "f(x) := 5*x+2"), modifies the environment accordingly and returns 'Object.Success'.
///    
/// Moreover, `extra_vars` allows to specify identifiers that temporarily should have a certain value. Each hashmap in `extra_vars` should map
/// identifiers to objects. The outer `Vec` acts as stack: this function first searches for identifers in the last hashmap in `extra_vars`, then
/// in the fore-last, etc. until a match is found or the start of the vector is reached. The reason for this becomes apparent in the case
/// `Expression::Function`: for recursive function calls, it is simpler to pass more and more hashmap references to `eval` than to modify
/// the existing hashmap and later revert it to its old value.
/// 
/// If the evaluation fails, returns the corresponding error message (wrapped in a 'Result').
pub fn eval(
    expr: &Expression,
    extra_vars: &VarStack,
    constants: &mut HashMap<String, Object>,
    functions: &mut HashMap<String, FunctionRepr>
) -> Result<Object, String> {
    match expr {
        Expression::None => Err("Received empty expression.".to_string()),
        Expression::Identifier(ident) => {
            // First, iterate `extra_vars` in reverse order and search for `ident`.
            if let Some(x) = extra_vars.lookup(ident) {
                Ok(x.clone())
            }
            // If nothing is found, look in `constants`.
            else if let Some(x) = constants.get(ident) {
                Ok(x.clone())
                // We only call 'clone' for every time a variable from 'constants' is used, which can only happen so often
                // since the user still has to enter at least one character per time it is used. Therefore,
                // even if these are large matrices, it is a totally acceptable runtime.
            }
            // If still, nothing is found, this is an error.
            else {
                Err(format!("Unknown identifier: {:?}", ident))
            }
        },
        Expression::Number(x) => Ok(Object::Float(*x)),
        Expression::Vector(entries) => {
            Ok(Object::Vector(math::Vector{values: entries.iter().map(
                |x| match eval(x, extra_vars, constants, functions) {
                    Ok(Object::Float(entry)) => Ok(entry),
                    Ok(_) => Err(format!("Entry {} is not a float.", x)),
                    Err(e) => Err(format!("Couldn't evaluate entry {}. Traceback: {}", x, e))
                }
            ).collect::<Result<Vec<_>, _>>()?}))
        },
        Expression::Matrix(m, n, entries) => {
            Ok(Object::Matrix(math::Matrix::from(*m, *n, entries.iter().map(
                |x| match eval(x, extra_vars, constants, functions) {
                    Ok(Object::Float(entry)) => Ok(entry),
                    Ok(_) => Err(format!("Entry {} is not a float.", x)),
                    Err(e) => Err(format!("Couldn't evaluate entry {}. Traceback: {}", x, e))
                }
            ).collect::<Result<Vec<_>, _>>()?)))
        },
        Expression::UnaryOperation(op, rhs) => {
            match op {
                UnaryOperation::Neg => {
                    match eval(rhs, extra_vars, constants, functions)? {
                        Object::Success => Ok(Object::Success),
                        Object::Undefined => Err("Operation 'Neg' not valid for undefined operand.".to_string()),
                        Object::Float(x) => Ok(Object::Float(-x)),
                        Object::Vector(x) => Ok(Object::Vector(-&x)),
                        Object::Matrix(x) => Ok(Object::Matrix(-&x)),
                        Object::LiteralExpression(e) => Ok(Object::LiteralExpression(Expression::UnaryOperation(UnaryOperation::Neg, Box::new(e)))),
                    }
                }
                UnaryOperation::Not => {
                    match eval(rhs, extra_vars, constants, functions)? {
                        Object::Success => Ok(Object::Success),
                        Object::Undefined => Err("Operation 'Not' not valid for undefined operand.".to_string()),
                        Object::Float(x) => Ok(Object::Float(if x == 0.0 {1.0} else {0.0})),
                        Object::Vector(v) => Ok(Object::Vector(v.into_new(|x| if x == 0.0 {1.0} else {0.0}))),
                        Object::Matrix(m) => Ok(Object::Matrix(m.into_new(|x| if x == 0.0 {1.0} else {0.0}))),
                        Object::LiteralExpression(e) => Ok(Object::LiteralExpression(Expression::UnaryOperation(UnaryOperation::Not, Box::new(e)))),
                    }
                }
                UnaryOperation::Factorial => {
                    match eval(rhs, extra_vars, constants, functions)? {
                        Object::Success => Ok(Object::Success),
                        Object::Undefined => Err("Operation 'Factorial' not valid for undefined operand.".to_string()),
                        Object::Float(x) => Ok(Object::Float(x)), // TODO: add gamma function here and in the two lines below
                        Object::Vector(x) => Ok(Object::Vector(-&x)),
                        Object::Matrix(x) => Ok(Object::Matrix(-&x)),
                        Object::LiteralExpression(e) => Ok(Object::LiteralExpression(Expression::UnaryOperation(UnaryOperation::Factorial, Box::new(e)))),
                    }
                }
                UnaryOperation::Abs => {
                    match eval(rhs, extra_vars, constants, functions)? {
                        Object::Success => Ok(Object::Success),
                        Object::Undefined => Err("Operation 'Abs' not valid for undefined operand.".to_string()),
                        Object::Float(x) => Ok(Object::Float(x.abs())),
                        Object::Vector(x) => Ok(Object::Float(x.norm())),
                        Object::Matrix(x) => Ok(Object::Float(x.det())),
                        Object::LiteralExpression(e) => Ok(Object::LiteralExpression(Expression::UnaryOperation(UnaryOperation::Abs, Box::new(e)))),
                    }
                }
            }
        },
        Expression::BinaryOperation(lhs, op, rhs) => {
            match (&**lhs, &**rhs, op) {
                // If the operation is a comparison and at least one of `lhs`, `rhs` is `Expression::Function` (which we'll call `this`; we'll call the remaining one `other`)...
                (a @ Expression::Function(..), b, BinaryOperation::Comp(_, param)) | (b, a @ Expression::Function(..), BinaryOperation::Comp(_, param)) => {
                    let this = a.clone(); let other = b.clone();
                    let mut free_variables = HashSet::<String>::new(); // This gathers all variables that will have to be randomized afterwards
                    list_unknown_identifiers(&this, extra_vars, constants, &mut free_variables, false);
                    let other_only_needs_single_eval = !list_unknown_identifiers(&other, extra_vars, constants, &mut free_variables, false);
                    let mut other_eval = Object::Success; // Placeholder
                    // If `other` doesn't contain any free variables (<=> the second `list_unknown_identifiers` call above actually modified the expression),
                    // it suffices to evaluate `other` once.
                    // Then, evaluating every time would be inefficient, especially if many values will be tested.
                    // Therefore, it makes sense to check whether this is the case beforehand, and if so, simply evaluate once and save the value for later.
                    if other_only_needs_single_eval {
                        other_eval = eval(&other, extra_vars, constants, functions)?;
                    }

                    // If a number of repetitions is given as `param` under the form of an expression, evaluate it and use it. Otherwise, use `DEFAULT_TESTEQ_REPETITIONS`
                    let repetitions = param.as_ref()
                        .map(|p| match eval(p, extra_vars, constants, functions) {
                            Ok(Object::Float(x)) => Ok(x.round() as i64),
                            Err(e) => Err(format!("Couldn't resolve number of repetitions `{}`. Traceback: {}", p, e)),
                            _ => Err(format!("Couldn't resolve number of repetitions `{}` to float.", p))
                        })
                        .unwrap_or(Ok(DEFAULT_TESTEQ_REPETITIONS))
                        ?;

                    for _ in 0..repetitions {
                        let random_numbers: Vec<Object> = (0..free_variables.len()).map(|_| Object::Float(rand::thread_rng().gen_range(-1000.0..1000.0))).collect();
                        let tmp_vars: HashMap<&String, &Object> = free_variables.iter().enumerate().map(|(i, ident)| (ident, &random_numbers[i])).collect();
                        let new_stack = VarStack::Frame { vars: &tmp_vars, parent: extra_vars };
                        let first_eval = eval(&this, &new_stack, constants, functions)
                            .map_err(|e| format!("Couldn't evaluate `{}` with environment {:?}. Traceback: {}", this, tmp_vars, e)) // Add information to the error message
                            ?;
                        if !other_only_needs_single_eval {
                            other_eval = eval(&other, &new_stack, constants, functions)
                                .map_err(|e| format!("Couldn't evaluate `{}` with environment {:?}. Traceback: {}", this, tmp_vars, e))
                                ?;
                        }
                        // If the objects' comparison yields `false`, return that. If the objects aren't comparable, return the appropriate error. Otherwise, continue.
                        match math::objects::try_operation(&first_eval, &other_eval, op) {
                            Ok(Object::Float(0.0)) => { return Ok(Object::Float(0.0)); }
                            Err(_) => { return Err(format!("Couldn't compare `{}` and `{}` (arising from environment {:?}).", first_eval, other_eval, constants)); }
                            _ => {}
                        }
                    }
                    Ok(Object::Float(1.0)) // If nothing previous returned, then the expressions fulfill the comparison.
                }
                // Otherwise...
                _ => math::objects::try_operation(&eval(lhs, extra_vars, constants, functions)?, &eval(rhs, extra_vars, constants, functions)?, op)
            }
        },
        Expression::Function(function_name, given_arg_exprs) => {
            // Note this case can only occur when we actually have a function call, not an assignment.
            // We can be sure about this because the assignment operator is given the lowest priority level by the tokenizer
            // and the case `Expression::Assignment` in this function does not call itself recursively on the LHS
            // of an assignment operation.
            
            // If `function_name` is of the form with `___diff_num_f`, this isn't a function contained in `functions` but the request to numerically differentiate `f`.
            if let Some(real_function_name) = function_name.strip_prefix("___diff_num_") {
                // Ensure that `given_arg_exprs` is even. There is a special case where an uneven number is tolerated: if only a single argument
                // is provided, simply set the direction as 1.0 (default for 1d derivative).
                let mut tmp: Vec<Expression>;
                let arg_exprs = if given_arg_exprs.len() % 2 != 0 {
                    if given_arg_exprs.len() == 1 {
                        tmp = given_arg_exprs.clone();
                        tmp.push(Expression::Number(1.0));
                        &tmp
                    }
                    else {
                        return Err("___diff_num_{{...}} takes an even number of arguments.".to_string()); // See splitting of arguments below
                    }
                } else { given_arg_exprs };
                let rm = functions.remove(real_function_name);
                let res = match rm {
                    Some(FunctionRepr::Direct(ref f)) => {
                        // The given arguments should then have the format `point <concat> direction`, so we have to split the arguments
                        // into two parts (splitting in the middle of the array which we ensured has even size).
                        let point = (0..arg_exprs.len()/2)
                            .map(|i| eval(&arg_exprs[i], extra_vars, constants, functions))
                            .collect::<Result<Vec<_>, _>>()?;
                        let direction = (arg_exprs.len()/2..arg_exprs.len())
                            .map(|i| eval(&arg_exprs[i], extra_vars, constants, functions))
                            .collect::<Result<Vec<_>, _>>()?;
                        math::differentiation::numerical_directional_derivative(f, point, direction)
                    }
                    Some(FunctionRepr::ByExpression(..)) => {
                        Err("Don't use ___diff_num_ to differentiate a function that has an explicit defining expression.".to_string())
                    }
                    None => Err(format!("No such function: {:?}", function_name))
                };
                if let Some(x) = rm {
                    functions.insert(real_function_name.to_string(), x);
                }
                res
            }

            // We're doing a little trick which is to remove the corresponding function from `functions` and reinserting it at the end.
            // This is necessary since `functions` can't be borrowed as mutable and immutable twice at the same time (caused by recursive call to `eval`).
            // By transfer of ownership, this is a very cheap operation compared to cloning a `FunctionRepr` because the latter's
            // defining expression (if present) can be highly nested.
            else if let Some(func) = functions.remove(function_name) {
                let ret_value = eval_function(function_name, &func, given_arg_exprs, extra_vars, constants, functions);
                functions.insert(function_name.clone(), func); // Reinsert the removed function
                ret_value
            }
            else {Err(format!("No such function: {:?}", function_name))}
        },
        Expression::Assignment(lhs, rhs) => {
            // Note that names starting with "___" are forbidden (prefix "___tmp_" reserved for temporary variables, prefix "___diff_" for the derivative of a function with direct representation).
            /// Helper function. We need this because multiple syntax structures lead to a function definition:
            /// - `Expression::Function(function_name, args)`
            /// - `Expression::BinaryOperation(Identifier(function_name), BinaryOperation::Mul, Identifier(arg))`
            /// - `Expression::BinaryOperation(Identifier(function_name), BinaryOperation::Mul, Vector(args))`
            fn define_function(
                function_name: &String,
                unparsed_args: std::slice::Iter<'_, Expression>,
                rhs: &Expression,
                constants: &mut HashMap<String, Object>,
                functions: &mut HashMap<String, FunctionRepr>
            ) -> Result<Object, String> {
                if function_name.starts_with("___") { Err("Names starting with \"___\" are forbidden".to_string()) }
                else if function_name == "D" || function_name.starts_with("D_") { Err("The name \"D\" and identifiers starting with \"D_\" are reserved for the total derivative.".to_string()) }
                else {
                    // First, check that all declared arguments on the LHS are in fact just identifiers.
                    let mut argnames = unparsed_args.into_iter()
                        .map(|lh_arg|
                            if let Expression::Identifier(x) = lh_arg {Ok(x.clone())}
                            else {Err("Parameters in LHS of function definition must be identifiers.".to_string())}
                        )
                        .collect::<Result<Vec<_>, _>>()?;
                    // Next, parse the RHS as explained in the documentation of `parse_function_definition`.
                    let expr = parse_function_definition(rhs, &argnames, constants);
                    // The argument names have to be prefixed too
                    argnames = argnames.into_iter().map(|x| format!("___tmp_{}", x)).collect();
                    functions.insert(function_name.clone(), FunctionRepr::ByExpression(
                        argnames,
                        expr
                    ));
                    // The .clone() above is no problem since function definitions are rare (in the sense that performance doesn't matter for this).
                    // Lastly, if there was already a function `__diff_{function_name}` present in `functions` (cf. `analytic_derivative`).
                    // If so, it is now outdated, so remove it.
                    functions.remove(&format!("___diff_num_{}", function_name));
                    Ok(Object::Success)
                }
            }
            match &**lhs {
                Expression::Identifier(ident) => { // Definition of a constant
                    if ident.starts_with("___") { Err("Names starting with \"___\" are forbidden".to_string()) }
                    else if ident == "D" || ident.starts_with("D_") { Err("The name \"D\" and identifiers starting with \"D_\" are reserved for the total derivative.".to_string()) }
                    else {
                        if let Ok(obj_rhs) = eval(rhs, extra_vars, constants, functions) {
                            // The '.clone()' in below line is due to the fact that we want to save the value on one hand (within 'constants')
                            // but also return it (e.g. the expression "x := 5" should not only define x as 5 but also return the value 5 so that
                            // one can write "... * (x := ...)" to save intermediate results).
                            constants.insert(ident.clone(), obj_rhs.clone());
                            Ok(obj_rhs)
                        }
                        else { Err(format!("Couldn't evaluate expression {}", **rhs)) }
                    }
                }
                Expression::BinaryOperation(x, BinaryOperation::Mul, y) => {
                    match (&**x, &**y) {
                        (Expression::Identifier(function_name), Expression::Identifier(_))
                            => define_function(function_name, std::slice::from_ref(&**y).iter(), rhs, constants, functions),
                        (Expression::Identifier(function_name), Expression::Vector(args))
                            => define_function(function_name, args.iter(), rhs, constants, functions),
                        _ => Err(format!("Invalid LHS of assignment expression: {}", **lhs))
                    }
                }
                Expression::Function(function_name, unparsed_args)
                    => define_function(function_name, unparsed_args.iter(), rhs, constants, functions),
                _ => {
                    Err(format!("Invalid LHS of assignment expression: {}", **lhs))
                }
            }
        }
        Expression::PartialDerivative(wrt, expr) => {
            math::differentiation::analytic_partial_derivative(expr, wrt, functions).map(Object::LiteralExpression)
        }
        Expression::DirectionalDerivative(vars, expr, point_exprs, direction_exprs) => {
            if point_exprs.len() != direction_exprs.len() {
                return Err("Point and direction of directional derivative must have the same dimension.".to_string());
            }
            let point = point_exprs.iter()
                .map(|p| eval(p, extra_vars, constants, functions))
                .collect::<Result<Vec<_>, _>>()?;
            let direction = direction_exprs.iter()
                .map(|p| eval(p, extra_vars, constants, functions))
                .collect::<Result<Vec<_>, _>>()?;
            math::differentiation::analytic_directional_derivative(vars, expr, &point, &direction, constants, functions)
        }
        Expression::IfElse(condition, iftrue, iffalse) => {
            match eval(condition, extra_vars, constants, functions) {
                Ok(Object::Float(1.0)) => eval(iftrue, extra_vars, constants, functions),
                Ok(Object::Float(0.0)) => eval(iffalse, extra_vars, constants, functions),
                Ok(x) => Err(format!("Couldn't evaluate condition {} to 0 or 1; got {x}", &**condition)),
                other => other
            }
        }
    }
}

fn eval_function(
    function_name: &String,
    func: &FunctionRepr,
    given_arg_exprs: &[Expression],
    extra_vars: &VarStack,
    constants: &mut HashMap<String, Object>,
    functions: &mut HashMap<String, FunctionRepr>
) -> Result<Object, String> {
    match func {
        FunctionRepr::ByExpression(ref argnames, ref defining_expr) => {
            if given_arg_exprs.len() != argnames.len() {
                return Err(format!("Wrong number of arguments provided for function '{}' (expected {}, got {}).", function_name, argnames.len(), given_arg_exprs.len()));
            }
            // Put all temporary variables (arguments) into a new hashmap and add it to `extra_vars`.
            let tmp_var_evals = given_arg_exprs.iter().enumerate().map(
                |(i, given_arg_expr)| {
                    eval(given_arg_expr, extra_vars, constants, functions)
                    .map_err(|e| format!("Couldn't resolve argument {} := {}. Traceback: {}", argnames[i], given_arg_expr, e))
                }
            ).collect::<Result<Vec<_>, _>>()?;
            let tmp_vars: HashMap<&String, &Object> = tmp_var_evals.iter().enumerate().map(|(i, x)| (&argnames[i], x)).collect();
            let new_stack = VarStack::Frame { vars: &tmp_vars, parent: extra_vars };
            eval(defining_expr, &new_stack, constants, functions)
        }
        FunctionRepr::Direct(ref f) => {
            (*f)(&given_arg_exprs.iter()
                .map(|arg_expr| eval(arg_expr, extra_vars, constants, functions))
                .collect::<Result<Vec<_>, _>>()?)
        }
    }
}