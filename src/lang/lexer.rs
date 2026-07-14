//! Responsible for tokenizing input strings.

use std::iter::Peekable;
use std::str::Chars;
use std::fmt;

use crate::math::Comparison;

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
    Comparison(Comparison, Option<Vec<Token>>),
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
    #[allow(clippy::upper_case_acronyms)] EOF
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", match self {
            Token::Identifier(x) => x.clone(),
            Token::Number(x) => x.to_string(),
            Token::Plus => "+".to_string(),
            Token::Minus => "-".to_string(),
            Token::Asterisk => "*".to_string(),
            Token::Slash => "/".to_string(),
            Token::DoubleSlash => "//".to_string(),
            Token::Percent => "%".to_string(),
            Token::Circumflex => "^".to_string(),
            Token::Comparison(comp, opt) => if let Some(v) = opt {
                format!("{}[{:?}]", comp, v.iter().map(|x| x.to_string()).collect::<Vec<String>>().join(" "))
            } else {
                comp.to_string()
            },
            Token::Assign => ":=".to_string(),
            Token::LParenthesis => "(".to_string(),
            Token::RParenthesis => ")".to_string(),
            Token::LBracket => "[".to_string(),
            Token::RBracket => "]".to_string(),
            Token::LBrace => "{".to_string(),
            Token::RBrace => "}".to_string(),
            Token::Comma => ",".to_string(),
            Token::Semicolon => ";".to_string(),
            Token::Backslash => "\\".to_string(),
            Token::Pipe => "|".to_string(),
            Token::If => "if".to_string(),
            Token::Else => "else".to_string(),
            Token::Ampersand => "&".to_string(),
            Token::DoubleAmpersand => "&&".to_string(),
            Token::DoublePipe => "||".to_string(),
            Token::ExclamationMark => "!".to_string(),
            Token::EOF => "EOF".to_string(),
        })
    }
}

/// Tokenizes a given input string recursively using the internal function 'tokenize_recursive'.
pub fn tokenize(input: &str) -> Result<Vec<Token>, String> {
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
fn parse_comparison_parameter(chars: &mut Peekable<Chars>) -> Result<Option<Vec<Token>>, String> {
    if char_peek_equals(chars, '{', true) {
        let mut toks = tokenize_recursive(chars, vec!['}']);
        if let Ok(t) = toks.as_mut() {t.push(Token::EOF);}
        chars.next(); // Consume right bracket
        toks.map(Some)
    }
    else {
        Ok(None) // No parameter given
    }
}

fn identifier_to_token(ident: String) -> Token {
    match ident.as_str() {
        "if" => Token::If,
        "else" => Token::Else,
        _ => Token::Identifier(ident)
    }
}

fn tokenize_recursive(chars: &mut Peekable<Chars>, return_early: Vec<char>) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::new();
    while let Some(&ch) = chars.peek() {
        if return_early.contains(&ch) {
            // Do NOT consume this char, it is important to know what it was.
            return Ok(tokens);
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
                    return Err("Unexpected lone colon (must be followed by '=').".to_string());
                }
            }
            '=' => {
                chars.next();
                tokens.push(Token::Comparison(Comparison::Eq, parse_comparison_parameter(chars)?));
            }
            '<' => {
                chars.next();
                tokens.push(Token::Comparison(
                    if char_peek_equals(chars, '=', true) {Comparison::Le} else {Comparison::Lt},
                    parse_comparison_parameter(chars)?
                ));
            }
            '>' => {
                chars.next();
                tokens.push(Token::Comparison(
                    if char_peek_equals(chars, '=', true) {Comparison::Ge} else {Comparison::Gt},
                    parse_comparison_parameter(chars)?
                ));
            }
            '(' => { chars.next(); tokens.push(Token::LParenthesis); }
            ')' => { chars.next(); tokens.push(Token::RParenthesis); }
            '[' => { chars.next(); tokens.push(Token::LBracket); }
            ']' => { chars.next(); tokens.push(Token::RBracket); }
            '{' => { chars.next(); tokens.push(Token::LBrace); }
            '}' => { chars.next(); tokens.push(Token::RBrace); }
            ',' => { chars.next(); tokens.push(Token::Comma); }
            ';' => { chars.next(); tokens.push(Token::Semicolon); }
            '!' => {
                chars.next();
                if let Some('=') = chars.peek() {
                    chars.next();
                    tokens.push(Token::Comparison(Comparison::Neq, parse_comparison_parameter(chars)?));
                } else {
                    tokens.push(Token::ExclamationMark);
                }
            }
            '\\' => { chars.next(); tokens.push(Token::Backslash); }
            '&' => {
                chars.next();
                match chars.peek() {
                    Some('&') => {chars.next(); tokens.push(Token::DoubleAmpersand);}
                    _ => {tokens.push(Token::Ampersand);}
                }
            }
            '|' => {
                chars.next();
                match chars.peek() {
                    Some('|') => {chars.next(); tokens.push(Token::DoublePipe);}
                    _ => {tokens.push(Token::Pipe);}
                }
            }
            c if c.is_ascii_alphabetic() || c.is_ascii_digit() || c == '_' => {
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
                while let Some(&nc) = chars.peek() && (nc.is_ascii_alphabetic() || nc.is_ascii_digit() || nc == '_') {
                    ident.push(nc);
                    chars.next();
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
            other => return Err(format!("Unexpected character in input: {:?}.", other))
        }
    }
    tokens.push(Token::EOF);
    Ok(tokens)
}