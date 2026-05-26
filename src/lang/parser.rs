//! Responsible for parsing tokenized inputs into expressions, which includes respecting operator precedence.
//! 
use std::collections::HashSet;

use crate::math::{BinaryOperation, UnaryOperation, Expression, FunctionRepr, Env};
use crate::lang::lexer::Token;


#[derive(Debug)]
pub struct Parser {
    pub tokens: Vec<Token>,
    pub pos: usize,
}


fn get_unknown_identifiers(expr: &Expression, identifiers: &mut HashSet<String>, env: &Env) {
    match expr {
        Expression::None | Expression::Number(_) => {}
        Expression::Identifier(x) => {
            if !env.constants.contains_key(x) {
                identifiers.insert(x.clone());
            }
        }
        Expression::Vector(v) | Expression::Matrix(.., v) | Expression::Function(_, v) => {
            for x in v {
                get_unknown_identifiers(x, identifiers, env);
            }
        }
        Expression::UnaryOperation(_, x) => get_unknown_identifiers(x, identifiers, env),
        Expression::BinaryOperation(x, _, y) | Expression::Assignment(x, y) => {
            get_unknown_identifiers(x, identifiers, env);
            get_unknown_identifiers(y, identifiers, env);
        }
        Expression::PartialDerivative(wrt, x) => {
            let b = identifiers.contains(wrt);
            get_unknown_identifiers(x, identifiers, env);
            if !b {identifiers.remove(wrt);}
        }
        Expression::DirectionalDerivative(vars, x, ..) => {
            let rm: Vec<&String> = vars.iter().filter(|v| !identifiers.contains(*v)).collect(); // Need to collect to call next line
            get_unknown_identifiers(x, identifiers, env);
            for v in rm {identifiers.remove(v);}
        }
        Expression::IfElse(x, y, z) => {
            get_unknown_identifiers(x, identifiers, env);
            get_unknown_identifiers(y, identifiers, env);
            get_unknown_identifiers(z, identifiers, env);
        }
    }
}

impl Parser {
    pub fn from(tokens: Vec<Token>) -> Self {
        Parser { tokens, pos: 0 }
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
    fn parse_comma_expression(&mut self, closer: &Token, env: &mut Env) -> Result<Vec<Expression>, String> {
        let mut exprs = Vec::<Expression>::new();
        loop {
            exprs.push(self.parse_expression(0, None, env)?);
            match self.next() {
                Token::Comma => {},
                some if &some == closer => {break;},
                other => panic!("Expected ')', found {:?}", other),
            }
        }
        Ok(exprs)
    }

    /// Allows to recursively parse vectors of tokens.
    /// 
    /// If `expect_closer` is `Some(x)` and `x` is encountered in a place of an operator, the function returns early instead.
    /// This is usually unnecessary (e.g. expressions between parentheses are parsed just fine without this), but is strictly required
    /// when parsing an expression between e.g. double pipes (`||`), because this token cannot necessarily be distinguished from the "or" operator.
    fn parse_expression(&mut self, min_precedence: u8, expect_closer: Option<Token>, env: &mut Env) -> Result<Expression, String> {
        // First, determine the LHS of the next operation to execute.
        // This is either an identifier, a number or a further expression between parentheses.
        let mut lhs = match self.next() {
            Token::Minus => Expression::UnaryOperation(UnaryOperation::Neg, Box::new(self.parse_expression(5, None, env)?)),
            Token::ExclamationMark // An exclamation mark before an expected expression signifies a `not` operator
                => Expression::UnaryOperation(UnaryOperation::Not, Box::new(self.parse_expression(3, None, env)?)),
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
                let mut function_expr = self.parse_expression(8, None, env)?;
                // At this point, the next token can either be a parenthesis or a bracket.
                let point = match (self.peek(), &mut function_expr) {
                    // This case means the point is yet to parse.
                    (Token::LParenthesis, _) => {
                        if id == "D" { // Then, only parse arguments now (since we need to know `function_expr` for this)
                            let mut identifiers = HashSet::<String>::new();
                            get_unknown_identifiers(&function_expr, &mut identifiers, env);
                            argnames = identifiers.into_iter().collect::<Vec<String>>();
                            argnames.sort_unstable();
                        }
                        self.next();
                        self.parse_comma_expression(&Token::RParenthesis, env)?
                    }
                    // The following case is only valid if `function_expr` is a `Expression::Function(f, x)`, in which case `x` is the actual point
                    // and the true arguments given to `f` should be its argnames in order (if f has direct representation, use x_1, ..., x_n
                    // instead where n is the length of `x`). The syntax `D f` should be used here (and not e.g. `D_x f`);
                    // if `D_x f` was used, it is treated as `D` (formally, the previous value of `argnames` (here, `["x"]`) is overwritten).
                    (Token::LBracket, Expression::Function(name, args)) if env.functions.contains_key(name) => {
                        argnames = match env.functions.get(name).unwrap() {
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
                let direction = self.parse_comma_expression(&Token::RBracket, env)?;
                Expression::DirectionalDerivative(argnames, Box::new(function_expr), point, direction)
            }
            Token::Identifier(x) => {
                // We have to check whether this will be a function call: we judge this to be the case iff the next token is an LParenthesis and
                // either the identifier `x` is contained in `functions` or we are on the LHS of an assignment operator. There is no efficient
                // way to know yet whether there will be an assignment operator on the same precedence level as we currently are. Therefore,
                // this case will be handled afterwards by 'eval'. So, we only have to check the case:
                match self.peek() {
                    Token::LParenthesis if env.functions.contains_key(&x) || x.starts_with("___diff_num_") => {
                        self.next();
                        Expression::Function(x, self.parse_comma_expression(&Token::RParenthesis, env)?)
                    }
                    _ => Expression::Identifier(x)
                }
            }
            Token::Number(x) => Expression::Number(x),
            Token::LParenthesis => {
                // Parse expression between parentheses recursively. It could just be a single expression of multiple entries separated by commas.
                let mut entries = Vec::<Expression>::new();
                loop {
                    entries.push(self.parse_expression(0, None, env)?);
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
                let mut current_n: usize = 0;
                loop {
                    current_n += 1;
                    entries.push(self.parse_expression(0, None, env)?);
                    match self.next() {
                        Token::Comma => {},
                        Token::Semicolon | Token::Backslash => {
                            if n == 0 { // If n has not been set yet
                                n = current_n; // Set n
                            }
                            else if n != current_n {
                                panic!("Got matrix row of wrong length (expected {n}, got {}.", current_n);
                            }
                            current_n = 0;
                            m += 1;
                        }
                        Token::RBracket => {
                            if n == 0 { // If n has not been set yet
                                n = entries.len(); // Set n
                            }
                            else if n != current_n {
                                panic!("Got matrix row of wrong length (expected {n}, got {}.", current_n);
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
                let inner = self.parse_expression(0, None, env)?;
                match self.next() {
                    Token::Pipe => Expression::UnaryOperation(UnaryOperation::Abs, Box::new(inner)),
                    other => panic!("Expected closing '|', found {:?}", other),
                }
            }
            Token::DoublePipe => { // In this context: opener of a norm
                let inner = self.parse_expression(0, Some(Token::DoublePipe), env)?;
                match self.next() {
                    Token::DoublePipe => {
                        match self.peek() {
                            Token::Identifier(ident) if ident.starts_with('_') => {
                                let norm_type = if ident == "_" {
                                    self.next();
                                    match self.next() {
                                        Token::Identifier(a) => Expression::Identifier(a),
                                        Token::Number(a) => Expression::Number(a),
                                        Token::LBrace => {
                                            let res = self.parse_expression(0, None, env)?;
                                            match self.next() {
                                                Token::RBrace => {},
                                                other => {return Err(format!("Expected closing '}}', found {:?}", other));}
                                            }
                                            res
                                        }
                                        other => {return Err(format!("Expected norm type after '||_', found {:?}", other));}
                                    }
                                } else {
                                    let cloned_ident = ident.clone();
                                    let mut chars = cloned_ident.chars(); chars.next();
                                    self.next();
                                    Expression::Identifier(chars.collect::<String>())
                                };
                                Expression::UnaryOperation(UnaryOperation::Norm(Some(Box::new(norm_type))), Box::new(inner))
                            }
                            _ => Expression::UnaryOperation(UnaryOperation::Norm(None), Box::new(inner))
                        }
                    }
                    other => panic!("Expected closing '||', found {:?}", other),
                }
            }
            Token::If => {
                let condition = self.parse_expression(0, None, env)?; // Will return wenn LBrace is encountered.
                assert!(matches!(self.next(), Token::LBrace), "Expected '{{' after condition {condition}.");
                let iftrue = self.parse_expression(0, None, env)?;
                assert!(matches!((self.next(), self.next(), self.next()), (Token::RBrace, Token::Else, Token::LBrace)), "Expected '}} else {{' after first `if` case.");
                let iffalse = self.parse_expression(0, None, env)?;
                assert!(matches!(self.next(), Token::RBrace), "Expected '}}' after `else` case.");
                Expression::IfElse(Box::new(condition), Box::new(iftrue), Box::new(iffalse))
            }
            other => panic!("Unexpected token where expression expected: {:?}", other)
        };

        // Then, parse the RHS recursively.
        if let Some(c) = expect_closer {
            if *self.peek() == c {
                // Importantly, do not consume the closer so the caller can check it.
                return Ok(lhs);
            }
        }
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
                        let parsed_param = if let Some(p) = param {
                            // `unwrap` in the following line is acceptable since the `map` at the beginning ensures that `param` is `Some`
                            Some(Box::new(Parser{tokens: p, pos: 0}.parse(env)?.into_iter().next().unwrap()))
                        } else {None};
                        (BinaryOperation::Comp(c, parsed_param), 4, false) // `false` since the token was already consumed
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

            // If we encountered an operator of lower precedence, the current expression ends here.
            if prec < min_precedence {
                break;
            }
            if consume { self.next(); } // Implicit operators, e.g. left parentheses interpreted as "*(", do not lead to the consumption of the next token.

            // The RHS can only contain operators of strictly larger precedence, so we parse it with parameter 'prec+1'.
            let rhs = self.parse_expression(prec + 1, None, env)?;
            lhs = match (lhs, &op, &rhs) {
                // Special case: the `^` operator is traditionally right-associative and not left-associative, i.e. 2^3^2 = 2^(3^2) and not (2^3)^2.
                (Expression::BinaryOperation(a, BinaryOperation::Pow, b), ..) if op == BinaryOperation::Pow
                    => Expression::BinaryOperation(
                        a,
                        op,
                        Box::new(Expression::BinaryOperation(b, BinaryOperation::Pow, Box::new(rhs)))
                    ),
                // Assignment operator
                (lhs, ..) if prec == 0 => Expression::Assignment(Box::new(lhs), Box::new(rhs)),
                // Partial derivative
                (
                    Expression::Identifier(lhs_ident),
                    BinaryOperation::Div,
                    Expression::Identifier(ident)
                ) if lhs_ident == "d" && ident.len() > 1
                    // Parse the function to differentiate recursively.
                    => Expression::PartialDerivative(ident[1..].to_string(), Box::new(self.parse_expression(8, None, env)?)),
                // Default
                (lhs, ..) => Expression::BinaryOperation(Box::new(lhs), op, Box::new(rhs))
            }
        }

        Ok(lhs)
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
    pub fn parse(&mut self, env: &mut Env) -> Result<Vec<Expression>, String> {
        let mut exprs = Vec::<Expression>::new();
        loop {
            let expr = self.parse_expression(0, None, env)?;
            exprs.push(expr);
            match self.peek() {
                Token::EOF => {return Ok(exprs);},
                Token::Semicolon => {self.next(); continue;}
                other => panic!("Unexpected trailing token: {:?}", other),
            }
        }
    }
}