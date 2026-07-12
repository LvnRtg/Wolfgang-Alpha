//! Responsible for parsing tokenized inputs into expressions, which includes respecting operator precedence.
//! 
use std::collections::HashSet;
use std::iter::Peekable;
use std::vec::IntoIter;

use crate::math::{BinaryOperation, Comparison, UnaryOperation, FoldedOperation, Expression, FunctionRepr, Env, VarStack};
use crate::lang::lexer::{Token, tokenize};

pub struct Parser {
    pub tokens: Peekable<IntoIter<Token>>
}

impl Parser {
    pub fn from(tokens: Vec<Token>) -> Self {
        Parser { tokens: tokens.into_iter().peekable() }
    }
    fn peek(&mut self) -> Result<&Token, String> {
        self.tokens.peek().ok_or("Expected token but none was found.".to_string())
    }
    fn next(&mut self) -> Result<Token, String> {
        self.tokens.next().ok_or("Expected token but none was found.".to_string())
    }

    // This approach is slightly more inefficient, but I keep this code in case a future syntax requires looking further ahead.
    // pub fn from(tokens: Vec<Token>) -> Self {
    //     Parser { tokens, pos: 0 }
    // }
    // fn peek(&self) -> &Token {
    //     &self.tokens[self.pos]
    // }
    // fn next(&mut self) -> Token {
    //     let t = self.tokens[self.pos].clone();
    //     if self.pos < self.tokens.len() { self.pos += 1; }
    //     t
    // }

    /// Consumes and compares the next token with `token`. Returns `()` if they are equal, otherwise an appropriate error string.
    /// Adding a parameter `context` allows to specify something in the returned error message.
    fn expect_token(&mut self, token: Token, context: Option<&str>) -> Result<(), String> {
        let next = self.next()?;
        if next != token {
            Err(format!("Expected {}{}, got {}.", token, context.unwrap_or_default(), next))
        } else {
            Ok(())
        }
    }

    /// Uses the following functions to parse expressions separated by commas until the token `closer` follows an expression.
    /// 
    /// Consumes the closer.
    fn parse_comma_expression(&mut self, closer: &Token, env: &mut Env) -> Result<Vec<Expression>, String> {
        let mut exprs = Vec::<Expression>::new();
        loop {
            exprs.push(self.parse_expression(0, None, env)?);
            match self.next()? {
                Token::Comma => {},
                some if &some == closer => {break;},
                other => {return Err(format!("Expected '{:?}', found {:?}.", closer, other));}
            }
        }
        Ok(exprs)
    }

    /// Expects either `LBrace, ..., RBrace` (then parses `...` and returns the result) or `Identifier(...) | Number(...)`
    /// (then returns `...` directly). All other syntaxes return `Err`.
    /// 
    /// For example, you'd call this after encountering `sum_`.
    fn expect_brace_expr(&mut self, env: &mut Env) -> Result<Expression, String> {
        match self.next()? {
            Token::Identifier(x) => Ok(Expression::Identifier(x)),
            Token::Number(x) => Ok(Expression::Number(x)),
            Token::LBrace => {
                let f: Box<dyn Fn(&Token) -> bool> = Box::new(|t: &Token| matches!(t, Token::RBrace));
                let res = self.parse_expression(0, Some(&f), env)?;
                self.expect_token(Token::RBrace, None)?;
                Ok(res)
            }
            other => Err(format!("Expected '{{', identifier or number; got {:?} instead.", other))
        }
    }
    /// Expects either `LBrace, ..., RBrace` (then parses `...`, splitting expressions between commas, and returns the result) or `Identifier(...) | Number(...)`
    /// (then returns `vec![...]` directly). All other syntaxes return `Err`.
    /// 
    /// For example, you'd call this after encountering `sum_`.
    fn expect_brace_expr_with_commas(&mut self, env: &mut Env) -> Result<Vec<Expression>, String> {
        match self.next()? {
            Token::Identifier(x) => Ok(vec![Expression::Identifier(x)]),
            Token::Number(x) => Ok(vec![Expression::Number(x)]),
            Token::LBrace => self.parse_comma_expression(&Token::RBrace, env),
            other => Err(format!("Expected '{{', identifier or number; got {:?} instead.", other))
        }
    }

    /// Allows to recursively parse vectors of tokens.
    /// 
    /// If `return_early_if` is `Some(f)` and a token `x` with `f(x) == true` is encountered in a place of an operator, the function returns early instead.
    /// This is usually unnecessary (e.g. expressions between parentheses are parsed just fine without this), but is strictly required
    /// when parsing an expression between e.g. double pipes (`||`), because this token cannot necessarily be distinguished from the "or" operator.
    #[allow(clippy::type_complexity)]
    fn parse_expression(&mut self, min_precedence: u8, return_early_if: Option<&Box<dyn Fn(&Token) -> bool>>, env: &mut Env) -> Result<Expression, String> {
        // First, determine the LHS of the next operation to execute.
        // This is either an identifier, a number or a further expression between parentheses.
        let mut lhs = match self.next()? {
            Token::Minus => Expression::UnaryOperation(UnaryOperation::Neg, Box::new(self.parse_expression(6, None, env)?)),
            Token::ExclamationMark // An exclamation mark before an expected expression signifies a `not` operator
                => Expression::UnaryOperation(UnaryOperation::Not, Box::new(self.parse_expression(3, None, env)?)),
            Token::Identifier(id) if id == "D" || id == "D_" => { // Total derivative
                // Expected tokens: ("D" | "D_{...}") <FunctionExpr> (<point>) [<direction>].
                // For a list of all accepted syntaxes, see the documentation of the program's syntax.
                let mut argnames = Vec::<String>::new();
                if id == "D_" { // Then, parse argnames now. Otherwise, we need knowledge of `function_expr` for this.
                    for inner_expr in self.expect_brace_expr_with_commas(env)?.into_iter() {
                        if let Expression::Identifier(s) = inner_expr {
                            argnames.push(s);
                        } else {
                            return Err(format!("Expected identifier, got {}.", inner_expr));
                        }
                    }
                }
                let mut function_expr = self.parse_expression(8, None, env)?;
                // At this point, the next token can either be a parenthesis or a bracket.
                let point = match (self.peek()?, &mut function_expr) {
                    // This case means the point is yet to parse.
                    (Token::LParenthesis, _) => {
                        if id == "D" { // Then, only parse arguments now (since we need to know `function_expr` for this)
                            let mut identifiers = HashSet::<String>::new();
                            function_expr.list_unknown_identifiers(&VarStack::Empty, env, &mut identifiers);
                            argnames = identifiers.into_iter().collect::<Vec<String>>();
                            argnames.sort_unstable();
                        }
                        self.next()?;
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
                    _ => return Err("Missing point to differentiate at in total derivative expression.".to_string())

                };
                self.expect_token(Token::LBracket, None)?;
                let direction = self.parse_comma_expression(&Token::RBracket, env)?;
                Expression::DirectionalDerivative(argnames, Box::new(function_expr), point, direction)
            }
            Token::Identifier(id) if let Some(op) = FoldedOperation::from_string(&id) => { // Folded operation
                // Expected tokens:
                // <op_name>, Ident("_"), LBrace,
                //     Ident, Eq, Comparison(Eq, None), Vec<Token>,
                // RBrace, Circumflex, LBrace | None,
                //     Vec<Token> | (Identifier | Number),
                // RBrace | None,
                // Vec<Token>
                let mut subscript = self.expect_brace_expr_with_commas(env)?; // Should be ["i = ...", *conditions]
                let (index_var_name, index_var_init, conditions) = match subscript.remove(0) {
                    Expression::BinaryOperation(lhs, BinaryOperation::Comp(Comparison::Eq, None), rhs) => match *lhs {
                        Expression::Identifier(s) => (s, *rhs, subscript), // Notice "i = ..." was removed from `subscript` already
                        other => return Err(format!("Expected an identifier as LHS of `=`, got {:?}.", other))
                    }
                    other => return Err(format!("Expected an expression of the form `Identifier(...) = ...`, got {:?}.", other))
                };
                self.expect_token(Token::Circumflex, Some(" to specify end of range"))?;
                let superscript = self.expect_brace_expr(env)?;
                let inner = self.parse_expression(op.priority() + 1, None, env)?;
                Expression::FoldedOperation(op, index_var_name, Box::new(index_var_init), conditions, Box::new(superscript), Box::new(inner))
            }
            Token::Identifier(id) if id.starts_with("int_") => {
                let rest = id.strip_prefix("int_").unwrap(); // Safe because of `starts_with` call above
                let subscript = if !rest.is_empty() {
                    // Parse `rest` and expect to get a single expression out.
                    match Parser::from(tokenize(rest)?).parse(env)? {
                        v if v.len() == 1 => v.into_iter().next().unwrap(), // Safe to unwrap because of length check before
                        other => return Err(format!("Subscript after integral must be single expression (got {:?}).", other))
                    }
                } else {
                    self.expect_brace_expr(env)?
                };
                self.expect_token(Token::Circumflex, Some(" to specify end of range"))?;
                let superscript = self.expect_brace_expr(env)?;
                // Parse inner expression but stop immediately if an identifier of length >= 2 starting with `d` is encountered.
                let stopper: Box<dyn Fn(&Token) -> bool> = Box::new(|t: &Token| matches!(t, Token::Identifier(id) if id.starts_with('d') && id.len() >= 2));
                let inner = self.parse_expression(0, Some(&stopper), env)?;
                let int_var = match self.next()? {
                    Token::Identifier(id) if id.starts_with("d") && id.len() >= 2 => id.strip_prefix("d").unwrap().to_string(),
                    other => return Err(format!("Expected \"dv\" where \"v\" is some identifier; got {:?}.", other))
                };
                Expression::Integral(Box::new(inner), Box::new(subscript), Box::new(superscript), int_var)
            }
            Token::Identifier(x) => {
                // We have to check whether this will be a function call: we judge this to be the case iff the next token is an LParenthesis and
                // either the identifier `x` is contained in `functions` or we are on the LHS of an assignment operator. There is no efficient
                // way to know yet whether there will be an assignment operator on the same precedence level as we currently are. Therefore,
                // this case will be handled afterwards by 'eval'. So, we only have to check the case:
                match self.peek()? {
                    Token::LParenthesis if env.functions.contains_key(&x) || x.starts_with("___diff_num_") => {
                        self.next()?;
                        Expression::Function(x, self.parse_comma_expression(&Token::RParenthesis, env)?)
                    }
                    _ => Expression::Identifier(x)
                }
            }
            Token::Number(x) => Expression::Number(x),
            Token::LParenthesis => {
                // Parse expression between parentheses recursively. It could just be a single expression of multiple entries separated by commas.
                let mut entries = self.parse_comma_expression(&Token::RParenthesis, env)?;
                match entries.len() {
                    0 => Expression::Vector(Vec::new()),
                    1 => entries.pop().unwrap(), // I decided to not box the elements rightaway since the case `entries.len() == 1` is more common.
                    _ => Expression::Tuple(entries)
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
                    match self.next()? {
                        Token::Comma => {},
                        Token::Semicolon | Token::Backslash => {
                            if n == 0 { // If n has not been set yet
                                n = current_n; // Set n
                            }
                            else if n != current_n {
                                return Err(format!("Got matrix row of wrong length (expected {n}, got {}.", current_n));
                            }
                            current_n = 0;
                            m += 1;
                        }
                        Token::RBracket => {
                            if n == 0 { // If n has not been set yet
                                n = entries.len(); // Set n
                            }
                            else if n != current_n {
                                return Err(format!("Got matrix row of wrong length (expected {n}, got {}.", current_n));
                            }
                            break;
                        },
                        other => return Err(format!("Expected ')', found {:?}", other))
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
                self.expect_token(Token::Pipe, Some(" as closer"))?;
                Expression::UnaryOperation(UnaryOperation::Abs, Box::new(inner))
            }
            Token::DoublePipe => { // In this context: opener of a norm
                let f: Box<dyn Fn(&Token) -> bool> = Box::new(|t: &Token| matches!(t, Token::DoublePipe));
                let inner = self.parse_expression(0, Some(&f), env)?;
                self.expect_token(Token::DoublePipe, Some(" as closer"))?;
                match self.peek()? {
                    Token::Identifier(ident) if ident.starts_with('_') => {
                        let norm_type = if ident == "_" {
                            self.next()?;
                            match self.next()? {
                                Token::Identifier(a) => Expression::Identifier(a),
                                Token::Number(a) => Expression::Number(a),
                                Token::LBrace => {
                                    let res = self.parse_expression(0, None, env)?;
                                    self.expect_token(Token::RBrace, None)?;
                                    res
                                }
                                other => {return Err(format!("Expected norm type after '||_', found {:?}", other));}
                            }
                        } else {
                            let cloned_ident = ident.clone();
                            let mut chars = cloned_ident.chars(); chars.next();
                            self.next()?;
                            Expression::Identifier(chars.collect::<String>())
                        };
                        Expression::UnaryOperation(UnaryOperation::Norm(Some(Box::new(norm_type))), Box::new(inner))
                    }
                    _ => Expression::UnaryOperation(UnaryOperation::Norm(None), Box::new(inner))
                }
            }
            Token::If => {
                let condition = self.parse_expression(0, None, env)?; // Will return wenn LBrace is encountered.
                self.expect_token(Token::LBrace, Some(" after condition"))?;
                let iftrue = self.parse_expression(0, None, env)?;
                self.expect_token(Token::RBrace, Some(" before `iftrue` expression"))?;
                self.expect_token(Token::Else, None)?;
                self.expect_token(Token::LBrace, Some(" after `else`"))?;
                let iffalse = self.parse_expression(0, None, env)?;
                self.expect_token(Token::RBrace, Some(" after `iffalse` expression"))?;
                Expression::IfElse(Box::new(condition), Box::new(iftrue), Box::new(iffalse))
            }
            other => return Err(format!("Unexpected token where expression expected: {:?}", other))
        };

        // Then, parse the RHS recursively.
        // This is the first time we check `return_early_if`. The function could typically stop here if
        // `return_early_if` checks for the presence of a closing `||`, which would have to be in the place
        // where an operator is expected.
        if let Some(f) = return_early_if && f(self.peek()?) {
            // Importantly, do not consume the encountered token so the caller can check it.
            return Ok(lhs);
        }
        loop {
            let (mut op, prec, consume) = match self.peek()? {
                Token::Plus => (BinaryOperation::Add, 5, true),
                Token::Minus => (BinaryOperation::Sub, 5, true),
                Token::Asterisk => (BinaryOperation::Mul, 6, true),
                Token::LParenthesis => (BinaryOperation::Mul, 6, false), // Expressions such as "2(x+1)" are parsed as "2*(x+1)".
                Token::LBracket => (BinaryOperation::Mul, 6, false), // Same if a vector/matrix follows.
                Token::Identifier(_) => (BinaryOperation::Mul, 6, false), // Expressions such as "(x+1)y" are parsed as "(x+1)*y".
                Token::Slash => (BinaryOperation::Div, 6, true),
                Token::DoubleSlash => (BinaryOperation::Quo, 6, true),
                Token::Percent => (BinaryOperation::Rem, 6, true),
                Token::Circumflex => (BinaryOperation::Pow(true), 7, true),
                Token::Assign => (BinaryOperation::Add, 0, true), // We don't need any operation here, so 'Add' is just a placeholder to simplify notation
                // Importantly, we only fetch the comparison's optional parameter later, when we actually consume the operator (avoids cloning).
                Token::Comparison(c, _) => (BinaryOperation::Comp(*c, None), 4, true),
                Token::DoublePipe => (BinaryOperation::Or, 1, true),
                Token::DoubleAmpersand => (BinaryOperation::And, 2, true),

                Token::ExclamationMark => { // An exclamation mark after an expression signifies a factorial operation
                    self.next()?; // Consume exclamation mark
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
            if consume { // Implicit operators, e.g. left parentheses interpreted as "*(", do not lead to the consumption of the next token.
                if let Token::Comparison(c, param) = self.next()? { // As mentioned above, fetch the missing comparison parameter (if there is one)
                    let parsed_param = if let Some(p) = param {
                        // `unwrap` in the following line is acceptable since the `map` at the beginning ensures that `param` is `Some`
                        Some(Box::new(Parser::from(p).parse(env)?.into_iter().next().unwrap()))
                    } else {None};
                    op = BinaryOperation::Comp(c, parsed_param)
                }
            }

            // The RHS can only contain operators of strictly larger precedence, so we parse it with parameter 'prec+1'.
            let rhs = self.parse_expression(prec + 1, return_early_if, env)?;

            if let BinaryOperation::Pow(_) = op {
                // Special case: the `^` operator is traditionally right-associative and not left-associative, i.e. 2^3^2 = 2^(3^2) and not (2^3)^2.
                // Naively, we would check if `lhs = left ^ middle` and then return `left ^ (middle ^ rhs)`. However, `middle` might be an expression of
                // type `x ^(true) y` too; in that case, we'd have to pass `^rhs` further down until we reach an expression that it not of this form.
                let mut last_exponent_in_chain = &mut lhs;
                // The following line is also the only spot in the entire codebase where the bool inside `BinaryOperation::Pow` matters.
                while let Expression::BinaryOperation(_, BinaryOperation::Pow(true), next_exponent) = last_exponent_in_chain {
                    last_exponent_in_chain = next_exponent;
                }
                // Now, we can replace `last_exponent_in_chain` by `last_exponent_in_chain ^ rhs`.
                // Because of ownership reasons, we first need to move `last_exponent_in_chain` out of `lhs`, leaving a placeholder behind,
                // and then put back in the modified version.
                let old = std::mem::replace(last_exponent_in_chain, Expression::None);
                *last_exponent_in_chain = Expression::BinaryOperation(Box::new(old), BinaryOperation::Pow(true), Box::new(rhs));
            } else {
                // Otherwise, check a few more special cases or just combine `lhs`, `op` and `rhs` as expected.
                lhs = match (lhs, &op, &rhs) {
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
                };
            }

            // To handle the right-associative operator `^`, in case `lhs` now is of the form `a ^ b`, we need to store whether
            // `lhs` is followed by a parenthesis (because this breaks right-associativity).
            if let (Expression::BinaryOperation(_, _op @ BinaryOperation::Pow(_), _), Ok(Token::RParenthesis)) = (&mut lhs, self.peek()) {
                *_op = BinaryOperation::Pow(false);
            }

            // This is the second place where `return_early_if` is checked. It was passed down when parsing the
            // RHS recursively, so maybe, this recursive call was interrupted by `return_early_if`. However,
            // this only stops the recursive call in `let rhs = ...`, not this current function call, which
            // we therefore have to do now. An example where we'd land in this case is `int_0^1 x-1 dx`.
            // For the integrand `x^2` instead, we still return because of the following lines of code,
            // but it wans't necessary to pass `return_early_if` down while parsing the RHS.
            if let Some(f) = return_early_if && f(self.peek()?) {
                return Ok(lhs);
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
    /// <tr> <td>+, - (binary)<td/> <td>5<td/> </tr>
    /// <tr> <td>*, /, //, %, - (unary)<td/> <td>6<td/> </tr>
    /// <tr> <td>^<td/> <td>7<td/> </tr>
    /// <tr> <td>d/dx, D<td/> <td>8<td/> </tr>
    /// </table>
    pub fn parse(&mut self, env: &mut Env) -> Result<Vec<Expression>, String> {
        let mut exprs = Vec::<Expression>::new();
        loop {
            let expr = self.parse_expression(0, None, env)?;
            exprs.push(expr);
            match self.next()? {
                Token::EOF => {return Ok(exprs);},
                Token::Semicolon => {continue;}
                other => return Err(format!("Unexpected trailing token: {:?}", other))
            }
        }
    }
}