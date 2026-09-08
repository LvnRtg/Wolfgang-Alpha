//! Responsible for parsing tokenized inputs into expressions, which includes respecting operator precedence.
//! 
use std::collections::HashSet;
use std::iter::Peekable;
use std::vec::IntoIter;

use crate::expr_unary_op;
use crate::lang::lexer::{Keyword, Token, tokenize};
use crate::math::operations::{BinaryOperation, Comparison, UnaryOperation, FoldedOperation};
use crate::math::{Env, Expression, FunctionRepr, utils, VarStack};
use crate::status::Status;

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
    fn parse_comma_expression(&mut self, closer: &Token, env: &mut Env) -> Result<Status<Vec<Expression>>, String> {
        let mut exprs = Vec::<Expression>::new();
        if let Ok(t) = self.peek() && t == closer {
            _ = self.next();
            return Ok(Status::ok(exprs));
        }
        let mut warnings = Vec::new();
        loop {
            exprs.push(self.parse_expression(0, None, env)?.unpack_into(&mut warnings));
            match self.next()? {
                Token::Comma => {},
                some if &some == closer => {break;},
                other => {return Err(format!("Expected '{:?}', found {:?}.", closer, other));}
            }
        }
        Ok(Status {
            value: exprs,
            warnings
        })
    }

    /// Expects either `LBrace, ..., RBrace` (then parses `...` and returns the result) or `Identifier(...) | Number(...)`
    /// (then returns `...` directly). All other syntaxes return `Err`.
    /// 
    /// For example, you'd call this after encountering `sum_`.
    fn expect_brace_expr(&mut self, env: &mut Env) -> Result<Status<Expression>, String> {
        match self.next()? {
            Token::Identifier(x) => Ok(Status::ok(Expression::Identifier(x))),
            Token::Number(x) => Ok(Status::ok(Expression::Number(x))),
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
    fn expect_brace_expr_with_commas(&mut self, env: &mut Env) -> Result<Status<Vec<Expression>>, String> {
        match self.next()? {
            Token::Identifier(x) => Ok(Status::ok(vec![Expression::Identifier(x)])),
            Token::Number(x) => Ok(Status::ok(vec![Expression::Number(x)])),
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
    fn parse_expression(&mut self, min_precedence: u8, return_early_if: Option<&Box<dyn Fn(&Token) -> bool>>, env: &mut Env) -> Result<Status<Expression>, String> {
        // First, determine the LHS of the next operation to execute.
        // This is either an identifier, a number or a further expression between parentheses.
        let mut warnings = Vec::new();
        let mut lhs = match self.next()? {
            Token::Minus => expr_unary_op!(Neg, self.parse_expression(6, None, env)?.unpack_into(&mut warnings)),
            Token::ExclamationMark // An exclamation mark before an expected expression signifies a `not` operator
                => expr_unary_op!(Not, self.parse_expression(6, None, env)?.unpack_into(&mut warnings)),
            Token::Identifier(id) if id == "D" || id == "D_" => { // Total derivative
                // Expected tokens: ("D" | "D_{...}") <FunctionExpr> (<point>) [<direction>].
                // For a list of all accepted syntaxes, see the documentation of the program's syntax.
                let mut argnames = Vec::<String>::new();
                if id == "D_" { // Then, parse argnames now. Otherwise, we need knowledge of `function_expr` for this.
                    for inner_expr in self.expect_brace_expr_with_commas(env)?.unpack_into(&mut warnings).into_iter() {
                        if let Expression::Identifier(s) = inner_expr {
                            argnames.push(s);
                        } else {
                            return Err(format!("Expected identifier, got {}.", inner_expr));
                        }
                    }
                }
                let mut function_expr = self.parse_expression(8, None, env)?.unpack_into(&mut warnings);
                // At this point, the next token can either be a parenthesis or a bracket.
                let point = match (self.peek()?, &mut function_expr) {
                    // This case means the point is yet to parse.
                    (Token::LParenthesis, _) => {
                        if id == "D" { // Then, only parse arguments now (since we need to know `function_expr` for this)
                            let mut identifiers = HashSet::<String>::new();
                            function_expr.get_unknown_identifiers(&VarStack::Empty, env, &mut identifiers);
                            argnames = identifiers.into_iter().collect::<Vec<String>>();
                            argnames.sort_unstable();
                        }
                        self.next()?;
                        self.parse_comma_expression(&Token::RParenthesis, env)?.unpack_into(&mut warnings)
                    }
                    // The following case is only valid if `function_expr` is a `Expression::Function(f, x)`, in which case `x` is the actual point
                    // and the true arguments given to `f` should be its argnames in order (if f has direct representation, use x_1, ..., x_n
                    // instead where n is the length of `x`). The syntax `D f` should be used here (and not e.g. `D_x f`);
                    // if `D_x f` was used, it is treated as `D` (formally, the previous value of `argnames` (here, `["x"]`) is overwritten).
                    (Token::LBracket, Expression::Function(name, args)) if env.functions.contains_key(name) => {
                        argnames = match env.functions.get(name).unwrap() {
                            FunctionRepr::ByExpression(argnames, _) => argnames.clone(),
                            FunctionRepr::Direct(..) => (0..args.len()).map(|i| format!("x_{}", i)).collect()
                        };
                        std::mem::replace(args, argnames.iter().map(|x| Expression::Identifier(x.clone())).collect())
                    }
                    _ => return Err("Missing point to differentiate at in total derivative expression.".to_string())
                };
                self.expect_token(Token::LBracket, None)?;
                let direction = self.parse_comma_expression(&Token::RBracket, env)?.unpack_into(&mut warnings);
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
                let mut subscript = self.expect_brace_expr_with_commas(env)?.unpack_into(&mut warnings); // Should be ["i = ...", *conditions]
                let (index_var_name, index_var_init, conditions) = match subscript.remove(0) {
                    Expression::BinaryOperation(lhs, BinaryOperation::Comp(Comparison::Eq, None), rhs) => match *lhs {
                        Expression::Identifier(s) => (s, *rhs, subscript), // Notice "i = ..." was removed from `subscript` already
                        other => return Err(format!("Expected an identifier as LHS of `=`, got {:?}.", other))
                    }
                    other => return Err(format!("Expected an expression of the form `Identifier(...) = ...`, got {:?}.", other))
                };
                self.expect_token(Token::Circumflex, Some(" to specify end of range"))?;
                let superscript = self.expect_brace_expr(env)?.unpack_into(&mut warnings);
                let inner = self.parse_expression(op.priority() + 1, None, env)?.unpack_into(&mut warnings);
                Expression::FoldedOperation(op, index_var_name, Box::new(index_var_init), conditions, Box::new(superscript), Box::new(inner))
            }
            Token::Identifier(id) if id.starts_with("int_") => {
                let rest = id.strip_prefix("int_").unwrap(); // Safe because of `starts_with` call above
                let subscript = if !rest.is_empty() {
                    // Parse `rest` and expect to get a single expression out.
                    let mut rest_parser = Parser::from(tokenize(rest)?);
                    let rest = rest_parser.parse_next(env).ok_or("No expression to parse for subscript of integral.".to_string())??;
                    if rest_parser.parse_next(env).is_some() {
                        return Err("Multiple expressions encountered while parsing subscript of integral.".to_string())
                    }
                    rest.unpack_into(&mut warnings)
                } else {
                    self.expect_brace_expr(env)?.unpack_into(&mut warnings)
                };
                self.expect_token(Token::Circumflex, Some(" to specify end of range"))?;
                let superscript = self.expect_brace_expr(env)?.unpack_into(&mut warnings);
                // Parse inner expression but stop immediately if an identifier of length > 1 starting with `d` is encountered.
                let stopper: Box<dyn Fn(&Token) -> bool> = Box::new(|t: &Token| matches!(t, Token::Identifier(id) if id.starts_with('d') && id.len() > 1));
                let inner = self.parse_expression(0, Some(&stopper), env)?.unpack_into(&mut warnings);
                let int_var = match self.next()? {
                    Token::Identifier(id) if id.starts_with("d") && id.len() > 1 => id.strip_prefix("d").unwrap().to_string(),
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
                    Token::LParenthesis if env.functions.contains_key(&x) || x == "del" => {
                        let _ = self.next();
                        Expression::Function(x, self.parse_comma_expression(&Token::RParenthesis, env)?.unpack_into(&mut warnings))
                    }
                    _ => Expression::Identifier(x)
                }
            }
            Token::Number(x) => {
                // The only case where this isn't simply an `Expression::Number` is when writing "1(...";
                // then, it should be parsed as `Expression::Function("1", ...)`.
                match self.peek()? {
                    Token::LParenthesis => {
                        self.next()?;
                        Expression::Function("1".to_string(), self.parse_comma_expression(&Token::RParenthesis, env)?.unpack_into(&mut warnings))
                    }
                    _ => Expression::Number(x)
                }
            }
            Token::LParenthesis => {
                // Parse expression between parentheses recursively. It could just be a single expression of multiple entries separated by commas.
                // It could also be empty.
                let mut entries = self.parse_comma_expression(&Token::RParenthesis, env)?.unpack_into(&mut warnings);
                match entries.len() {
                    0 => Expression::Tuple(Vec::new()),
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
                    entries.push(self.parse_expression(0, None, env)?.unpack_into(&mut warnings));
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
                let inner = self.parse_expression(0, None, env)?.unpack_into(&mut warnings);
                self.expect_token(Token::Pipe, Some(" as closer"))?;
                Expression::UnaryOperation(UnaryOperation::Abs, Box::new(inner))
            }
            Token::DoublePipe => { // In this context: opener of a norm
                let f: Box<dyn Fn(&Token) -> bool> = Box::new(|t: &Token| matches!(t, Token::DoublePipe));
                let inner = self.parse_expression(0, Some(&f), env)?.unpack_into(&mut warnings);
                self.expect_token(Token::DoublePipe, Some(" as closer"))?;
                match self.peek()? {
                    Token::Identifier(ident) if ident.starts_with('_') => {
                        let norm_type = if ident == "_" {
                            self.next()?;
                            match self.next()? {
                                Token::Identifier(a) => Expression::Identifier(a),
                                Token::Number(a) => Expression::Number(a),
                                Token::LBrace => {
                                    let res = self.parse_expression(0, None, env)?.unpack_into(&mut warnings);
                                    self.expect_token(Token::RBrace, None)?;
                                    res
                                }
                                other => {return Err(format!("Expected norm type after '||_', found {:?}", other));}
                            }
                        } else {
                            let cloned_ident = ident.clone();
                            self.next()?;
                            if let Ok(x) = cloned_ident[1..].parse::<f64>() {
                                Expression::Number(x)
                            } else {
                                Expression::Identifier(cloned_ident.chars().skip(1).collect::<String>())
                            }
                        };
                        Expression::UnaryOperation(UnaryOperation::Norm(Some(Box::new(norm_type))), Box::new(inner))
                    }
                    _ => Expression::UnaryOperation(UnaryOperation::Norm(None), Box::new(inner))
                }
            }
            Token::Keyword(Keyword::If) => {
                let condition = self.parse_expression(0, None, env)?.unpack_into(&mut warnings); // Will return wenn LBrace is encountered.
                self.expect_token(Token::LBrace, Some(" after condition"))?;
                let iftrue = self.parse_expression(0, None, env)?.unpack_into(&mut warnings);
                self.expect_token(Token::RBrace, Some(" before `iftrue` expression"))?;
                self.expect_token(Token::Keyword(Keyword::Else), None)?;
                self.expect_token(Token::LBrace, Some(" after `else`"))?;
                let iffalse = self.parse_expression(0, None, env)?.unpack_into(&mut warnings);
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
            return Ok(Status{value: lhs, warnings});
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
                | Token::Keyword(_) | Token::EOF
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
                        let mut param_parser = Parser::from(p);
                        let res = param_parser.parse_next(env).ok_or(format!("No expression to parse for parameter of comparison {c}."))??.unpack_into(&mut warnings);
                        if param_parser.parse_next(env).is_some() {
                            return Err(format!("Multiple expressions encountered while parsing parameter of comparison {c}."))
                        }
                        Some(Box::new(res))
                    } else {None};
                    op = BinaryOperation::Comp(c, parsed_param)
                }
            }

            // The RHS can only contain operators of strictly larger precedence, so we parse it with parameter 'prec+1'.
            let rhs = self.parse_expression(prec + 1, return_early_if, env)?.unpack_into(&mut warnings);

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
                lhs = match (lhs, op, rhs) {
                    // Assignment operator
                    (lhs, _, rhs) if prec == 0 => Expression::Assignment(Box::new(lhs), Box::new(rhs)),
                    // Partial derivative
                    (
                        numerator,
                        BinaryOperation::Div,
                        denominator
                    ) if let Some((numerator_exp, mut denominator_exps)) = check_partial_derivative_syntax(&numerator, &denominator) => {
                        // Check if the exponents in numerator and denominator match. If not, ignore the numerator but emit a warning.
                        match crate::lang::evaluator::compare_expressions(
                            &numerator_exp,
                            &crate::expr_binop_from_iter!(Add, Sum, denominator_exps.iter().map(|&(_, ref x)| x.clone())),
                            Comparison::Eq,
                            &None,
                            &crate::math::VarStack::Empty, // During parsing, no temporary extra vars are given
                            env
                        )
                        .and_then(|s| s.value.expect_bool()) {
                            Ok(true) => {}
                            Ok(false) => {
                                // There is an edge case here which we have to take care of.
                                // If the user types `d^2/dxdy`, the denominator is initially interpreted as `d(xdy)`. Only the exponent in the numerator tells us
                                // something is off: indeed, if it were `d/dxdy`, then `d(xdy)` would be the correct interpretation.
                                // So, since we just detected that there is a mismatch between numerator and denominator, we look at the denominator again
                                // to see if some identifiers contain the character 'd'.
                                // In doing this, we create an alternate interpretation of the denominator:
                                let mut alt_denominator_exps = Vec::<(String, Expression)>::new();
                                for (s, n) in denominator_exps.iter() {
                                    // `d(xdydz)^2` shall become `dx dy dz^2`.
                                    for substr in s.split('d') {
                                        alt_denominator_exps.push((substr.to_string(), Expression::Number(1.0)));
                                    }
                                    alt_denominator_exps.last_mut().unwrap().1 = n.clone();
                                }
                                // Check if now, the exponents match
                                if let Ok(true) = crate::lang::evaluator::compare_expressions(
                                    &numerator_exp,
                                    &crate::expr_binop_from_iter!(Add, Sum, alt_denominator_exps.iter().map(|&(_, ref x)| x.clone())),
                                    Comparison::Eq,
                                    &None,
                                    &crate::math::VarStack::Empty, // During parsing, no temporary extra vars are given
                                    env
                                )
                                .and_then(|s| s.value.expect_bool()) {
                                    // If so, use the alternate exponents instead
                                    denominator_exps = alt_denominator_exps;
                                } else {
                                    // Otherwise, keep the old interpretation
                                    warnings.push("Exponents in numerator and denominator of partial derivative operator do not match. Ignoring numerator.".to_string())
                                }
                            }
                            Err(e) => warnings.push(format!("Error while comparing exponents in numerator and denominator of partial derivative operator.\nTraceback: {e}"))
                        }
                        Expression::PartialDerivative(
                            denominator_exps.into_iter().map(
                                |(s, e)| match e {
                                    Expression::Number(f) => {
                                        let i = f.round();
                                        if utils::approx_eq(f, i) && i >= 0.0 {
                                            Ok((s, crate::math::expressions::Repeat::Number(i as usize)))
                                        } else {
                                            Err(format!("Exponents in denominator must be integers; got {f}."))
                                        }
                                    }
                                    other => Ok((s, crate::math::expressions::Repeat::Expression(other)))
                                }
                            ).collect::<Result<Vec<_>, _>>()?,
                            Box::new(self.parse_expression(8, None, env)?.unpack_into(&mut warnings))
                        )
                    }
                    // Default
                    (lhs, op, rhs) => crate::expr_binop_from_enum!(lhs, op, rhs)
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
                return Ok(Status{value: lhs, warnings});
            }
        }

        Ok(Status{value: lhs, warnings})
    }

    /// Parse the given vector of tokens recursively while consuming it until the end of the expression is met
    /// (marked by a semicolon in the right context or the EOF). Returns `None` once the EOF is reached.
    /// If an error occurs while parsing, return `Some(Err(...))`.
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
    pub fn parse_next(&mut self, env: &mut Env) -> Option<Result<Status<Expression>, String>> {
        // Note on design choice: works as an iterator instead of a vector because in some cases,
        // the parsing depends on the environment (e.g. `x(y+z)` is parsed differently depending on whether
        // `x` is a function or constant). Therefore, in some specific cases, it is important to evaluate the
        // expressions one by one, modifying the environment step by step, e.g. for `f(x) := x^2; g(x) := f(2x)`.
        // Hence, a part should only be parsed once all previous parts have been parsed _and_ evaluated.
        self.tokens.peek()?; // Return `None` if there are no tokens left in `self.tokens`
        let status = match self.parse_expression(0, None, env) {
            Ok(s) => s,
            Err(e) => return Some(Err(e))
        };
        match self.next() {
            Ok(Token::EOF) | Ok(Token::Semicolon) => Some(Ok(status)),
            Ok(other) => Some(Err(format!("Unexpected trailing token: {:?}", other))),
            Err(e) => Some(Err(e))
        }
    }
}


/// Given a fraction `numerator / denominator`, returns `Some(numerator_exp, denominator_exps)` if the fraction satisfies
/// the syntax of a (possibly high-order) partial derivative, that is, `numerator = d^(numerator_exp)` and `denominator`
/// is a chain of terms of the form `d(var)^(exp)`.
/// 
/// Then, `denominator_exps` is the collection of all tuples `(var, exp)` for which `d(var)^(exp)` appears in the denominator's
/// chain, parsed from **right to left** (because this is the order the partial derivatives are evaluated in).
fn check_partial_derivative_syntax(
    numerator: &Expression,
    mut denominator: &Expression
) -> Option<(Expression, Vec<(String, Expression)>)> {
    // The numerator must be of the form `dx^n`.
    let numerator_exp = match numerator {
        Expression::Identifier(s) if s == "d" => Expression::Number(1.0),
        Expression::BinaryOperation(l, BinaryOperation::Pow(_), exp)
        if let Expression::Identifier(s) = &**l && s == "d" => {
            *exp.clone() // TODO check if this works without cloning. Idea: return only a pointer and move out of the pointer if the guard is satisfied
        }
        _ => return None
    };
    let mut v = Vec::new();
    /* After parsing, the denominator could look like this:
    `(((dx)^n * d) * var^2) * (dy)`
    Notice:
    - Multiplications are left-associative
    - Inputs like `dx` will be parsed as one single identifier
    - Inputs like `d(var)` will be parsed as `d * var`
    - In cases like `d(var)^2`, `d` and `var^2` can effectively be separated by a parenthesis.
    */
    loop {
        match denominator {
            Expression::Identifier(s) if s.starts_with('d') => {
                v.push((s[1..].into(), Expression::Number(1.0)));
                break; // There can be no further nesting
            }
            Expression::BinaryOperation(l, BinaryOperation::Pow(_), r) => {
                if let Expression::Identifier(s) = &**l && s.starts_with('d') {
                    v.push((s[1..].into(), *r.clone()));
                    break; // There can be no further nesting
                } else {
                    return None; // Expression cannot be a derivative
                }
            }
            Expression::BinaryOperation(l, BinaryOperation::Mul, r) => {
                // The first crucial information is whether `l` is only `Identifier("d")`, `... * "d"` or something else.
                match &**l {
                    Expression::Identifier(ls) if ls == "d" => {
                        // In this case, `r` can be `ident` of `ident ^ ...`
                        match &**r {
                            Expression::Identifier(s) => {
                                v.push((s.clone(), Expression::Number(1.0)));
                            }
                            Expression::BinaryOperation(_l, BinaryOperation::Pow(_), _r) => {
                                if let Expression::Identifier(s) = &**_l {
                                    v.push((s.clone(), *_r.clone()));
                                } else {
                                    return None; // Expression cannot be a derivative. We break after `match *r` anyway.
                                }
                            }
                            _ => {return None;}
                        }
                        // and there cannot be any further nesting
                        break;
                    }
                    Expression::BinaryOperation(_l, BinaryOperation::Mul, _r)
                    if let Expression::Identifier(_s) = &**_r && _s == "d" => {
                        // Then, `r` should be parsed as if `l` was just `Identifier("d")`.
                        match &**r {
                            Expression::Identifier(s) => {
                                v.push((s.clone(), Expression::Number(1.0)));
                            }
                            Expression::BinaryOperation(_l, BinaryOperation::Pow(_), _r) => {
                                if let Expression::Identifier(s) = &**_l {
                                    v.push((s.clone(), *_r.clone()));
                                } else {
                                    return None; // Expression cannot be a derivative. We break after `match *r` anyway.
                                }
                            }
                            _ => {return None;}
                        }
                        // However, this time, there can be further nesting.
                        denominator = &**_l;
                    }
                    other_l => {
                        // Other valid `l` would be e.g. power operations or multiplications such as `d * var^2`.
                        // Then, we ignore `l` completely for the moment (we'll parse it in the next loop iteration).
                        // For the moment, `r` has to be an expression like `dx^n` on its own, it can't rely on `"d"` coming from `l`.
                        // Note: by left-associativity of the multiplication, `r` can't be a multiplication anymore.
                        // The user _could_ circumvent this by typing `d^3/(dx * (dy * dz))`, but what maniac would do this?!
                        match &**r {
                            Expression::Identifier(s) if s.starts_with('d') => {
                                v.push((s[1..].into(), Expression::Number(1.0)));
                            }
                            Expression::BinaryOperation(l, BinaryOperation::Pow(_), r) => {
                                if let Expression::Identifier(s) = &**l && s.starts_with('d') {
                                    v.push((s[1..].into(), *r.clone()));
                                } else {
                                    return None;
                                }
                            }
                            _ => {
                                return None;
                            }
                        }
                        denominator = other_l; // Next thing to parse: `l`
                    }
                }
            }
            _ => {
                // If `denominator` is anything else, it doesn't correspond to the syntax of a partial derivative.
                return None;
            }
        }
    }
    Some((numerator_exp, v))
}