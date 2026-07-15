# Wolfgang Alpha

A web and CLI symbolic/numeric calculator.

## Disclaimer
The UI was made using GPT 5.6 and Claube Fable 5 because I am terrible at graphic design.

## Usage

### Web / Desktop
```sh
dx serve          # web (default)
dx serve --platform desktop
```

### CLI REPL
```sh
cargo run --bin cli
```

The CLI REPL reads expressions line by line. Type `exit` or `quit` to quit.

---

## Syntax

The basic syntax is the natural one with usual operator precedence. A few special features are the following.

### Matrices and vectors
- Matrices can be initialized by typing `[1, 2, 3 \ 4, 5, 6 \ 7, 8, 9]` where the rows will be `[1,2,3]`, `[4,5,6]` and `[7,8,9]` respectively.
  The backslash can be used interchangeably with a semicolon `;`, even within the same matrix.
- Unlike for tuples (see below), the environment is not
  captured before evaluation, meaning that e.g. `x := 1; [x := 2, x + 1]` returns `[2, 3]` and not `[2, 2]`. The reason for this is performance:
  matrices aren't made to contain definitions as entries (unlike tuples in some scenarios) and capturing the environment is costly. In this regard,
  vectors behave like matrices.
- Vectors can be initialized by typing either `[1; 2; 3]` or `[1 \ 2 \ 3]` (as one would initialize a matrix with only one column).
- A range of standard functions for matrices and vectors are pre-defined, such as `det`, `tr` and `adj`.
  For precise lists and explanations on the implementations, see `defaults.rs`.
- Many matrix functions (e.g. matrix multiplication, transposition) are written with view to efficiency for large matrices
  (using optimization strategies like tiling for better cache locality and parallelization using the `raylib` crate),
  even though in this specific application, most matrices are likely small.

### Folded operations
- `sum_{i=a}^b ...` acts as one would expect. `i` has to be an identifier, `a` must be evaluable to an integer and `b` to a float (`a`, `b` need not to be constants).
  The type of the object inside the sum is inferred. If `a > b` initially, then `0` is returned (in the appropriate type).
  The same holds for `prod`.
- An arbitrary amount of conditions can be added to a sum as follows: `sum_{i=a, i != 5, ...}^b ...`.
  All values of `i` that do not satisfy all of the given conditions will be skipped.

### Custom definitions
- Definition of constants: `identifier := expr`, where `expr` can be any expression that can be evaluated at the time of the definition.<br/>
  This returns the evaluation of `expr`, so one can write e.g. `(x := 2) + 1` to obtain `3` as output and define `x` simultaneously.<br/>
  If `identifier` is already a defined constant, this will re-define it and permanently suppress the old value.
- Tuple assignment is supported: write e.g. `(x, y) := rhs` where `rhs` can be evaluated to a tuple of the same size. Function assignment is not allowed in this way.
  The environment is captured before evaluation such that the entire tuple is evaluated based on the same environment.
- Definition of functions: `f(x, y) := 2x + y`. If e.g. `x` already exists as a constant/function, this will be ignored for the sake of the function's definition.
  The `x` on the RHS of the definition will always be the `x` passed as argument, not the constant.<br/>
  If one wants to include a constant from the current environment, simply type `f(y) := 2x + y` where `x` is a pre-defined constant. Note that the
  current value of `x` will be captured at the time of the definition; if you change `x` later on, `f` will still use its old value.

### Built-in constants and functions
- The built-in constants currently are `pi` and `e`.
- The built-in functions are:
  - `1` (indicator function)
  - `exp`, `ln` and `log(x, base)`
  - `sign` (with the convention `sign(0) = 1`)
  - `sqrt`
  - `cos`, `sin`, `tan` as well as hyperbolic versions (e.g. `cosh`) and all inverses (e.g. `acos`, `acosh`)
  - Matrix functions `eig`, `adj`, `det`, `tr`
- There are some helper functions prefixed with `___helper_` to increase efficiency. These don't have built-in derivatives.

### Comparisons
- Test if two values are equal: `expr = other_expr` where both expressions must be evaluable to an `Object`. Very small errors are tolerated.
- The same works for `<`, `<=`, `>` and `>=`. The strict comparison signs do _not_ tolerate small errors.
  As for equality, two vectors/matrices of the same size satisfy a comparison iff all of their components satisfy it.
- Running `lhs = rhs` where at least one of `lhs`, `rhs` contains unknown identifiers (and is thus considered a function), both sides are evaluated at every point
  in `linspace(0, 1, n)`, `linspace(1, 100, n)` and `(101, ..., 100 + n)` as well as their negative counterparts. If they differ at some point, `0` is immediately
  returned. If they match at all points, `1` is returned. Per default, `n = lang::evaluator::DEFAULT_TESTEQ_REPETITIONS`. One can specify `n` by using `lhs ={e} rhs`
  where `e` can be any expression evaluable to a float (will then be rounded to the nearest integer). 
  The same works for `<`, `<=`, `>` and `>=`.

### Differentiation
- Partially differentiate: `d/dx (x^3 + 2x + 1)` returns `3x^2 + 2` as expression. The parentheses are not needed when differentiating e.g. a monome.<br/>
  The output can be stored in a function: `f(x) := d/dx ...`.<br/>
  Differentiating a function with a matrix/vector as output will differentiate component-wise and return the corresponding matrix/vector-valued function.<br/>
  If the differentiated function `f(x)` outputs a vector/matrix, the output will be the function `p \mapsto D_x f(p)[1]`, that is, the direction to differentiate in will be set to 1.0 by default.
  This means the syntax is still accepted although not recommended.
- Directionally differentiate: multiple syntaxes:
    - `D_x <expr1> (expr2)[expr3]` leads to `point := {x: expr2}` and `direction := {x: expr3}`.
    - `D_{x, y} <expr1> (expr2x, expr2y)[expr3x, expr3y]` leads to `point := {x: expr2x, y expr2y}` and analogously for `direction`. Analogously for any higher number of variables.
    - `D f(4)[2]`: free variables are set to be the argnames of `f` (these will be the keys of the hashmap, cf. implementation).
    - `D <expr> (expr_1, ..., expr_n)[expr'_1, ..., expr'_m]`: collect all unknown identifiers within `expr` into a vector in ascending alphabetic order `x_1, ..., x_l`.
      If `l=m=n`, infer that these should be the keys of the hashmaps (cf. implementation). Otherwise, return `Err`.

### Special syntaxes and remarks
- Tuples can be initialized by typing `(1, 2, 3)`. Tuples are polymorphic but only support very few operations. They are primarily intended to support multiple
  simultaneous assignments (see below); generally, the use of vectors is preferred.
- `debug` prints the entire current environment (constants + functions). In the web UI this goes to the browser console (`F12`); in the CLI it prints to stdout.
- Notice that the token `!` acts as both the `not` operator and the factorial operator. In context, one can always differentiate between the two, with one minor downside:
  the syntax `x * (!y)` cannot be shortened to `x !y` (since these spaces disappear while tokenizing, one would not be able to differentiate this with `(x!) * y`).
