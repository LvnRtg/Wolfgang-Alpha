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

It has less front-end features for the user's convenience than the web version, but the back-end is identical.

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
- The operation `not` (`!`) is performed component-wise. Operations `and` and `or` are currently not implemented for matrices/vectors.

### Tuples
- Tuples can be initialized by typing `(1, 2, 3)`.
- Tuples are polymorphic but only support very few operations. They are primarily intended to support multiple simultaneous assignments (see [custom definitions](#custom-definitions)); generally, the use of vectors is preferred.
- The size of tuples isn't taken into account during type checks. This is used to our advantage for functions returning an unknown amount of values, e.g. the function `eig` (that computes the eigenvalues of a matrix).

### Folded operations
- `sum_{i=a}^b ...` acts as one would expect. `i` has to be an identifier, `a` must be evaluable to an integer and `b` to a float (`a`, `b` need not to be constants).
  The type of the object inside the sum is inferred. If `a > b` initially, then `0` is returned (in the appropriate type).
  The same holds for `prod`.
- An arbitrary amount of conditions can be added to a sum as follows: `sum_{i=a, i != 5, ...}^b ...`.
  All values of `i` that do not satisfy all of the given conditions will be skipped.
- Consider `sum_{i=a}^b f(i)`. Non-constant bounds `a`, `b` are supported in the following sense.
  - `a` is only evaluated once: it may _not_ depend on `i` and it is useless to have it change as `f` is evaluated.
  - `b` is allowed to depend on `i` or change as `f` is evaluated. Then, `b` will be freshly evaluated after every iteration until `i > b`, after which the iteration is terminated. If no such variability of `b` is detected, it is only evaluated once at the start of the iteration. The precise criterion is: the expression `b` is evaluated in every iteration iff `b` contains the identifier `i` (in any form) or `f(i)` contains an assignment `x := ...` for which `b` contains the identifier `x`.
  - Conditions will evidently depend on `i`, but they may also change as `f` is evaluated. Usually, conditions will be evaluated during the iteration anyway, so no further checks need to be done here.
  - In certain built-in double-sums, further optimizations are done to evaluate `a`, `b` and all conditions as few times as possible (for instance `compute_product_derivative_helper` [here](./src/math/operations/folded_operations.rs)).

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
- Delete a constant/function with `del(x)`. You can delete as many constants/functions as you want at once, e.g. `del(x, y, f)`.

### Built-in constants and functions
#### Constants
- `π` (alias `pi`)
- `e` (Euler's number)
- `i` (imaginary unit)
- `∞` (alias `inf`)
#### Functions operating on objects
- `1` (indicator function)
- `exp`, `ln` and `log(x, base)`
- `sign` (with the convention `sign(0) = 1`)
- `sqrt`
- `cos`, `sin`, `tan` as well as hyperbolic versions (e.g. `cosh`) and all inverses (e.g. `acos`, `acosh`)
- Matrix functions `eig`, `adj`, `det`, `tr`, `transpose`
#### Functions operating on expressions
- `del` to delete custom constants/functions, see [custom definitions](#custom-definitions).
- `show_components(expr)`: if `expr` can be evaluated to a vector, gives back a representation of `expr` in which the components of the result are clearly visible (i.e. a representation as `Expression::Vector`). For example, `show_components([x; 1] + [1; y])` outputs `[x + 1; 1 + y]`. Behaves analogously if `expr` can be evaluated to a matrix.

#### Helper functions
These functions are prefixed with `___helper_` and made increase efficiency.
- `___helper_prod_rule(x_val, x, i, a(x), b(x), f(i,x), f'(i,x))`, where only the first argument is an object (all other arguments are expressions), computes `sum_{i=a(x_val)}^{b(x_val)} f'(i, x_val) * prod_{j=a(x_val), j!=i}^{b(x_val)} f(j, x_val)`. You can add an arbitrary number of conditions as expressions after these arguments.
- `___helper_matrix_prod(j, k, a, b, i, f(i))`, where the first 4 arguments are objects and the last 2 are expressions, computes the `(j, k)`-entry of the matrix `prod_{i=a}^b f(i)` You can add an arbitrary number of conditions as expressions after these arguments.

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
- `debug` prints the entire current environment (constants + functions). In the web UI this goes to the browser console (`F12`); in the CLI it prints to stdout.
- Notice that the token `!` acts as both the `not` operator and the factorial operator. In context, one can always differentiate between the two, with one minor downside:
  the syntax `x * (!y)` cannot be shortened to `x !y` (since these spaces disappear while tokenizing, one would not be able to differentiate this with `(x!) * y`).
