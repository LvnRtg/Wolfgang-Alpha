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

The basic syntax is the natural one with usual operator precedence.

### General operations
- `+`, `-`, `*`, `/`, `^`
- Integer division: `x // y` returns the largest integer $n$ such that $n \cdot y \leq x$.
- Modulo (`%`), always returns a non-negative number.
- And (`&&`), Or (`||`), Not (`!`)
- Factorial (`!`), Absolute Value (`|...|`)

#### Comparisons
- If `lhs` and `rhs` are both evaluable to objects of the same type, `lhs = rhs` returns 1 if they are equal and 0 otherwise. Very small errors are tolerated.
- The same works for `<`, `<=`, `>` and `>=`. The strict comparison signs do _not_ tolerate small errors.
  As for equality, two vectors/matrices of the same size satisfy a comparison iff all of their components satisfy it.
- When executing `lhs = rhs` when at least one of `lhs`, `rhs` contains unknown identifiers (and is thus considered a function), both sides are evaluated at every point in `linspace(0, 1, n)`, `linspace(1, 100, n)` and `(101, ..., 100 + n)` as well as their negative counterparts. If they differ at some point, `0` is immediately returned. If they match at all points, `1` is returned. Per default, `n = lang::evaluator::DEFAULT_TESTEQ_REPETITIONS`. One can specify `n` by using `lhs ={e} rhs` where `e` can be any expression evaluable to a float (will then be rounded to the nearest integer). The same works for `<`, `<=`, `>` and `>=`.

### User-defined constants and functions
- Definition of constants: type `x := expr`, where `expr` can be any expression that can be evaluated to an object at the time of the definition.
- Names must start with a character that is either alphabetic, an underscore or among the special characters in the sidebar (e.g. `α`). The following characters may be digits too. Names starting with three underscores (`___`) are forbidden, as they are reserved for internal helper functions and variables.
- Such an assignment returns the evaluation of `expr`, so one can write e.g. `(x := 2) + 1` to obtain `3` as output and define `x` simultaneously.
- If `x` was already a defined constant, this will re-define it and permanently suppress the old value.
- Tuple assignment is supported: write e.g. `(x, y) := rhs` where `rhs` can be evaluated to a tuple of the same size. Function assignment is not allowed in this way.
  The environment is captured before evaluation such that the entire tuple is evaluated based on the same environment (see also section [tuples](#tuples)).
- Definition of functions: type `f(x, y) := 2x + y`. If e.g. `x` already exists as a constant/function, this will be ignored for the sake of the function's definition.
  The `x` on the RHS of the definition will always be the `x` passed as argument, not the constant.<br/>
  If one wants to include a constant from the current environment, simply type `f(y) := 2x + y` where `x` is a pre-defined constant. Note that the
  current value of `x` will be captured at the time of the definition; if you change `x` later on, `f` will still use its old value.
- Delete a constant/function with `del(x)`. You can delete as many constants/functions as you want at once, e.g. `del(x, y, f)`.

### Matrices and vectors
#### Initialization
- Matrices can be initialized by typing `[1, 2, 3; 4, 5, 6; 7, 8, 9]` where the rows will be `[1,2,3]`, `[4,5,6]` and `[7,8,9]` respectively.
  The semicolon `;` can be used interchangeably with a single backslash `\`, even within the same matrix.
- Vectors can be initialized by typing either `[1; 2; 3]` or `[1 \ 2 \ 3]` (as one would initialize a matrix with only one column).
- Unlike for tuples (see section [tuples](#tuples)), the environment is not
  captured before evaluation, meaning that e.g. `x := 1; [x := 2, x + 1]` returns `[2, 3]` and not `[2, 2]`. The reason for this is performance:
  matrices aren't made to contain definitions as entries (unlike tuples in some scenarios) and capturing the environment is costly. In this regard,
  vectors behave like matrices.

#### Elementary operations
The basic matrix/vector operations are all implemented. Below, we only list a remarkable facts about the elementary operations. More complex operations are discussed in the next subsection.
- The product of two vectors of the same dimension is valid and returns their inner product.
- The operations `/`, `%`, `//` are valid whenever one operand (on either side) is a scalar and the other one a matrix/vector. The operation is then simply performed component-wise. For instance, `2 % [x; y]` is equivalent to `[2 % x; 2 % y]`.<br>
Note: defining $1/v$ as $(1/v_1, \ldots, 1/v_n)$ is consistent with the interpretation of vector multiplication as inner product: indeed, for $v \in \R^n$, we then have $v \cdot (1/v) = n = \overrightarrow{1} \cdot \overrightarrow{1}$.
- The operation `not` (`!`) is performed component-wise. Operations `and` and `or` are currently not implemented for matrices/vectors.
- The inverse of a matrix can be computed by simply raising the matrix to the power `-1`, e.g. `A^(-1)`. If the matrix is not invertible, this throws an error.
- A square matrix can be raised to a power $n \in \N_0$ by simply typing `A^n`. The power $n=0$ returns the identity matrix. If the matrix is invertible, negative integer powers are valid too.
- Many matrix functions (e.g. matrix multiplication, transposition) are implemented with view to efficiency for large matrices (using optimization strategies like tiling for better cache locality and parallelization using the `raylib` crate), even though in this specific application, most matrices are likely small.

#### Norms
Norms of vectors/matrices can be computed by typing `||x||_{type}`. The available norm types are listed below. The braces can be omitted if e.g. `type = 1`. If one only types `||x||`, the 2-norm will be used.
- For vectors, the available norms are all $p$-norms for $p \in \mathbb{R} \cup \{\infty\}$.
- For matrices, the same $p$-norms are available. For $p \neq 1, \infty$, these norms can only be approximated. The implementation is based on the [article](https://link.springer.com/article/10.1007/BF01396242) "_Estimating the matrix $p$-norm_" by Nicholas Highham.
- The matrix-$2$-norm is interpreted as the spectral norm here, not as the Frobenius norm. For convenience, the spectral norm can also be accessed via `||A||_{spec}`.
- The Frobenius norm can be accessed by typing either `||A||_f` or `||A||_F`.

#### Matrix functions
A range of standard functions for matrices and vectors are pre-defined:
- `det` computes the determinant of a square matrix.
- `tr` computes the trace of a square matrix.
- `tranpose` transposes the given matrix.
- `eig` returns all eigenvalues (both real and complex) of a square matrix as tuple. Uses a QR-algorithm.
- `adj` returns the adjugate of a square matrix in $\mathcal{O}(n^3)$.

### Tuples
- Tuples can be initialized by typing `(1, 2, 3)`.
- Tuples are polymorphic but only support very few operations. They are primarily intended to support multiple simultaneous assignments (see [custom definitions](#custom-definitions)); generally, the use of vectors is preferred.
- The size of tuples isn't taken into account during type checks. This is used to our advantage for functions returning an unknown amount of values, e.g. the function `eig` (that computes the eigenvalues of a matrix).

### Folded operations
Currently, the implemented folded operations are `sum` and `prod` (product).
- Syntax: `sum_{i=a}^b f(i)` where `i` has to be an identifier, `a` must be evaluable to an integer and `b` to a float (`a`, `b` do not need to be constants). The type of the object inside the sum is inferred. If `a > b` initially, then `0` is returned (in the appropriate type).
- An arbitrary amount of conditions can be added to a folded operation as follows: `sum_{i=a, i != 5, ...}^b f(i)`. All values of `i` that do not satisfy all of the given conditions will be skipped.
- Consider `sum_{i=a}^b f(i)`. Non-constant bounds `a`, `b` are supported in the following sense.
  - `a` is only evaluated once: it may _not_ depend on `i` and it is useless to have it change as `f` is evaluated.
  - `b` is allowed to depend on `i` or change as `f` is evaluated. Then, `b` will be freshly evaluated after every iteration until `i > b`, after which the iteration is terminated. If no such variability of `b` is detected, it is only evaluated once at the start of the iteration. The precise criterion is: the expression `b` is evaluated in every iteration iff `b` contains the identifier `i` (in any form) or `f(i)` contains an assignment `x := ...` for which `b` contains the identifier `x`.
  - Conditions will evidently depend on `i`, but they may also change as `f` is evaluated. Usually, conditions will be evaluated during the iteration anyway, so no further checks need to be done here.
  - In certain built-in double-sums, further optimizations are done to evaluate `a`, `b` and all conditions as few times as possible (for instance `compute_product_derivative_helper` [here](./src/math/operations/folded_operations.rs)).

### Built-in constants and functions
Some constants have aliases for ergonomic purposes. For example, the constant $\pi$ can be accessed both by `pi` and by `π` (the latter being accessible from the sidebar).

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
- `del` allows to delete user-defined objects, see section [user-defined constants and functions](#user-defined-constants-and-functions).
- `show_components(expr)`: if `expr` can be evaluated to a vector, gives back a representation of `expr` in which the components of the result are clearly visible (i.e. a representation as `Expression::Vector`). For example, `show_components([x; 1] + [1; y])` outputs `[x + 1; 1 + y]`. Behaves analogously if `expr` can be evaluated to a matrix.

#### Helper functions
These functions are prefixed with `___`.
- `___helper_matrix_prod(j, k, a, b, i, f(i))`, where the first 4 arguments are objects and the last 2 are expressions. Computes the `(j, k)`-entry of the matrix `prod_{i=a}^b f(i)` You can add an arbitrary number of conditions as expressions after these arguments.
- `___diff_num(expr, v_1, ..., v_n, p_1, ..., p_n, d_1, ..., d_n)`, where `expr` is an expression, `v_i` are identifiers and `p_i`, `d_i` are objects. Computes the numerical directional derivative of `expr` w.r.t. the variables `v_1, ..., v_n` at point `(p_1, ..., p_n)` in direction `(d_1, ..., d_n)`. Hint: set `d_i := \delta_{i,j}` to compute the partial derivative of `expr` w.r.t. `v_j`.

### Differentiation

#### First-order partial derivatives
Partially differentiate an expression by typing e.g. `d/dx (x^3 + 2x + 1)`. This returns `3x^2 + 2` as expression.
- Beware that by mathematical convention, the operator `d/dx` binds very strongly. Hence, `d/dx x^2` will be interpreted as `(d/dx x)^2`.
- The resulting expression can be stored in a function by typing `f(x) := d/dx g(x)`. Then, the partial derivative will be evaluated _before defining `f`_, that is, the defining expression of `f(x)` will the result of the partial derivative, not the unmodified expression `d/dx g(x)`. For instance, typing `f(x) := d/dx (x^2)` will effectively define `f(x) := 2x`.
- Differentiating a function with a matrix/vector as output will differentiate component-wise and return the corresponding matrix/vector-valued function.

#### Higher order partial derivatives
For higher derivatives, use the natural notation, for example `d^2/(dx dy) f(x, y)` or `d^3/(dx^2 dy) f(x, y)`. This will also return an expression.
  - If the exponent in the numerator doesn't match with the sum of the exponents in the denominator, e.g. in `d^3/(dx dy)`, a warning is emitted and the numerator's exponent is ignore. For instance, `d^3/(dx dy) x*y` will emit a warning but then output `1`.
  - Exponents do not need to be constant numbers. For instance, the syntax `d^n/dx^n` for a user-defined variable `n` is allowed.
  - Recall that writing `f(x) := d^n/dx^n g(x)` will evaluate the derivative before defining `f`. Therefore, the syntax `f(x, n) := d^n/dx^n g(x)` is generally _not_ valid (except if `n` already exists outside of this definition).

#### Directional derivatives
For the directional derivative, multiple syntaxes are available. All of them return an object and not an expression (unlike the partial derivative).
- `D_x expr (p)[d]` computes the derivative of `expr` w.r.t. `x` at point `p` in direction `d`.
- `D_{x, y} expr (p_x, p_y)[d_x, d_y]` computes the derivative of `expr` w.r.t. `(x, y)` at point `(p_x, p_y)` in direction `(d_x, d_y)`. Analogously for any higher number of variables.
- `D f(p)[d]`: If `f` is a function, then the shortened syntax `D f(p)[d]` computes the exact same result as `D_{x_1, ..., x_n} f(x_1, ..., x_n) (p)[d]` where `n` is the number of arguments that `f` expects.
- `D expr (p_1, ..., p_n)[d_1, ..., d_m]`: collects all unknown identifiers within `expr` into a vector in ascending alphabetic order `x_1, ..., x_l`. If `l=m=n`, this syntax then returns the same result as `D_{x_1, ..., x_n} expr (p_1, ..., p_n)[d_1, ..., d_n]`. Otherwise, throws an error.

### Special syntaxes and remarks
- `debug` prints the entire current environment (constants + functions).
- Notice that the token `!` acts as both the `not` operator and the factorial operator. In context, one can always differentiate between the two, with one minor downside:
  the syntax `x * (!y)` cannot be shortened to `x !y` (since these spaces disappear while tokenizing, one would not be able to distinguish this from `(x!) * y`).
