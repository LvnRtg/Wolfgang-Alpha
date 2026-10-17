use crate::{dispatch_matrix, dispatch_vector};
/// Implements an algorithm to attempt to estimate the limit of a sequence
/// of scalars (real or complex). This generality leads to a slight loss in
/// performance because one cannot track minimum and maximum in the active
/// window (cf. struct `Window` below), as there is no canonical `>`-relation
/// on the complex numbers.

use crate::math::{Object, traits::*};
use crate::status::{ExtResult, Status, WARNING_CAP};

const WINDOW_SIZE: usize = 5;
const MAX_ITERATIONS: usize = 100000;
/// When `(sup_n y_n) - first` exceeds this threshold, we assume that `y_n` diverges
/// to +∞. The same holds for the infimum.
const INFINITY_THRESHOLD: f64 = 1e8;
const LOCAL_TOL: f64 = 1e-6;
const DRIFT_TOL: f64 = 1e-4;


/// Attempts to numerically approximate the limit  of the sequence `(f(n))_{n ∈ indices}`,
/// where `indices` is an iterator progressively yielding "larger" `n` ("larger" could
/// also mean "smaller" when considering a limit like `n -> -∞`).
/// 
/// `indices` is assumed to be an infinite iterator.
/// 
/// Such an algorithm can never be correct because the limit of a sequence
/// is independent of any finite number of terms of the sequence.
#[inline]
pub fn compute_limit_numerically<F>(
    mut f: F,
    mut indices: impl Iterator<Item=f64>
) -> ExtResult
where
    F: FnMut(f64) -> ExtResult
{
    f(indices.next().unwrap())
    .and_then(|s| s.try_map_flatten(
        |y_0| compute_recursive_limit_numerically(
            y_0,
            |n, _| f(n),
            indices
        )
    ))
}

fn is_infinite(obj: &Object) -> bool {
    match obj {
        Object::Real(x) => x.is_infinite(),
        Object::Complex(c) => c.is_infinite(),
        Object::Vector(x) => dispatch_vector!(x, v => v.iter().any(|e| e.is_infinite())),
        Object::Matrix(x) => dispatch_matrix!(x, v => v.iter().any(|e| e.is_infinite())),
        _ => false
    }
}

/// Returns `max_{x,y ∈ A} |x-y|` for `A := v[..-WINDOW_SIZE]`. There is an asymptotically
/// more efficient algorithm using convex hulls, but since the
/// window size is typically small, the overhead would be larger
/// than the gained performance.
fn stride(v: &[Object]) -> Result<f64, String> {
    let mut max = 0.0;
    for (i, x) in v[v.len() - WINDOW_SIZE..].iter().enumerate() {
        for y in v[v.len() - WINDOW_SIZE + i + 1..].iter() {
            let d = x.distance_to(y)?;
            if d > max {
                max = d;
            }
        }
    }
    Ok(max)
}

/// Attempts to numerically approximate the limit  of the sequence `(y_n)_{n ∈ indices}`
/// where the first term of the sequence must be explicitly provided as a parameter
/// and the subsequent terms are given by the relation `y_{next_index(n)} = f(next_index(n), y_n)`.
/// 
/// Such an algorithm can never be correct because the limit of a sequence
/// is independent of any finite number of terms of the sequence.
/// 
/// Multiple criteria are checked to determine whether the iteration should end.
/// For the purposes of documentation, assume for simplicity that `indices = ℕ₀`.
/// - For the window `W := {y_{n-WINDOW_SIZE+1}, ..., y_n}`, we must have `max W - min W <= LOCAL_TOL`.
/// - For the drift `d := |y_n - y_{n/2}|`, we must have `d <= DRIFT_TOL`.
/// - We take a reference point `r` after a few terms (specifically, this will be `y_{WINDOW_SIZE}`).
///   If we have `y_n - r >= INFINITY_THRESHOLD`, we assume the sequence diverges to `+∞`.
///   Likewise, if `r - y_n >= INFINITY_THRESHOLD`, we assume the sequence diverges to `-∞`.
/// 
/// Notice in the implementation that the call to `Window::construct` forces us to construct
/// at least `WINDOW_SIZE` terms of the sequence. This is intentional to avoid singularities
/// and strong variations that may occur at the start of a sequence.
pub fn compute_recursive_limit_numerically<F>(
    y_0: Object,
    mut f: F,
    mut indices: impl Iterator<Item=f64>
) -> ExtResult
where
    F: FnMut(f64, &Object) -> ExtResult
{
    let mut warnings = Vec::new();
    let mut sequence = Vec::with_capacity(WINDOW_SIZE);
    sequence.push(y_0);
    for i in 1..WINDOW_SIZE {
        sequence.push(f(indices.next().unwrap(), &sequence[i - 1])?.unpack_into_with_cap(&mut warnings, WARNING_CAP));
    }
    let reference = sequence[sequence.len() - 1].clone();
    let ref_as_float = reference.expect_float();
    // If `x₂₀` is infinite, it is reasonable to assume that the limit is infinite too.
    if is_infinite(&reference) {
        return Ok(Status { value: reference, warnings });
    }
    for _ in WINDOW_SIZE..MAX_ITERATIONS {
        let n = sequence.len() - 1;
        sequence.push(f(indices.next().unwrap(), &sequence[n])?.unpack_into_with_cap(&mut warnings, WARNING_CAP));
        // Check for convergence
        if stride(&sequence)? < LOCAL_TOL && sequence[n].distance_to(&sequence[sequence.len() / 2])? < DRIFT_TOL {
            return Ok(Status { value: sequence.pop().unwrap(), warnings });
        }
        // Check for divergence
        else if let (Ok(y), Ok(r)) = (&sequence[n].expect_float(), &ref_as_float) {
            let diff = y - r;
            if diff >= INFINITY_THRESHOLD {
                return Ok(Status { value: Object::Real(f64::INFINITY), warnings });
            } else if diff <= -INFINITY_THRESHOLD {
                return Ok(Status { value: Object::Real(-f64::INFINITY), warnings });
            }
        } else if sequence[n].distance_to(&reference)? >= INFINITY_THRESHOLD {
            return Err("Complex sequence diverges.".to_string());
        }
    }
    if stride(&sequence)? < 0.1 && sequence[sequence.len() - 1].distance_to(&sequence[sequence.len() / 2])? < 0.1 {
        warnings.push("The convergence is uncertain: the error terms exceed the tolerance but are still relatively small.".to_string());
        Ok(Status { value: sequence.pop().unwrap(), warnings })
    } else {
        Err(format!("No convergence detected after {MAX_ITERATIONS} iterations."))
    }
}