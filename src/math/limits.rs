use std::mem::MaybeUninit;

use crate::math::{Object, utils};
use crate::status::{ExtResult, Status, WARNING_CAP};

const WINDOW_SIZE: usize = 10;
const MAX_ITERATIONS: usize = 100000;
/// When `(sup_n y_n) - first` exceeds this threshold, we assume that `y_n` diverges
/// to +∞. The same holds for the infimum.
const INFINITY_THRESHOLD: f64 = 1e8;
const LOCAL_TOL: f64 = 1e-6;
const DRIFT_TOL: f64 = 1e-4;


struct Window {
    start: usize,
    min: f64,
    max: f64,
    values: [f64; WINDOW_SIZE]
}
impl Window {
    fn construct<F>(
        mut y_0: f64,
        f: &mut F,
        indices: &mut impl Iterator<Item=f64>,
        warnings: &mut Vec<String>
    ) -> Result<Self, String>
    where
        F: FnMut(f64, f64) -> Result<Status<f64>, String>
    {
        let mut buf = [const { MaybeUninit::uninit() }; WINDOW_SIZE];
        buf.first_mut().unwrap().write(y_0);
        let mut max = y_0;
        let mut min = y_0;
        for slot in buf.iter_mut().skip(1) {
            y_0 = f(indices.next().unwrap(), y_0)?.unpack_into_with_cap(warnings, WARNING_CAP);
            if y_0 > max {
                max = y_0;
            } else if y_0 < min {
                min = y_0;
            }
            slot.write(y_0);
        }
        Ok(Window {
            start: 0,
            min,
            max,
            values: unsafe { std::mem::transmute(buf) }
        })
    }

    /// Adds the given value at the end of the window, removing the first value of the window.
    fn add(&mut self, new: f64) {
        let old = std::mem::replace(&mut self.values[self.start], new);
        if self.start < WINDOW_SIZE - 1 {
            self.start += 1;
        } else {
            self.start = 0;
        }
        // The following computations are made so that if a sequence is monotonic,
        // as few operations as possible are required.
        if new >= self.max {
            self.max = new;
        } else if old == self.max {
            self.max = *utils::max_of(self.values.iter()).unwrap();
        }
        if new <= self.min {
            self.min = new;
        } else if old == self.min {
            self.min = *utils::min_of(self.values.iter()).unwrap();
        }
    }

    /// Returns `max(self) - min(self)`.
    fn stride(&self) -> f64 {
        self.max - self.min
    }

    #[inline]
    fn get(&self, i: usize) -> f64 {
        self.values[(self.start + i) % WINDOW_SIZE]
    }
    #[inline]
    fn to_vec(&self) -> Vec<f64> {
        self.values.to_vec()
    }
}


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
    F: FnMut(f64) -> Result<Status<f64>, String>
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
    y_0: f64,
    mut f: F,
    mut indices: impl Iterator<Item=f64>
) -> ExtResult
where
    F: FnMut(f64, f64) -> Result<Status<f64>, String>
{
    let mut warnings = Vec::new();
    let mut window = Window::construct(y_0, &mut f, &mut indices, &mut warnings)?;
    // Due to the iteration cap, this won't exceed `8 * MAX_ITERATIONS` bytes in size.
    let mut sequence = window.to_vec();
    let mut y_n = window.get(WINDOW_SIZE - 1);
    let reference = y_n;
    // If `x₂₀` is infinite, it is reasonable to assume that the limit is infinite too.
    if reference.is_infinite() {
        return Ok(Status { value: Object::Real(reference), warnings });
    }
    for _ in WINDOW_SIZE..MAX_ITERATIONS {
        y_n = f(indices.next().unwrap(), y_n)?.unpack_into_with_cap(&mut warnings, WARNING_CAP);
        window.add(y_n);
        sequence.push(y_n);
        if window.stride() < LOCAL_TOL && (y_n - sequence[sequence.len() / 2]).abs() < DRIFT_TOL {
            return Ok(Status { value: Object::Real(y_n), warnings });
        } else if y_n - reference >= INFINITY_THRESHOLD {
            return Ok(Status { value: Object::Real(f64::INFINITY), warnings });
        } else if reference - y_n >= INFINITY_THRESHOLD {
            return Ok(Status { value: Object::Real(-f64::INFINITY), warnings });
        }
    }
    if window.stride() < 0.1 && (y_n - sequence[sequence.len() / 2]).abs() < 0.1 {
        warnings.push("The convergence is uncertain: the error terms exceed the tolerance but are still relatively small.".to_string());
        Ok(Status { value: Object::Real(y_n), warnings })
    } else {
        Err(format!("No convergence detected after {MAX_ITERATIONS} iterations."))
    }
}