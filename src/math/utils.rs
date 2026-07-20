//! Most functions in this module have for sole objective to simplify typing and enhance readability.

use std::collections::HashSet;

use crate::math::Object;


const ABS_TOL: f64 = 1e-12;
const REL_TOL: f64 = 1e-10;


pub trait Quo<Rhs = Self> {
    type Output;
    fn quo(self, rhs: Rhs) -> Self::Output;
}
pub trait QuoAssign<Rhs = Self> {
    fn quo_assign(&mut self, rhs: Rhs);
}


#[inline]
pub fn approx_eq(x: f64, y: f64) -> bool {
    // We use the criterion |x-y| <= max(ABS_TOL, REL_TOL * max(|x|, |y|))
    (x-y).abs() <= ABS_TOL.max(REL_TOL * x.abs().max(y.abs()))
}

pub fn quo(x: f64, y: f64) -> f64 {
    ((x - (x.rem_euclid(y))) / y).round()
}

#[inline]
/// Returns the maximum of the given iterator of floats. If the iterator is empty, returns 0.0.
pub fn max(iter: impl Iterator<Item=f64>) -> f64 {
    iter.fold(-f64::INFINITY, f64::max)
}

#[inline]
/// Returns the minimum of the given iterator of floats. If the iterator is empty, returns 0.0.
pub fn min(iter: impl Iterator<Item=f64>) -> f64 {
    iter.fold(f64::INFINITY, f64::min)
}

#[inline]
/// Returns the maximum absolute value of the given iterator of floats. If the iterator is empty, returns 0.0.
pub fn max_abs<'a>(iter: impl Iterator<Item=&'a f64>) -> f64 {
    iter.fold(0.0, |acc, x| f64::max(acc, x.abs()))
}

/// Returns the `i`-th row of `v` as slice where `n` is the length of a row.
/// 
/// The returned slice therefore has length `n`.
#[inline]
pub fn row(v: &[f64], i: usize, n: usize) -> &[f64] {
    &v[i * n .. (i+1) * n]
}
/// Returns the `j`-th column of `v` as iterator where `m` is the number of rows to take
/// and `n` is the length of each row.
/// 
/// The returned iterator therefore iterates over `m` elements.
#[inline]
pub fn col(v: &[f64], j: usize, m: usize, n: usize) -> std::iter::Map<std::ops::Range<usize>, impl FnMut(usize) -> f64> {
    (0..m).map(move |i| v[i * n + j])
}

/// Returns whether the given permutation has even parity (`true`) or odd parity (`false`).
/// 
/// `permutation` must be a permutation of the vector `[0, ..., n-1]` for some `n`.
pub fn permutation_parity(permutation: &[usize]) -> bool {
    // Fact: a permutation is odd iff it has an odd number of even-length cycles.
    let mut remaining: HashSet<usize> = HashSet::from_iter(0..permutation.len());
    let mut is_even = true;
    while let Some(&start) = remaining.iter().next() {
        remaining.remove(&start);
        let mut i = 1;
        let mut curr = start;
        while permutation[curr] != start {
            i += 1;
            curr = permutation[curr];
            remaining.remove(&curr);
        }
        if i % 2 == 0 {
            is_even = !is_even;
        }
    }
    is_even
}

/// Returns the inverse permutation of `permutation`.
pub fn transpose_permutation(permutation: &[usize]) -> Vec<usize> {
    let mut inv = vec![0; permutation.len()];
    for i in 0..permutation.len() {
        inv[permutation[i]] = i;
    }
    inv
}

/// Acts like `format!("{:.decimals}", x)` but cuts off trailing zeros.
pub fn format_trimmed(x: f64, decimals: usize) -> String {
    let s = format!("{:.prec$}", x, prec = decimals);
    let s = s.trim_end_matches('0');
    let s = s.trim_end_matches('.');
    s.to_string()
}

// /// Splits the interval [a, b] into n uniformly spread points, the first of which equals a and the last of which equals b.
// /// 
// /// Exceptions: if `n == 0`, returns an empty vector. If `n == 1`, returns `vec![a]`.
// /// 
// /// Note: if `a > b`, returns `linspace(b, a, n).rev()`.
// pub fn linspace(a: f64, b: f64, n: usize) -> Vec<f64> {
//     if n == 0 {return Vec::<f64>::new();}
//     if n == 1 {return vec![a];}
//     let step = (b-a) / ((n-1) as f64);
//     (0..n).map(|i| a + i as f64 * step).collect()
// }

/// As `linspace` but directly converts all floats to `Object`s.
pub fn linspace_as_objects(a: f64, b: f64, n: usize) -> Vec<Object> {
    if n == 0 {return Vec::<Object>::new();}
    if n == 1 {return vec![Object::Real(a)];}
    let step = (b-a) / ((n-1) as f64);
    (0..n).map(|i| Object::Real(a + i as f64 * step)).collect()
}