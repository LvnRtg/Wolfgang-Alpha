use std::collections::HashSet;

use crate::math::Object;

pub fn approx_eq(x: f64, y: f64) -> bool {
    (x-y).abs() <= 1e-10
}

#[inline]
/// Returns the maximum of the given iterator of floats. If the iterator is empty, returns 0.0.
pub fn max(iter: impl Iterator<Item=f64>) -> f64 {
    iter.fold(0.0, f64::max)
}

#[inline]
/// Returns the maximum absolute value of the given iterator of floats. If the iterator is empty, returns 0.0.
pub fn max_abs<'a>(iter: impl Iterator<Item=&'a f64>) -> f64 {
    iter.fold(0.0, |acc, x| f64::max(acc, x.abs()))
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
    if n == 1 {return vec![Object::Float(a)];}
    let step = (b-a) / ((n-1) as f64);
    (0..n).map(|i| Object::Float(a + i as f64 * step)).collect()
}