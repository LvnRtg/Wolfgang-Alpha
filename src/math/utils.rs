use std::collections::HashSet;

pub fn approx_eq(x: &f64, y: &f64) -> bool {
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
pub fn permutation_parity(permutation: &Vec<usize>) -> bool {
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