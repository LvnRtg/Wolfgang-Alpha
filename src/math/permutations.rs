//! Implements permutations, by which we understand a reordering of a finite set `{0, ..., n-1}`.

use num_traits::{One, Zero};
use std::collections::HashSet;

use super::Matrix;

/// A permutation of a set `{0, ..., n-1}` for some natural number `n`.
/// 
/// Applying it to a vector `v` would turn `[v[0], ..., v[n]]` into `[v[perm[0]], v[perm[1]], ..., v[perm[n]]]`.
pub struct Permutation {
    permutation: Vec<usize>
}

impl std::ops::Index<usize> for Permutation {
    type Output = usize;
    fn index(&self, index: usize) -> &Self::Output {
        &self.permutation[index]
    }
}

impl Permutation {
    /// Checks if this vector is indeed a permutation of `{0, ..., n-1}`. If so, converts it to a permutation.
    pub fn from_vec(v: Vec<usize>) -> Option<Permutation> {
        let mut seen = vec![false; v.len()];
        for x in v.iter() {
            if std::mem::replace(&mut seen[*x], true) {
                return None;
            }
        }
        Some(Permutation{permutation: v})
    }
    /// Initializes a permutation without checking that the given `v` correponds to one.
    /// 
    /// ## This may cause unexpected behavior and panicking.
    pub fn from_vec_unchecked(v: Vec<usize>) -> Permutation {
        Permutation {permutation: v}
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.permutation.len()
    }
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item=&usize> {
        self.permutation.iter()
    }

    /// Returns whether the given permutation has even parity (`true`) or odd parity (`false`).
    pub fn parity(&self) -> bool {
        // Fact: a permutation is odd iff it has an odd number of even-length cycles.
        let mut remaining: HashSet<usize> = HashSet::from_iter(0..self.permutation.len());
        let mut is_even = true;
        while let Some(&start) = remaining.iter().next() {
            remaining.remove(&start);
            let mut loop_has_even_length = false;
            let mut curr = start;
            while self.permutation[curr] != start {
                loop_has_even_length ^= true;
                curr = self.permutation[curr];
                remaining.remove(&curr);
            }
            if loop_has_even_length {
                is_even ^= true;
            }
        }
        is_even
    }

    /// Returns the inverse permutation of `self`.
    pub fn transpose(&self) -> Permutation {
        let mut inv = vec![0; self.len()];
        for (i, p) in self.iter().enumerate() {
            inv[*p] = i;
        }
        Permutation{permutation: inv}
    }

    /// Converts the given permutation to a matrix `P` such that `P*A` permutes the rows of `A` according to the permutation
    /// and `A*P` permutes the columns of `A` according to the permutation.
    pub fn to_matrix<T>(&self) -> Matrix<T> where T: Copy + One + Zero {
        // The `i`-th row of `P` should be the row `self[i]` of the identity matrix, that is, it should have the `1` at index `self[i]`.
        let mut values = Vec::<T>::with_capacity(self.len() * self.len());
        for p in self.iter() {
            values.extend(std::iter::repeat_n(T::zero(), *p));
            values.push(T::one());
            values.extend(std::iter::repeat_n(T::zero(), self.len() - p - 1));
        }
        Matrix::<T>::from(self.len(), self.len(), values)
    }

    /// Applies the given permutation to the rows of the given matrix.
    /// 
    /// Effectively, the row `i` of the new matrix will be the row `permutation[i]` of `self`.
    /// 
    /// Returns `None` if the permutation's size doesn't match the matrices size.
    pub fn apply_to_matrix_rows<T>(&self, matrix: &Matrix<T>) -> Option<Matrix<T>> where T: Copy {
        if self.len() != matrix.m() {return None;}
        let mut values = Vec::<T>::with_capacity(matrix.total_size());
        for p in self.iter() {
            // Writes are sequential and reads are sequential within each row. By the nature
            // of permutations, this is optimal in the sense that even with tiling, you may
            // have to jump between rows either when reading or when writing.
            values.extend(matrix.row_slice(*p).iter());
        }
        Some(Matrix::<T>::from(matrix.m(), matrix.n(), values))
    }
    /// Applies the given permutation to columns of the given matrix.
    /// 
    /// Effectively, the column `j` of the new matrix will be the column `permutation[j]` of `self`.
    /// 
    /// Returns `None` if the permutation's size doesn't match the matrices size.
    pub fn apply_to_matrix_columns<T>(&self, matrix: &Matrix<T>) -> Option<Matrix<T>> where T: Copy {
        if self.len() != matrix.n() {return None;}
        let mut values = Vec::<T>::with_capacity(matrix.total_size());
        for i in 0..matrix.m() {
            // As for row permutations, writes are sequential and reads are sequential within each row.
            self.iter().for_each(|p| values.push(matrix.get(i, *p).clone()));
        }
        Some(Matrix::<T>::from(matrix.m(), matrix.n(), values))
    }
}