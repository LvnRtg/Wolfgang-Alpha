//! Implements transpositions both in simple ways (for small matrices) and more complex approaches (for large matrices).
//! The main optimization strategies are tiling and parallelization (using `rayon`).

use rayon::prelude::*;

use crate::math::{Matrix};
use crate::math::BLOCK_SIZE;

/// Somewhat arbitrary constant, no benchmarks done yet to fine-tune it.
const PARALLELIZATION_THRESHOLD: usize = 16;

impl Matrix {
    /// Transposes `self` using parallelization if `self` is large.
    pub fn transpose(&self) -> Matrix {
        if self.m == 0 || self.n == 0 {
            Matrix::from(0, 0, vec![])
        } else if self.m.max(self.n) >= PARALLELIZATION_THRESHOLD {
            self.transpose_parallel()
        } else {
            self.transpose_simple()
        }
    }

    /// Computes `self^T` using simple blocking.
    /// 
    /// We use this method for smaller matrices.
    fn transpose_simple(&self) -> Matrix {
        let (m, n) = (self.m, self.n);
        let mut out = vec![0.0f64; self.values.len()];
        for ib_start in (0..m).step_by(BLOCK_SIZE) {
            let i_end = (ib_start + BLOCK_SIZE).min(m);
            for jb_start in (0..n).step_by(BLOCK_SIZE) {
                let jb_end = (jb_start + BLOCK_SIZE).min(n);
                self.transpose_tile(&mut out, ib_start, i_end, jb_start, jb_end);
            }
        }
        Matrix { m: self.n, n: self.m, values: out }
    }

    /// Computes `self^T` using parallelization and tiling.
    ///
    /// Splits the output into contiguous row-chunks of `BLOCK_SIZE` rows each.
    /// Each chunk is a disjoint mutable slice, so no `unsafe` is needed.
    /// 
    /// We use this method for larger matrices.
    fn transpose_parallel(&self) -> Matrix {
        let mut out = vec![0.0f64; self.values.len()];
        out.par_chunks_mut(BLOCK_SIZE * self.m)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let j_start = chunk_idx * BLOCK_SIZE; // First output row contained in the current `out_chunk`
                let j_end = (j_start + BLOCK_SIZE).min(self.n); // Last one
                self.transpose_row_range_into_chunk(out_chunk, j_start, j_end);
            });
        Matrix { m: self.n, n: self.m, values: out }
    }

    /// Transpose rows `[j_start, j_end)` of `self` into the given chunk `out_chunk`,
    /// which should represent rows `[j_start, j_end)` of the output packed contiguously.
    fn transpose_row_range_into_chunk(
        &self,
        out_chunk: &mut [f64],
        j_start: usize,
        j_end: usize,
    ) {
        // `ib` stands for `i block`, similarly for `j`.
        for ib_start in (0..self.m).step_by(BLOCK_SIZE) {
            let ib_end = (ib_start + BLOCK_SIZE).min(self.m);
            for jb_start in (j_start..j_end).step_by(BLOCK_SIZE) {
                let jb_end = (jb_start + BLOCK_SIZE).min(j_end);
                self.transpose_tile_into_chunk(out_chunk, ib_start, ib_end, jb_start, jb_end, j_start);
            }
        }
    }

    /// Transposes the single tile `(ib_start..ib_end, jb_start..jb_end)` from `self` into `out`.
    /// 
    /// `ib` stands for `i block`, similarly for `j`.
    ///
    /// The read side (`src_row`) is sliced and iterated so LLVM can prove bounds and autovectorize
    /// instead of having bound-checks for every element..
    /// The write side is still strided (inherent to transpose) but confined to a
    /// tile small enough to stay resident in L1 while it's being written.
    fn transpose_tile(
        &self,
        out: &mut [f64],
        ib_start: usize,
        ib_end: usize,
        jb_start: usize,
        jb_end: usize,
    ) {
        for i in ib_start..ib_end {
            let row_start = i * self.n + jb_start;
            let src_row = &self.values[row_start..row_start + (jb_end - jb_start)];
            for (dj, &v) in src_row.iter().enumerate() {
                let j = jb_start + dj;
                out[j * self.m + i] = v;
            }
        }
    }

    /// Same as `transpose_tile`, but writes into a chunk that only holds output
    /// rows starting at `j_offset` (so row `j` lives at local row `j - j_offset`).
    fn transpose_tile_into_chunk(
        &self,
        out_chunk: &mut [f64],
        ib_start: usize,
        i_end: usize,
        jb_start: usize,
        jb_end: usize,
        j_offset: usize,
    ) {
        for i in ib_start..i_end {
            let row_start = i * self.n + jb_start;
            let src_row = &self.values[row_start..row_start + (jb_end - jb_start)];
            for (dj, &v) in src_row.iter().enumerate() {
                let j = jb_start + dj;
                out_chunk[(j - j_offset) * self.m + i] = v;
            }
        }
    }
}