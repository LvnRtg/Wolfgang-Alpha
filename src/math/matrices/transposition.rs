//! Implements transpositions both in simple ways (for small matrices) and more complex approaches (for large matrices).
//! The main optimization strategies are tiling and parallelization (using `rayon`).

use rayon::prelude::*;

use crate::math::BLOCK_SIZE;
use crate::math::traits::Scalar;
use super::*;


/// Somewhat arbitrary constant, no benchmarks done yet to fine-tune it.
const PARALLELIZATION_THRESHOLD: usize = 16;

impl<T: Scalar> Matrix<T> {
    /// Transposes `self` using parallelization if `self` is large.
    #[inline]
    pub fn transpose(&self) -> Matrix<T> {
        self.view().transpose()
    }
}


impl<'a, T: Scalar> MatrixView<'a, T> {
    /// Returns the transpose as a new contiguous `Matrix`. Parallelizes if the view is large.
    pub fn transpose(&self) -> Matrix<T> {
        if self.rows() == 0 || self.cols() == 0 {
            Matrix::from(self.cols(), self.rows(), vec![])
        } else if self.rows().max(self.cols()) >= PARALLELIZATION_THRESHOLD {
            self.transpose_parallel()
        } else {
            self.transpose_simple()
        }
    }

    /// Simple blocked transpose, used for smaller views.
    fn transpose_simple(&self) -> Matrix<T> {
        let (m, n) = (self.rows(), self.cols());
        let mut out = vec![T::zero(); m * n];
        for ib_start in (0..m).step_by(BLOCK_SIZE) {
            let ib_end = (ib_start + BLOCK_SIZE).min(m);
            for jb_start in (0..n).step_by(BLOCK_SIZE) {
                let jb_end = (jb_start + BLOCK_SIZE).min(n);
                self.transpose_tile_into_chunk(&mut out, ib_start, ib_end, jb_start, jb_end, 0);
            }
        }
        Matrix::from(n, m, out)
    }

    /// Parallel blocked transpose, used for larger views.
    ///
    /// The output (which has `self.cols` rows of length `self.rows`) is split into
    /// disjoint chunks of `BLOCK_SIZE` output rows each, so no `unsafe` is needed.
    fn transpose_parallel(&self) -> Matrix<T> {
        let (m, n) = (self.rows(), self.cols());
        let mut out = vec![T::zero(); m * n];
        out.par_chunks_mut(BLOCK_SIZE * m)
            .enumerate()
            .for_each(|(chunk_idx, out_chunk)| {
                let j_start = chunk_idx * BLOCK_SIZE; // first output row in this chunk
                let j_end = (j_start + BLOCK_SIZE).min(n); // one past the last
                self.transpose_row_range_into_chunk(out_chunk, j_start, j_end);
            });
        Matrix::from(n, m, out)
    }

    /// Transposes columns `[j_start, j_end)` of `self` into `out_chunk`, which holds
    /// output rows `[j_start, j_end)` packed contiguously.
    fn transpose_row_range_into_chunk(&self, out_chunk: &mut [T], j_start: usize, j_end: usize) {
        for ib_start in (0..self.rows()).step_by(BLOCK_SIZE) {
            let ib_end = (ib_start + BLOCK_SIZE).min(self.rows());
            for jb_start in (j_start..j_end).step_by(BLOCK_SIZE) {
                let jb_end = (jb_start + BLOCK_SIZE).min(j_end);
                self.transpose_tile_into_chunk(out_chunk, ib_start, ib_end, jb_start, jb_end, j_start);
            }
        }
    }

    /// Transposes the tile `(ib_start..ib_end, jb_start..jb_end)` of `self` into a chunk
    /// that holds output rows starting at `j_offset` (row `j` lives at local row
    /// `j - j_offset`). The output row length is `self.rows`.
    ///
    /// The read side is a contiguous slice (using `self.stride` to find the row start),
    /// so bounds checks are hoisted and LLVM can vectorize. The strided write stays
    /// confined to a tile small enough for L1.
    fn transpose_tile_into_chunk(
        &self,
        out_chunk: &mut [T],
        ib_start: usize,
        ib_end: usize,
        jb_start: usize,
        jb_end: usize,
        j_offset: usize,
    ) {
        for i in ib_start..ib_end {
            let src_row = &self.row_slice(i)[jb_start..jb_end];
            for (dj, &v) in src_row.iter().enumerate() {
                let j = jb_start + dj;
                out_chunk[(j - j_offset) * self.rows() + i] = v;
            }
        }
    }
}