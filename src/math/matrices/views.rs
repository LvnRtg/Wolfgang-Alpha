use itertools::Itertools;
use rayon::prelude::*;
use std::marker::PhantomData;
use std::ptr::NonNull;

use super::*;


/// In some scenarios, mathematically safe code can't be written in safe Rust, see e.g.
/// `square_root.rs/sqrt_of_upper_triangular_blocking_method`.
#[derive(Clone, Copy)]
pub struct MatrixView<'a, T> {
    ptr: NonNull<T>,
    rows: usize,
    cols: usize,
    stride: usize,
    _marker: PhantomData<&'a T>,
}
pub struct MatrixViewMut<'a, T> {
    /// Points to first entry of the underlying matrix.
    ptr: NonNull<T>,
    rows: usize,
    cols: usize,
    stride: usize,
    _marker: PhantomData<&'a mut T>
}

// Sound because a view only ever touches its own disjoint elements.
unsafe impl<T: Send> Send for MatrixView<'_, T> {}
unsafe impl<T: Sync> Sync for MatrixView<'_, T> {}
unsafe impl<T: Send> Send for MatrixViewMut<'_, T> {}
unsafe impl<T: Sync> Sync for MatrixViewMut<'_, T> {}


impl<T> Matrix<T> {
    pub fn view(&self) -> MatrixView<'_, T> {
        MatrixView::from_slice(&self.values, self.m, self.n)
    }
    pub fn view_mut(&mut self) -> MatrixViewMut<'_, T> {
        MatrixViewMut::from_slice(&mut self.values, self.m, self.n)
    }
}

impl<'a, T> MatrixView<'a, T> {
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T where T: Copy {
        assert!(i < self.rows && j < self.cols);
        unsafe { self.ptr.as_ptr().add(i * self.stride + j).read() }
    }
    #[inline] pub fn rows(&self) -> usize {self.rows}
    #[inline] pub fn cols(&self) -> usize {self.cols}

    pub fn from_slice(data: &'a [T], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        let ptr = NonNull::new(data.as_ptr() as *mut T).unwrap_or(NonNull::dangling());
        MatrixView { ptr, rows, cols, stride: cols, _marker: PhantomData }
    }

    pub fn to_matrix(self) -> Matrix<T> where T: Copy {
        Matrix::from(
            self.rows,
            self.cols,
            [0..self.rows, 0..self.cols].into_iter().multi_cartesian_product().map(
                |v| self.get(v[0], v[1])
            ).collect()
        )
    }

    #[inline]
    pub fn row_slice(&self, i: usize) -> &'a [T] {
        // SAFETY: row `i` lies within this view's element set; the returned
        // lifetime `'a` is valid since the source data outlives this view.
        assert!(i < self.rows);
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(i * self.stride), self.cols) }
    }

    pub fn submatrix(&self, r0: usize, c0: usize, rows: usize, cols: usize) -> MatrixView<'a, T> {
        assert!(r0 + rows <= self.rows && c0 + cols <= self.cols);
        let ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(r0 * self.stride + c0)) };
        MatrixView { ptr, rows, cols, stride: self.stride, _marker: PhantomData }
    }
}

impl<'a, T> MatrixViewMut<'a, T> {
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> T where T: Copy {
        assert!(i < self.rows && j < self.cols);
        unsafe { self.ptr.as_ptr().add(i * self.stride + j).read() }
    }
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, value: T) {
        assert!(i < self.rows && j < self.cols);
        unsafe { self.ptr.as_ptr().add(i * self.stride + j).write(value); }
    }
    #[inline] pub fn rows(&self) -> usize {self.rows}
    #[inline] pub fn cols(&self) -> usize {self.cols}
    
    /// Builds a view over a full, densely packed `rows x cols` buffer.
    pub fn from_slice(data: &'a mut [T], rows: usize, cols: usize) -> Self {
        let ptr = NonNull::new(data.as_mut_ptr()).unwrap_or(NonNull::dangling());
        MatrixViewMut { ptr, rows, cols, stride: cols, _marker: PhantomData }
    }

    pub fn to_matrix(self) -> Matrix<T> where T: Copy {
        Matrix::from(
            self.rows,
            self.cols,
            [0..self.rows, 0..self.cols].into_iter().multi_cartesian_product().map(
                |v| self.get(v[0], v[1])
            ).collect()
        )
    }

    /// Mutable access to row `i` as a contiguous slice.
    #[inline]
    pub fn row_mut(&mut self, i: usize) -> &mut [T] {
        assert!(i < self.rows);
        // SAFETY: row `i` (i < self.rows) occupies `[ptr + i*stride, ptr + i*stride + cols)`,
        // which lies within the element set owned by this view, and `&mut self` here
        // ensures no other live borrow of this view's rows exists.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr().add(i * self.stride), self.cols) }
    }

    /// Read-only access to row `i`, useful when a kernel needs to both read
    /// and write within the same view (e.g. in-place algorithms).
    #[inline]
    pub fn row(&self, i: usize) -> &[T] {
        assert!(i < self.rows);
        // SAFETY: same reasoning as `row_mut`, but shared; `&self` here means
        // this doesn't race with a concurrent `row_mut` on the *same* index,
        // since that would require `&mut self` too.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr().add(i * self.stride), self.cols) }
    }

    /// Shared reborrow of the whole view, e.g. to pass into `mul_views` as an input.
    pub fn as_view(&self) -> MatrixView<'_, T> {
        MatrixView { ptr: self.ptr, rows: self.rows, cols: self.cols, stride: self.stride, _marker: PhantomData }
    }

    /// Reborrows with a shorter lifetime, so `self` can still be used afterward
    /// (e.g. to take a submatrix without consuming the original view).
    pub fn reborrow(&mut self) -> MatrixViewMut<'_, T> {
        MatrixViewMut { ptr: self.ptr, rows: self.rows, cols: self.cols, stride: self.stride, _marker: PhantomData }
    }

    /// A mutable sub-block.
    pub fn submatrix_mut(&mut self, r0: usize, c0: usize, rows: usize, cols: usize) -> MatrixViewMut<'a, T> {
        assert!(r0 + rows <= self.rows && c0 + cols <= self.cols);
        // SAFETY: (r0..r0+rows, c0..c0+cols) is contained in `self`'s element set.
        let ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(r0 * self.stride + c0)) };
        MatrixViewMut { ptr, rows, cols, stride: self.stride, _marker: PhantomData }
    }
    /// An immutable sub-block.
    pub fn submatrix_immut(&self, r0: usize, c0: usize, rows: usize, cols: usize) -> MatrixView<'a, T> {
        assert!(r0 + rows <= self.rows && c0 + cols <= self.cols);
        // SAFETY: (r0..r0+rows, c0..c0+cols) is contained in `self`'s element set.
        let ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(r0 * self.stride + c0)) };
        MatrixView { ptr, rows, cols, stride: self.stride, _marker: PhantomData }
    }

    /// Splits into disjoint top (`0..r`) and bottom (`r..rows`) row ranges.
    pub fn split_at_row_mut(self, r: usize) -> (MatrixViewMut<'a, T>, MatrixViewMut<'a, T>) {
        assert!(r <= self.rows);
        let bottom_ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(r * self.stride)) };
        (
            MatrixViewMut { ptr: self.ptr, rows: r, cols: self.cols, stride: self.stride, _marker: PhantomData },
            MatrixViewMut { ptr: bottom_ptr, rows: self.rows - r, cols: self.cols, stride: self.stride, _marker: PhantomData },
        )
    }

    /// Splits into disjoint left (`0..c`) and right (`c..cols`) column ranges.
    /// Both halves share `stride`, so row `i` of each still lands at the
    /// correct offset in the underlying buffer.
    pub fn split_at_col_mut(self, c: usize) -> (MatrixViewMut<'a, T>, MatrixViewMut<'a, T>) {
        assert!(c <= self.cols);
        let right_ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(c)) };
        (
            MatrixViewMut { ptr: self.ptr, rows: self.rows, cols: c, stride: self.stride, _marker: PhantomData },
            MatrixViewMut { ptr: right_ptr, rows: self.rows, cols: self.cols - c, stride: self.stride, _marker: PhantomData },
        )
    }

    /// Splits into four disjoint quadrants: [upper-left, upper-right, lower-left, lower-right].
    pub fn split_quadrants_mut(self, r: usize, c: usize) -> [MatrixViewMut<'a, T>; 4] {
        assert!(r <= self.rows && c <= self.cols);
        let (rows, cols, stride, ptr) = (self.rows, self.cols, self.stride, self.ptr);
        let mk = |dr: usize, dc: usize, rows: usize, cols: usize| MatrixViewMut {
            // SAFETY: (dr, dc) + (rows, cols) always stays within the original
            // (self.rows, self.cols) region for the four quadrant offsets used below.
            ptr: unsafe { NonNull::new_unchecked(ptr.as_ptr().add(dr * stride + dc)) },
            rows, cols, stride, _marker: PhantomData,
        };
        [
            mk(0, 0, r, c),
            mk(0, c, r, cols - c),
            mk(r, 0, rows - r, c),
            mk(r, c, rows - r, cols - c),
        ]
    }

    /// A parallel iterator over mutable rows, for use in place of
    /// `par_chunks_mut` when the view isn't backed by one contiguous slice.
    pub fn par_rows_mut(&mut self) -> impl rayon::iter::IndexedParallelIterator<Item = &mut [T]>
    where
        T: Send,
    {
        struct RowsMut<T> { ptr: *mut T, cols: usize, stride: usize, rows: usize }

        // SAFETY: distinct `i` values yield disjoint, non-overlapping row ranges
        // within the parent view's element set (the same invariant `row_mut`
        // relies on), so handing out one such slice per index across threads
        // never aliases. `T: Send` lets element values cross threads; there is
        // no shared mutable state being raced on.
        unsafe impl<T: Send> Send for RowsMut<T> {}
        unsafe impl<T: Send> Sync for RowsMut<T> {}

        let handle = RowsMut { ptr: self.ptr.as_ptr(), cols: self.cols, stride: self.stride, rows: self.rows };
        (0..handle.rows).into_par_iter().map(move |i| {
            let handle = &handle; // forces capturing `handle` as a unit, not its fields
            unsafe { std::slice::from_raw_parts_mut(handle.ptr.add(i * handle.stride), handle.cols) }
        })
    }
}

impl<'a, 'b, T> Add<MatrixView<'b, T>> for MatrixView<'a, T> where T: Copy + Add<T, Output=T> {
    type Output = Option<Matrix<T>>;
    fn add(self, rhs: MatrixView<'b, T>) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            None
        } else {
            Some(Matrix::from(
                self.rows,
                self.cols,
                [0..self.rows, 0..self.cols].into_iter().multi_cartesian_product().map(|v| self.get(v[0], v[1]).add(rhs.get(v[0], v[1]))).collect()
            ))
        }
    }
}
impl<'a, 'b, T> Sub<MatrixView<'b, T>> for MatrixView<'a, T> where T: Copy + Sub<T, Output=T> {
    type Output = Option<Matrix<T>>;
    fn sub(self, rhs: MatrixView<'b, T>) -> Self::Output {
        if self.rows != rhs.rows || self.cols != rhs.cols {
            None
        } else {
            Some(Matrix::from(
                self.rows,
                self.cols,
                [0..self.rows, 0..self.cols].into_iter().multi_cartesian_product().map(|v| self.get(v[0], v[1]).sub(rhs.get(v[0], v[1]))).collect()
            ))
        }
    }
}