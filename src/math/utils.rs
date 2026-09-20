//! Most functions in this module have for sole objective to simplify typing and enhance readability.

use num_traits::{Euclid, Float, int::PrimInt, NumCast};

use crate::math::objects::{Object, try_operation};
use crate::math::operations::BinaryOperation;
use crate::math::traits::*;
use crate::status::{ExtResult, Status};


const ABS_TOL: f64 = 1e-12;
const REL_TOL: f64 = 1e-10;


pub fn quoi<T: Euclid + PrimInt>(x: T, y: T) -> T {
    (x - (x.rem_euclid(&y))) / y
}
pub fn quof<T: Euclid + Float>(x: T, y: T) -> T {
    ((x - (x.rem_euclid(&y))) / y).round()
}

#[inline]
pub fn approx_eq<T: Scalar>(x: T, y: T) -> bool {
    // We use the criterion |x-y| <= max(ABS_TOL, REL_TOL * max(|x|, |y|))
    (x.sub(y)).abs() <= max(<T as Scalar>::UnderlyingReal::from_f64(ABS_TOL), <T as Scalar>::UnderlyingReal::from_f64(REL_TOL).mul(max(x.abs(), y.abs())))
}

pub fn max<T: PartialOrd>(x: T, y: T) -> T {
    if x >= y {x} else {y}
}
pub fn min<T: PartialOrd>(x: T, y: T) -> T {
    if x <= y {x} else {y}
}

#[inline]
/// Returns the maximum of the given iterator. If the iterator is empty, returns `None`.
pub fn max_of<T: PartialOrd>(iter: impl Iterator<Item = T>) -> Option<T> {
    iter.fold(None, |acc, x| match acc {
        None => Some(x),
        Some(m) => Some(if x > m { x } else { m }),
    })
}
#[inline]
/// Returns the minimum of the given iterator. If the iterator is empty, returns `None`.
pub fn min_of<T: PartialOrd>(iter: impl Iterator<Item = T>) -> Option<T> {
    iter.fold(None, |acc, x| match acc {
        None => Some(x),
        Some(m) => Some(if m <= x { m } else { x }),
    })
}

#[inline]
/// Returns the maximum absolute value of the given iterator of floats. If the iterator is empty, returns 0.0.
pub fn max_abs_of<'a, T: 'a + Scalar>(iter: impl Iterator<Item=&'a T>) -> T::UnderlyingReal {
    iter.fold(T::UnderlyingReal::zero(), |acc, x| {
        max(acc, x.abs())
    })
}

/// Returns the `i`-th row of `v` as slice where `n` is the length of a row.
/// 
/// The returned slice therefore has length `n`.
#[inline]
pub fn row<T>(v: &[T], i: usize, n: usize) -> &[T] {
    &v[i * n .. (i+1) * n]
}
/// Returns the `j`-th column of `v` as iterator where `m` is the number of rows to take
/// and `n` is the length of each row.
/// 
/// The returned iterator therefore iterates over `m` elements.
#[inline]
pub fn col<T: Copy>(v: &[T], j: usize, m: usize, n: usize) -> std::iter::Map<std::ops::Range<usize>, impl FnMut(usize) -> T> {
    (0..m).map(move |i| v[i * n + j])
}

/// Splits the interval `[a, b]` into `n` uniformly spread points, the first of which equals `a` and the last of which equals `b`.
pub fn linspace_as_objects(a: f64, b: f64, n: usize) -> Vec<Object> {
    if n == 0 {return Vec::<Object>::new();}
    if n == 1 {return vec![Object::Real(a)];}
    let step = (b-a) / ((n-1) as f64);
    (0..n).map(|i| Object::Real(a + i as f64 * step)).collect()
}


/// Folds all elements in the iterator, short-circuiting if an `Err` is found. Returns `None` iff the iterator is empty.
pub fn fold_res_obj_iter(mut iter: impl Iterator<Item=Result<Object, String>>, binop: &BinaryOperation) -> Option<ExtResult> {
    let first = iter.next()?;
    Some(iter.fold(
        first.map(Status::ok),
        |acc, new| acc.and_then(
            |lhs_s| new.and_then(
                |rhs|
                // "safe" to pass `None` because this method never gets called on `binop = Comparison(..)`
                // and iterating a comparison in this way doesn't make sense anyway.
                lhs_s.try_map_flatten(|lhs| try_operation(&lhs, &rhs, binop, None))
            )
        )
    ))
}

pub fn expect_int<T: NumCast + Copy>(f: f64) -> Option<T> {
    let i = f.round();
    if approx_eq(f, i) {
        T::from(i)
    } else {
        None
    }
}

/// Returns `Some(x)` if `it` returns exactly one element `x`, otherwise `None`.
pub fn expect_exactly_one<T>(mut it: impl Iterator<Item=T>) -> Option<T> {
    if let Some(x) = it.next() {
        if it.next().is_none() {
            Some(x)
        } else {
            None
        }
    } else {
        None
    }
}
/// Returns `Ok(Some(x))` if `it` returns exactly one element `Ok(x)`, otherwise `None`. Short-circuits on `Err`.
pub fn try_expect_exactly_one<T, E>(mut it: impl Iterator<Item=Result<T, E>>) -> Result<Option<T>, E> {
    if let Some(r) = it.next() {
        let t = r?;
        if it.next().is_none() {
            Ok(Some(t))
        } else {
            Ok(None)
        }
    } else {
        Ok(None)
    }
}

pub fn all_res<T, E, F: FnMut(T) -> Result<bool, E>>(it: impl Iterator<Item=T>, mut f: F) -> Result<bool, E> {
    for t in it {
        if !f(t)? {
            return Ok(false);
        }
    }
    Ok(true)
}
pub fn any_res<T, E, F: FnMut(T) -> Result<bool, E>>(it: impl Iterator<Item=T>, mut f: F) -> Result<bool, E> {
    for t in it {
        if f(t)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Contiguous slice dot product. Does not check if the dimensions of `a, b` match.
#[inline]
pub fn unchecked_dot<T, U, V>(a: &[T], b: &[U]) -> V
where
    T: Copy + Mul<U, Output=V>,
    U: Copy,
    V: std::iter::Sum<V>
{
    a.iter().zip(b.iter()).map(|(&x, &y)| x.mul(y)).sum()
}
/// Contiguous slice dot product. Does not check if the dimensions of `a, b` match.
#[inline]
pub fn unchecked_dot_iter<T, U, V>(a: std::iter::Map<std::ops::Range<usize>, impl FnMut(usize) -> T>, b: &[U]) -> V
where
    T: Mul<U, Output=V>,
    U: Copy,
    V: std::iter::Sum<V>
{
    a.zip(b.iter()).map(|(x, &y)| x.mul(y)).sum()
}