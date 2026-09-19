//! Implements basic arithmetic operations for vectors.
//! 
//! Note: unused implementations will be removed by the compiler, so this is zero-cost at runtime.

use paste::paste;
use std::ops::{Index, IndexMut};
use std::slice::SliceIndex;

use crate::math::Complex;
use crate::math::traits::*;
use super::Vector;

// Indexing just operates on the values directly
impl<I, T> Index<I> for Vector<T> where I: SliceIndex<[T]> {
    type Output = I::Output;
    fn index(&self, index: I) -> &Self::Output {
        &self.values[index]
    }
}
impl<I, T> IndexMut<I> for Vector<T> where I: SliceIndex<[T]> {
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.values[index]
    }
}


/// Multiplication of vectors is treated as the euclidian inner product (i.e. v * v := v^t * v).
/// This makes it more convenient to obtain the inner product.
/// 
/// Returns None in case the dimensions mismatch.
impl<T, U, V> Mul<Vector<U>> for Vector<T>
where
    T: Copy + Mul<U, Output=V>,
    U: Copy,
    V: std::iter::Sum<V>
{
    type Output = Option<V>;
    fn mul(self, rhs: Vector<U>) -> Self::Output {
        if self.values.len() != rhs.values.len() {
            None
        } else {
            Some(self.values.into_iter().zip(rhs.values.into_iter()).map(|(x, y)| x.mul(y)).sum())
        }
    }
}
impl<T, U, V> Mul<&Vector<U>> for Vector<T>
where
    T: Copy + Mul<U, Output=V>,
    U: Copy,
    V: std::iter::Sum<V>
{
    type Output = Option<V>;
    fn mul(self, rhs: &Vector<U>) -> Self::Output {
        if self.values.len() != rhs.values.len() {
            None
        } else {
            Some(self.values.into_iter().zip(rhs.iter()).map(|(x, &y)| x.mul(y)).sum())
        }
    }
}
impl<T, U, V> Mul<Vector<U>> for &Vector<T>
where
    T: Copy + Mul<U, Output=V>,
    U: Copy,
    V: std::iter::Sum<V>
{
    type Output = Option<V>;
    fn mul(self, rhs: Vector<U>) -> Self::Output {
        if self.values.len() != rhs.values.len() {
            None
        } else {
            Some(self.values.iter().zip(rhs.values.into_iter()).map(|(&x, y)| x.mul(y)).sum())
        }
    }
}
impl<T, U, V> Mul<&Vector<U>> for &Vector<T>
where
    T: Copy + Mul<U, Output=V>,
    U: Copy,
    V: std::iter::Sum<V>
{
    type Output = Option<V>;
    fn mul(self, rhs: &Vector<U>) -> Self::Output {
        if self.values.len() != rhs.values.len() {
            None
        } else {
            Some(self.values.iter().zip(rhs.iter()).map(|(&x, &y)| x.mul(y)).sum())
        }
    }
}


/// Implements the given binary operation for vectors of the same size, where the operation is to be performed component-wise.
macro_rules! impl_componentwise_binop {
    ($op:ident) => {
        paste!{
            impl<T, U, V> $op<Vector<U>> for Vector::<T> where T: $op<U, Output=V> {
                type Output = Option<Vector<V>>;
                fn [<$op:lower>](self, rhs: Vector<U>) -> Self::Output {
                    if self.len() != rhs.len() {
                        None
                    } else {
                        Some(Vector::<V> {
                            values: self.values.into_iter().zip(rhs.values.into_iter()).map(|(x, y)| x.[<$op:lower>](y)).collect::<Vec<V>>()
                        })
                    }
                }
            }
            impl<T, U, V> $op<&Vector<U>> for Vector::<T> where T: $op<U, Output=V>, U: Copy {
                type Output = Option<Vector<V>>;
                fn [<$op:lower>](self, rhs: &Vector<U>) -> Self::Output {
                    if self.len() != rhs.len() {
                        None
                    } else {
                        Some(Vector::<V> {
                            values: self.values.into_iter().zip(rhs.values.iter()).map(|(x, &y)| x.[<$op:lower>](y)).collect::<Vec<V>>()
                        })
                    }
                }
            }
            impl<T, U, V> $op<Vector<U>> for &Vector::<T> where T: Copy + $op<U, Output=V> {
                type Output = Option<Vector<V>>;
                fn [<$op:lower>](self, rhs: Vector<U>) -> Self::Output {
                    if self.len() != rhs.len() {
                        None
                    } else {
                        Some(Vector::<V> {
                            values: self.values.iter().zip(rhs.values.into_iter()).map(|(&x, y)| x.[<$op:lower>](y)).collect::<Vec<V>>()
                        })
                    }
                }
            }
            impl<T, U, V> $op<&Vector<U>> for &Vector::<T> where T: Copy + $op<U, Output=V>, U: Copy {
                type Output = Option<Vector<V>>;
                fn [<$op:lower>](self, rhs: &Vector<U>) -> Self::Output {
                    if self.len() != rhs.len() {
                        None
                    } else {
                        Some(Vector::<V> {
                            values: self.values.iter().zip(rhs.values.iter()).map(|(&x, &y)| x.[<$op:lower>](y)).collect::<Vec<V>>()
                        })
                    }
                }
            }
            /// Behavior:
            /// - If the RHS is shorter than the LHS, treat it as if it were extended by zeros.
            /// - If the RHS is longer than the LHS, ignore the trailing values.
            impl<T, U> [<$op Assign>]<Vector<U>> for Vector::<T> where T: Copy + $op<U, Output=T>, U: Copy {
                fn [<$op:lower _assign>](&mut self, rhs: Vector<U>) {
                    for i in 0..self.len().min(rhs.len()) {
                        self.values[i] = self.values[i].[<$op:lower>](rhs.values[i]);
                    }
                }
            }
            impl<T, U> [<$op Assign>]<&Vector<U>> for Vector::<T> where T: Copy + $op<U, Output=T>, U: Copy {
                fn [<$op:lower _assign>](&mut self, rhs: &Vector<U>) {
                    for i in 0..self.len().min(rhs.len()) {
                        self.values[i] = self.values[i].[<$op:lower>](rhs.values[i]);
                    }
                }
            }
        }
    }
}
impl_componentwise_binop!(Add);
impl_componentwise_binop!(Sub);


impl<T> Neg for Vector::<T> where T: Neg<Output=T> {
    type Output = Vector<T>;
    fn neg(self) -> Self::Output {
        Vector::<T> {
            values: self.values.into_iter().map(|x| x.neg()).collect::<Vec<T>>()
        }
    }
}
impl<T> Neg for &Vector::<T> where T: Copy + Neg<Output=T> {
    type Output = Vector<T>;
    fn neg(self) -> Self::Output {
        Vector::<T> {
            values: self.values.iter().map(|&x| x.neg()).collect::<Vec<T>>()
        }
    }
}

/// Implements `Vector<T> $op T` component-wise for all types for which this is a valid operation.
/// The scalar type `U` is restricted to be scalar because a general implementation would conflict with multiplication between vectors.
macro_rules! impl_r_scalar_op {
    ($op:ident) => {
        paste!{
            // When scalar is on the right
            impl<T, U, V> $op<U> for Vector::<T> where T: $op<U, Output=V>, U: Scalar {
                type Output = Vector<V>;
                fn [<$op:lower>](self, rhs: U) -> Self::Output {
                    Vector {
                        values: self.values.into_iter().map(|x: T| x.[<$op:lower>](rhs)).collect::<Vec<V>>()
                    }
                }
            }
            impl<T, U, V> $op<U> for &Vector::<T> where T: Copy + $op<U, Output=V>, U: Scalar {
                type Output = Vector<V>;
                fn [<$op:lower>](self, rhs: U) -> Self::Output {
                    Vector {
                        values: self.values.iter().map(|x: &T| (*x).[<$op:lower>](rhs)).collect::<Vec<V>>()
                    }
                }
            }
            impl<T, U> [<$op Assign>]<U> for Vector::<T> where T: Copy + $op<U, Output=T>, U: Scalar {
                fn [<$op:lower _assign>](&mut self, rhs: U) {
                    for x in self.values.iter_mut() {
                        *x = (*x).[<$op:lower>](rhs);
                    }
                }
            }
        }
    }
}
impl_r_scalar_op!(Mul);
impl_r_scalar_op!(Div);
impl_r_scalar_op!(Rem);
impl_r_scalar_op!(Quo);

/// Implements `$t $op Vector<U>` for all given types `$t` and all valid `U`.
/// 
/// We need to implement these for concrete types only because of an infinite recursion problem that would arise otherwise:
/// On stable Rust, there is no way to specify that a blanket type won't implement some trait even if we know this won't be
/// the case here.
macro_rules! impl_l_scalar_op {
    ($op:ident, $($t:ty),*) => {
        $(
            paste!{
                impl<U, V> $op<Vector<U>> for $t where $t: $op<U, Output=V> {
                    type Output = Vector<V>;
                    fn [<$op:lower>](self, rhs: Vector<U>) -> Self::Output {
                        Vector {
                            values: rhs.values.into_iter().map(|x: U| self.[<$op:lower>](x)).collect::<Vec<V>>()
                        }
                    }
                }
                impl<'a, U, V> $op<&'a Vector<U>> for $t where $t: $op<U, Output=V>, U: Copy {
                    type Output = Vector<V>;
                    fn [<$op:lower>](self, rhs: &'a Vector<U>) -> Self::Output {
                        Vector {
                            values: rhs.values.iter().map(|x: &U| self.[<$op:lower>](*x)).collect::<Vec<V>>()
                        }
                    }
                }
            }
        )*
    }
}
impl_l_scalar_op!(Mul, f32, f64, Complex<f32>, Complex<f64>);
// There is no canonical inverse to a vector, but defining `x / v := (x/v_1, ..., x/v_n)` makes somewhat sense
// because we already interpret `v*w` as the inner product, and with the above definition, we have
// `(x/v) * (v/x) = sum_{i=1}^n (x/v_i) * (v_i/x) = n`, which is consistent with the fact that
// `(1, ..., 1) * (1, ..., 1) = n`.
impl_l_scalar_op!(Div, f32, f64, Complex<f32>, Complex<f64>);
impl_l_scalar_op!(Rem, f32, f64, Complex<f32>, Complex<f64>);
impl_l_scalar_op!(Quo, f32, f64, Complex<f32>, Complex<f64>);