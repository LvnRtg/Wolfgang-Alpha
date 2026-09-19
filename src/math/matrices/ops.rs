//! Implements basic arithmetic operations for matrices.

use itertools::Itertools;
use paste::paste;

use crate::math::{Complex, Vector};
use crate::math::traits::*;
use crate::math::utils;
use super::Matrix;


macro_rules! impl_componentwise_binop {
    ($op:ident) => {
        paste!{
            impl<T, U, V> $op<Matrix<U>> for Matrix::<T> where T: $op<U, Output=V> {
                type Output = Option<Matrix<V>>;
                fn [<$op:lower>](self, rhs: Matrix<U>) -> Self::Output {
                    if self.m != rhs.m || self.n != rhs.n {
                        None
                    } else {
                        Some(Matrix::<V> {
                            m: self.m,
                            n: self.n,
                            values: self.values.into_iter().zip(rhs.values.into_iter()).map(|(x, y)| x.[<$op:lower>](y)).collect::<Vec<V>>()
                        })
                    }
                }
            }
            impl<T, U, V> $op<&Matrix<U>> for Matrix::<T> where T: $op<U, Output=V>, U: Copy {
                type Output = Option<Matrix<V>>;
                fn [<$op:lower>](self, rhs: &Matrix<U>) -> Self::Output {
                    if self.m != rhs.m || self.n != rhs.n {
                        None
                    } else {
                        Some(Matrix::<V> {
                            m: self.m,
                            n: self.n,
                            values: self.values.into_iter().zip(rhs.values.iter()).map(|(x, &y)| x.[<$op:lower>](y)).collect::<Vec<V>>()
                        })
                    }
                }
            }
            impl<T, U, V> $op<Matrix<U>> for &Matrix::<T> where T: Copy + $op<U, Output=V> {
                type Output = Option<Matrix<V>>;
                fn [<$op:lower>](self, rhs: Matrix<U>) -> Self::Output {
                    if self.m != rhs.m || self.n != rhs.n {
                        None
                    } else {
                        Some(Matrix::<V> {
                            m: self.m,
                            n: self.n,
                            values: self.values.iter().zip(rhs.values.into_iter()).map(|(&x, y)| x.[<$op:lower>](y)).collect::<Vec<V>>()
                        })
                    }
                }
            }
            impl<T, U, V> $op<&Matrix<U>> for &Matrix::<T> where T: Copy + $op<U, Output=V>, U: Copy {
                type Output = Option<Matrix<V>>;
                fn [<$op:lower>](self, rhs: &Matrix<U>) -> Self::Output {
                    if self.m != rhs.m || self.n != rhs.n {
                        None
                    } else {
                        Some(Matrix::<V> {
                            m: self.m,
                            n: self.n,
                            values: self.values.iter().zip(rhs.values.iter()).map(|(&x, &y)| x.[<$op:lower>](y)).collect::<Vec<V>>()
                        })
                    }
                }
            }
            /// Produces garbage if the matrices dimensions do not match.
            impl<T, U> [<$op Assign>]<Matrix<U>> for Matrix::<T> where T: Copy + $op<U, Output=T>, U: Copy {
                fn [<$op:lower _assign>](&mut self, rhs: Matrix<U>) {
                    for i in 0..self.total_size().min(rhs.total_size()) {
                        self.values[i] = self.values[i].[<$op:lower>](rhs.values[i]);
                    }
                }
            }
            /// Produces garbage if the matrices dimensions do not match.
            impl<T, U> [<$op Assign>]<&Matrix<U>> for Matrix::<T> where T: Copy + $op<U, Output=T>, U: Copy {
                fn [<$op:lower _assign>](&mut self, rhs: &Matrix<U>) {
                    for i in 0..self.total_size().min(rhs.total_size()) {
                        self.values[i] = self.values[i].[<$op:lower>](rhs.values[i]);
                    }
                }
            }
        }
    }
}
impl_componentwise_binop!(Add);
impl_componentwise_binop!(Sub);

/// Where the scalar is on the RHS
macro_rules! impl_r_scalar_op {
    ($op:ident) => {
        paste!{
            impl<T, U, V> $op<U> for Matrix::<T> where T: Copy + $op<U, Output=V>, U: Scalar {
                type Output = Matrix<V>;
                fn [<$op:lower>](self, rhs: U) -> Self::Output {
                    Matrix {
                        m: self.m,
                        n: self.n,
                        values: self.values.into_iter().map(|x: T| x.[<$op:lower>](rhs)).collect::<Vec<V>>()
                    }
                }
            }
            impl<T, U, V> $op<U> for &Matrix::<T> where T: Copy + $op<U, Output=V>, U: Scalar {
                type Output = Matrix<V>;
                fn [<$op:lower>](self, rhs: U) -> Self::Output {
                    Matrix {
                        m: self.m,
                        n: self.n,
                        values: self.values.iter().map(|x: &T| (*x).[<$op:lower>](rhs)).collect::<Vec<V>>()
                    }
                }
            }
            impl<T, U> [<$op Assign>]<U> for Matrix::<T> where T: Copy + $op<U, Output=T>, U: Scalar {
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

/// Where the scalar is on the LHS
macro_rules! impl_l_scalar_op {
    ($op:ident, $($t:ty),*) => {
        $(
            paste!{
                impl<U, V> $op<Matrix<U>> for $t where $t: $op<U, Output=V> {
                    type Output = Matrix<V>;
                    fn [<$op:lower>](self, rhs: Matrix<U>) -> Self::Output {
                        Matrix {
                            m: rhs.m,
                            n: rhs.n,
                            values: rhs.values.into_iter().map(|x: U| self.[<$op:lower>](x)).collect::<Vec<V>>()
                        }
                    }
                }
                impl<'a, U, V> $op<&'a Matrix<U>> for $t where $t: $op<U, Output=V>, U: Copy {
                    type Output = Matrix<V>;
                    fn [<$op:lower>](self, rhs: &'a Matrix<U>) -> Self::Output {
                        Matrix {
                            m: rhs.m,
                            n: rhs.n,
                            values: rhs.values.iter().map(|x: &U| self.[<$op:lower>](*x)).collect::<Vec<V>>()
                        }
                    }
                }
            }
        )*
    }
}
impl_l_scalar_op!(Mul, f32, f64, Complex<f32>, Complex<f64>);
impl_l_scalar_op!(Rem, f32, f64, Complex<f32>, Complex<f64>);
impl_l_scalar_op!(Quo, f32, f64, Complex<f32>, Complex<f64>);

/// Division should behave differently: e.g. `1/A` should yield the inverse of `A` (if it exists)
macro_rules! impl_l_scalar_div {
    ($($t:ty),*) => {
        $(
            impl<U, V> Div<Matrix<U>> for $t where U: Scalar + Mul<$t, Output=V> {
                type Output = Option<Matrix<V>>;
                fn div(self, rhs: Matrix<U>) -> Self::Output {
                    rhs.inv().map(|inv: Matrix<U>| inv.mul(self))
                }
            }
            impl<U, V> Div<&Matrix<U>> for $t where U: Scalar + Mul<$t, Output=V> {
                type Output = Option<Matrix<V>>;
                fn div(self, rhs: &Matrix<U>) -> Self::Output {
                    rhs.inv().map(|inv| inv.mul(self))
                }
            }
        )*
    }
}
impl_l_scalar_div!(f32, f64, Complex<f32>, Complex<f64>);


impl<T> Neg for Matrix<T> where T: Copy + Neg<Output=T> {
    type Output = Matrix<T>;
    fn neg(self) -> Self::Output {
        Matrix {
            m: self.m,
            n: self.n,
            values: self.values.into_iter().map(|x| x.neg()).collect()
        }
    }
}
impl<T> Neg for &Matrix<T> where T: Copy + Neg<Output=T> {
    type Output = Matrix<T>;
    fn neg(self) -> Self::Output {
        Matrix {
            m: self.m,
            n: self.n,
            values: self.values.iter().map(|&x| x.neg()).collect()
        }
    }
}


/// Returns None in case the dimensions mismatch.
impl<T, U, V> Mul<&Vector<U>> for &Matrix<T>
where
    T: Copy + Mul<U, Output=V> + std::iter::Sum<T>,
    U: Copy,
    V: std::iter::Sum<V>
{
    type Output = Option<Vector<V>>;
    fn mul(self, rhs: &Vector<U>) -> Self::Output {
        if self.n != rhs.values.len() {
            None
        }
        else {
            Some(Vector{ values: (0..self.m).map(|i|
                utils::unchecked_dot(self.row_slice(i), &rhs.values)
            ).collect()})
        }
    }
}
impl<T, U, V> Mul<Vector<U>> for Matrix<T>
where
    T: Copy + Mul<U, Output=V> + std::iter::Sum<T>,
    U: Copy,
    V: std::iter::Sum<V>
{
    type Output = Option<Vector<V>>;
    fn mul(self, rhs: Vector<U>) -> Self::Output {
        (&self).mul(&rhs)
    }
}
impl<T, U, V> Mul<&Vector<U>> for Matrix<T>
where
    T: Copy + Mul<U, Output=V> + std::iter::Sum<T>,
    U: Copy,
    V: std::iter::Sum<V>
{
    type Output = Option<Vector<V>>;
    fn mul(self, rhs: &Vector<U>) -> Self::Output {
        (&self).mul(rhs)
    }
}
impl<T, U, V> Mul<Vector<U>> for &Matrix<T>
where
    T: Copy + Mul<U, Output=V> + std::iter::Sum<T>,
    U: Copy,
    V: std::iter::Sum<V>
{
    type Output = Option<Vector<V>>;
    fn mul(self, rhs: Vector<U>) -> Self::Output {
        self.mul(&rhs)
    }
}

/// This is mathematically not perfectly accurate, because one can only multiply a flipped vector with a matrix,
/// but this slight lack of rigorousness is less expensive than re-implementing all functions
/// for a new type 'FlippedVector' or using a 1xn-matrix.
/// 
/// Returns `None` in case the dimensions mismatch.
impl<T, U, V> Mul<&Matrix<U>> for &Vector<T>
where
    T: Copy + Mul<U, Output=V> + std::iter::Sum<T>,
    U: Copy,
    V: Scalar
{
    type Output = Option<Matrix<V>>;
    fn mul(self, rhs: &Matrix<U>) -> Self::Output {
        if rhs.m == 1 && rhs.n == self.len() {
            Some(Matrix::<V>::from(
                rhs.n,
                rhs.n,
                vec![0..rhs.n, 0..rhs.n].into_iter().multi_cartesian_product().map(|v| self[v[0]].mul(rhs.values[v[1]])).collect()
            ))
        } else if self.len() != rhs.m {
            None
        } else {
            let mut res = Matrix::<V>::zeros(rhs.n, 1);
            for k in 0..rhs.m { // Iterate in this order for better cache locality
                for i in 0..rhs.n {
                    res.values[i].add_assign(self.values[k].mul(rhs.get(k, i)));
                }
            }
            Some(res)
        }
    }
}
impl<T, U, V> Mul<Matrix<U>> for &Vector<T>
where
    T: Copy + Mul<U, Output=V> + std::iter::Sum<T>,
    U: Copy,
    V: Scalar
{
    type Output = Option<Matrix<V>>;
    fn mul(self, rhs: Matrix<U>) -> Self::Output {
        self.mul(&rhs)
    }
}
impl<T, U, V> Mul<&Matrix<U>> for Vector<T>
where
    T: Copy + Mul<U, Output=V> + std::iter::Sum<T>,
    U: Copy,
    V: Scalar
{
    type Output = Option<Matrix<V>>;
    fn mul(self, rhs: &Matrix<U>) -> Self::Output {
        (&self).mul(rhs)
    }
}
impl<T, U, V> Mul<Matrix<U>> for Vector<T>
where
    T: Copy + Mul<U, Output=V> + std::iter::Sum<T>,
    U: Copy,
    V: Scalar
{
    type Output = Option<Matrix<V>>;
    fn mul(self, rhs: Matrix<U>) -> Self::Output {
        (&self).mul(&rhs)
    }
}