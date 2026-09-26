use paste::paste;

use crate::math::{Complex, Expression};
use super::*;

macro_rules! pull_float_functions_inside {
    ($t:ty, $($f:ident),*) => {
        $(
            fn $f(&self) -> Self {<$t>::$f(*self)}
        )*
    }
}
macro_rules! pull_complex_functions_inside {
    ($($f:ident),*) => {
        $(
            fn $f(&self) -> Self {Complex::$f(self)}
        )*
    }
}

/// Implements the trait `Scalar` for a type `$f` of the form `f32`, `f64`.
macro_rules! impl_scalar_for_float {
    ($f:ty) => { paste! {
        impl Scalar for $f {
            type UnderlyingReal = $f;

            fn from_f64(x: f64) -> Self { x as $f }
            fn from_i32(x: i32) -> Self { x as $f }
            fn from_usize(x: usize) -> Self { x as $f }

            fn to_complex(&self) -> Complex<Self::UnderlyingReal> {
                Complex{real: *self, imag: Self::zero()}
            }

            fn min_positive() -> Self {<$f>::MIN_POSITIVE}

            fn zero() -> Self {[<0_ $f>]}
            fn is_zero(&self) -> bool {*self == 0.0}
            fn one() -> Self {[<1_ $f>]}
            fn is_one(&self) -> bool {*self == 1.0}

            fn abs(&self) -> Self::UnderlyingReal {<$f>::abs(*self)}
            fn powi(&self, pow: i32) -> Self {<$f>::powi(*self, pow)}
            fn abs_div_by_self(&self) -> Self {<$f>::signum(*self)}

            pull_float_functions_inside!(
                $f,
                round, ceil, floor,
                sqrt, exp, ln,
                cos, sin, tan, acos, asin, atan,
                cosh, sinh, tanh, acosh, asinh, atanh
            );

            fn checked_sqrt(&self) -> Option<Self> {
                if *self >= <$f>::zero() {
                    Some(<$f>::sqrt(*self))
                } else {
                    None
                }
            }

            fn atan2(&self, other: Self) -> Self {<$f>::atan2(*self, other)}

            fn format_trimmed(&self, decimals: usize) -> String {
                let s = format!("{:.prec$}", self, prec = decimals);
                let s = s.trim_end_matches('0');
                let s = s.trim_end_matches('.');
                s.to_string()
            }
            fn to_expression(&self) -> Expression {
                Expression::Number(*self as f64)
            }
        }
    }}
}
impl_scalar_for_float!(f32);
impl_scalar_for_float!(f64);

// Note: below implementation technically allows for `Complex<Complex<f64>>`, which of course is useless
// since it should be just `Complex<f64>` instead. However, keep in mind that as these are just traits,
// one could always implement `Complex<U>` for a garbage type `U`; `Complex<f64>` is just one such `U`.
impl<T: Real> Scalar for Complex<T> {
    type UnderlyingReal = T;

    fn from_f64(x: f64) -> Self {
        Complex::<T>{real: T::from_f64(x), imag: T::zero()}
    }
    fn from_i32(x: i32) -> Self {
        Complex::<T>{real: T::from_i32(x), imag: T::zero()}
    }
    fn from_usize(x: usize) -> Self {
        Complex::<T>{real: T::from_usize(x), imag: T::zero()}
    }

    fn to_complex(&self) -> Complex<Self::UnderlyingReal> {*self}

    fn min_positive() -> Self {
        Complex { real: T::min_positive(), imag: T::min_positive() }
    }

    fn zero() -> Self {Complex::<T>{real: T::zero(), imag: T::zero()}}
    fn is_zero(&self) -> bool {self.real.is_zero() && self.imag.is_zero()}
    fn one() -> Self {Complex::<T>{real: T::one(), imag: T::zero()}}
    fn is_one(&self) -> bool {self.real.is_one() && self.imag.is_zero()}

    fn abs(&self) -> Self::UnderlyingReal {self.modulus()}
    fn powi(&self, pow: i32) -> Self {self.pow(Complex::from_i32(pow))}
    fn abs_div_by_self(&self) -> Self {
        self.conjugate().div(self.abs())
    }

    fn round(&self) -> Self {
        Complex { real: self.real.round(), imag: self.imag.round() }
    }
    fn ceil(&self) -> Self {
        Complex { real: self.real.ceil(), imag: self.imag.ceil() }
    }
    fn floor(&self) -> Self {
        Complex { real: self.real.floor(), imag: self.imag.floor() }
    }
    
    pull_complex_functions_inside!(
        sqrt, exp, ln,
        cos, sin, tan, acos, asin, atan,
        cosh, sinh, tanh, acosh, asinh, atanh
    );

    fn checked_sqrt(&self) -> Option<Self> {
        Some(Complex::sqrt(self))
    }
    
    fn atan2(&self, other: Self) -> Self {Complex::atan2(self, other)}

    fn format_trimmed(&self, decimals: usize) -> String {
        format!("{} + {}*i", self.real.format_trimmed(decimals), self.imag.format_trimmed(decimals))
    }
    fn to_expression(&self) -> Expression {
        crate::expr_binop!(self.real.to_expression(), Add, crate::expr_binop!(self.imag.to_expression(), Mul, Expression::Identifier("i".to_string())))
    }
}