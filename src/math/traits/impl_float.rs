use paste::paste;

use crate::math::utils::quof;
use super::*;


macro_rules! impl_float {
    ($op:ident, $f:expr, $($t:ty),*) => {
        $(
            paste!{
                impl $op<$t> for $t {
                    type Output = $t;
                    fn [<$op:lower>](self, rhs: $t) -> Self::Output {
                        $f(self, rhs)
                    }
                }
                impl [<$op Assign>]<$t> for $t {
                    fn [<$op:lower _assign>](&mut self, rhs: $t) {
                        *self = $f(*self, rhs);
                    }
                }
            }
        )*
    }
}
impl_float!(Add, (|x, y| x + y), f32, f64);
impl_float!(Sub, (|x, y| x - y), f32, f64);
impl_float!(Mul, (|x, y| x * y), f32, f64);
impl_float!(Div, (|x, y| <f32 as std::ops::Div>::div(x, y)), f32);
impl_float!(Div, (|x, y| <f64 as std::ops::Div>::div(x, y)), f64);
impl_float!(Rem, (|x: f32, y| x.rem_euclid(y)), f32);
impl_float!(Rem, (|x: f64, y| x.rem_euclid(y)), f64);
impl_float!(Quo, (|x, y| quof(x, y)), f32, f64);
impl_float!(Pow, (|x: f32, y| x.powf(y)), f32);
impl_float!(Pow, (|x: f64, y| x.powf(y)), f64);

// The implementations for `Mul`, `Div`, `Rem` and `Quo` between complex numbers are delegated to `complex.rs`.
// The same goes for the corresponding assignment operations.

macro_rules! impl_neg {
    ($($t:ty),*) => {
        $(
            impl Neg for $t {
                type Output = $t;
                fn neg(self) -> Self::Output {
                    -self
                }
            }
        )*
    };
}
impl_neg!(f32, f64);


impl Float for f32 {}
impl Float for f64 {}