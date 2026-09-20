use crate::math::{Complex, Expression};

mod impl_float;
mod impl_scalar;


// We need to implement the following traits ourselves to circumvent orphan rules.
// Otherwise, we'd have to treat e.g. `Complex<T> + T` and `T + Complex<T>` separately,
// which makes the code really messy.
pub trait Add<Rhs> {
    type Output;
    fn add(self, rhs: Rhs) -> Self::Output;
}
pub trait AddAssign<Rhs> {
    fn add_assign(&mut self, rhs: Rhs);
}
pub trait Sub<Rhs> {
    type Output;
    fn sub(self, rhs: Rhs) -> Self::Output;
}
pub trait SubAssign<Rhs> {
    fn sub_assign(&mut self, rhs: Rhs);
}
pub trait Mul<Rhs> {
    type Output;
    fn mul(self, rhs: Rhs) -> Self::Output;
}
pub trait MulAssign<Rhs> {
    fn mul_assign(&mut self, rhs: Rhs);
}
pub trait Div<Rhs> {
    type Output;
    fn div(self, rhs: Rhs) -> Self::Output;
}
pub trait DivAssign<Rhs> {
    fn div_assign(&mut self, rhs: Rhs);
}
pub trait Rem<Rhs> {
    type Output;
    fn rem(self, rhs: Rhs) -> Self::Output;
}
pub trait RemAssign<Rhs> {
    fn rem_assign(&mut self, rhs: Rhs);
}
pub trait Quo<Rhs> {
    type Output;
    fn quo(self, rhs: Rhs) -> Self::Output;
}
pub trait QuoAssign<Rhs> {
    fn quo_assign(&mut self, rhs: Rhs);
}
pub trait Pow<Rhs> {
    type Output;
    fn pow(self, rhs: Rhs) -> Self::Output;
}
pub trait PowAssign<Rhs> {
    fn pow_assign(&mut self, rhs: Rhs);
}
pub trait Neg {
    type Output;
    fn neg(self) -> Self::Output;
}


pub trait Real: Scalar<UnderlyingReal=Self> + PartialOrd {}

/// Currently, this corresponds to a float (`f32` or `f64`) or `Complex`.
/// 
/// I did not simply use `num_traits::Real` because there are several added functions and "Real" doesn't
/// really correspond to complex values.
/// 
/// There are some methods included that are specifially required for the UI, not mathematical computations (e.g. `format_trimmed`).
pub trait Scalar:
    Copy
    + Send + Sync // Required for parallelization
    + PartialEq
    + std::fmt::Debug
    + Add<Self, Output=Self>
    + Sub<Self, Output=Self>
    + Mul<Self, Output=Self>
    + Div<Self, Output=Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + MulAssign<Self>
    + DivAssign<Self>
    + Neg<Output=Self>
    + Mul<Self::UnderlyingReal, Output=Self>
    + Div<Self::UnderlyingReal, Output=Self>
    + Rem<Self::UnderlyingReal, Output=Self>
    + Quo<Self::UnderlyingReal, Output=Self>
    + Pow<Self::UnderlyingReal, Output=Self>
    + MulAssign<Self::UnderlyingReal>
    + DivAssign<Self::UnderlyingReal>
    + RemAssign<Self::UnderlyingReal>
    + QuoAssign<Self::UnderlyingReal>
    + PowAssign<Self::UnderlyingReal>
    + std::iter::Sum<Self>
{
    /// `f32` or `f64`. E.g. for `Complex<f32>`, this would be `f32`.
    type UnderlyingReal: Real;
    
    fn from_f64(x: f64) -> Self;
    fn from_i32(x: i32) -> Self;
    fn from_usize(x: usize) -> Self;

    fn to_complex(&self) -> Complex<Self::UnderlyingReal>;

    fn min_positive() -> Self;

    fn zero() -> Self;
    fn is_zero(&self) -> bool;
    fn one() -> Self;

    fn abs(&self) -> Self::UnderlyingReal;
    fn powi(&self, pow: i32) -> Self;

    /// Returns `self.abs() / self` (but generally, this can be computed in a more efficient way).
    fn abs_div_by_self(&self) -> Self;
    /// Returns `self / self.abs()`, except if `self` is zero; then, it returns zero.
    fn normalized(&self) -> Self {
        if *self == Self::zero() {
            Self::zero()
        } else {
            self.div(self.abs())
        }
    }

    fn round(&self) -> Self;
    fn ceil(&self) -> Self;
    fn floor(&self) -> Self;

    /// Returns the square root of `self` with non-negative real part.
    fn sqrt(&self) -> Self;
    // I only put the functions I actually use here.
    fn exp(&self) -> Self;
    fn ln(&self) -> Self;
    fn cos(&self) -> Self;
    fn sin(&self) -> Self;
    fn tan(&self) -> Self;
    fn cosh(&self) -> Self;
    fn sinh(&self) -> Self;
    fn tanh(&self) -> Self;
    fn acos(&self) -> Self;
    fn asin(&self) -> Self;
    fn atan(&self) -> Self;
    fn acosh(&self) -> Self;
    fn asinh(&self) -> Self;
    fn atanh(&self) -> Self;
    fn atan2(&self, other: Self) -> Self;

    /// Acts like `format!("{:.decimals}", x)` (exact formatting depending on the concrete type) but cuts off trailing zeros.
    fn format_trimmed(&self, decimals: usize) -> String;
    fn to_expression(&self) -> Expression;
}