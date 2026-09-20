use std::fmt::{Debug, Display};

use crate::math::Object;
use crate::math::traits::*;
use crate::math::utils::approx_eq;

mod ops;


#[derive(Copy, Clone, PartialEq)]
pub struct Complex<T: Real> {
    pub real: T,
    pub imag: T
}

impl<T: Real + Display> Display for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}*i", self.real, self.imag)
    }
}
impl<T: Real + Debug> Debug for Complex<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "({:?}) + ({:?})*i", self.real, self.imag)
    }
}

impl<T: Real> std::default::Default for Complex<T> {
    fn default() -> Self {
        Complex::<T>::zero()
    }
}

impl<T: Real> Complex<T> {
    pub fn i() -> Complex<T> {
        Complex { real: T::zero(), imag: T::one() }
    }

    pub fn conjugate(&self) -> Complex<T> {
        Complex { real: self.real, imag: self.imag.neg() }
    }

    /// Computes the inverse of `self`.
    pub fn inv(&self) -> Complex<T> {
        let x = self.real.powi(2).add(self.imag.powi(2));
        Complex { real: self.real.div(x), imag: self.imag.neg().div(x) }
    }

    pub fn modulus(&self) -> T {
        self.real.powi(2).add(self.imag.powi(2)).sqrt()
    }

    /// Computes the argument (in radian) of the given complex number `a + bi`.
    /// We use the convention `arg(z) ∈ (-π, π]`.
    pub fn arg(&self) -> T {
        self.imag.atan2(self.real)
    }

    /// Computes `exp(a + bi)` using Euler's formula.
    pub fn exp(&self) -> Complex<T> {
        let x = self.real.exp();
        Complex { real: x.mul(self.imag.cos()), imag: x.mul(self.imag.sin()) }
    }

    pub fn ln(&self) -> Complex<T> {
        Complex { real: self.modulus().ln(), imag: self.arg() }
    }

    pub fn cos(&self) -> Complex<T> {
        Complex { real: self.real.cos().mul(self.imag.cosh()), imag: self.real.sin().mul(self.imag.sinh()).neg() }
    }
    pub fn sin(&self) -> Complex<T> {
        Complex { real: self.real.sin().mul(self.imag.cosh()), imag: self.real.cos().mul(self.imag.sinh()) }
    }
    pub fn tan(&self) -> Complex<T> {
        let denom = self.real.mul(T::from_usize(2))
        .cos()
        .add(
            self.imag.mul(T::from_usize(2))
            .cosh()
        );
        Complex {
            real: self.real.mul(T::from_usize(2)).sin().div(denom),
            imag: self.imag.mul(T::from_usize(2)).sinh().div(denom)
        }
    }
    pub fn cosh(&self) -> Complex<T> {
        Complex { real: self.real.cosh().mul(self.imag.cos()), imag: self.real.sinh().mul(self.imag.sin()) }
    }
    pub fn sinh(&self) -> Complex<T> {
        Complex { real: self.real.sinh().mul(self.imag.cos()), imag: self.real.cosh().mul(self.imag.sin()) }
    }
    pub fn tanh(&self) -> Complex<T> {
        let denom = self.real.mul(T::from_usize(2))
        .cosh()
        .add(
            self.imag.mul(T::from_usize(2))
            .cos()
        );
        Complex {
            real: self.real.mul(T::from_usize(2)).sinh().div(denom),
            imag: self.imag.mul(T::from_usize(2)).sin().div(denom)
        }
    }
    pub fn acos(&self) -> Complex<T> {
        Complex::i()
        .neg()
        .mul(
            self
            .add(
                self.mul(*self)
                .sub(T::one())
                .sqrt()
            )
            .ln()
        )
    }
    pub fn asin(&self) -> Complex<T> {
        Complex::i()
        .neg()
        .mul(
            Complex::i().mul(*self)
            .add(
                Complex::one()
                .sub(self.mul(*self))
                .sqrt()
            )
            .ln()
        )
    }
    pub fn atan(&self) -> Complex<T> {
        Complex::one().add(
            Complex::i().mul(*self)
        )
        .div(
            Complex::one().sub(Complex::i().mul(*self))
        )
        .ln()
        .div(
            Complex::<T>::i().mul(T::from_usize(2))
        )
    }
    pub fn acosh(&self) -> Complex<T> {
        self
        .add(
            self.add(T::one()).sqrt()
            .mul(
                self.sub(T::one()).sqrt()
            )
        )
        .ln()
    }
    pub fn asinh(&self) -> Complex<T> {
        self
        .add(
            self
            .mul(*self)
            .add(T::one())
            .sqrt()
        )
        .ln()
    }
    pub fn atanh(&self) -> Complex<T> {
        Complex::one().add(*self)
        .div(Complex::one().sub(*self))
        .ln()
        .div(T::from_usize(2))
    }
    pub fn atan2(&self, x: Complex<T>) -> Complex<T> {
        Complex::<T>::i().inv().mul((x.add(Complex::<T>::i().mul(*self)).div(x.sub(Complex::<T>::i().mul(*self)))).ln().div(T::from_usize(2)))
    }

    /// Returns the square root of `self` with non-negative real part.
    pub fn sqrt(&self) -> Complex<T> {
        let z = self.modulus();
        // Formula: sqrt(a + ib) = ±(sqrt((z+a)/2) + i * sign(b) * sqrt((z-a)/2))
        Complex {
            real: z.add(self.real).div(T::from_usize(2)).sqrt(),
            imag: self.imag.normalized().mul(z.sub(self.real).div(T::from_usize(2)).sqrt())
        }
    }
}

impl Complex<f64> {
    pub fn to_obj(self) -> Object {
        if approx_eq(self.imag, 0.0) {
            Object::Real(self.real)
        } else {
            Object::Complex(self)
        }
    }
}