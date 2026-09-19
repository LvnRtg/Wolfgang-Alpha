use paste::paste;

use crate::math::traits::*;
use crate::math::utils::approx_eq;
use super::Complex;


macro_rules! impl_add_sub_trait {
    ($op:ident) => {
        paste!{
            impl<T: Float> $op<T> for Complex<T> {
                type Output = Complex<T>;
                fn [<$op:lower>](self, rhs: T) -> Self::Output {
                    Complex::<T> {real: self.real.[<$op:lower>](rhs), imag: self.imag }
                }
            }
            impl<T: Float> $op<Complex<T>> for T {
                type Output = Complex<T>;
                fn [<$op:lower>](self, rhs: Complex<T>) -> Self::Output {
                    Complex::<T> {real: self.[<$op:lower>](rhs.real), imag: rhs.imag }
                }
            }
            impl<T: Float> $op<Complex<T>> for Complex<T> {
                type Output = Complex<T>;
                fn [<$op:lower>](self, rhs: Complex<T>) -> Self::Output {
                    Complex::<T> {real: self.real.[<$op:lower>](rhs.real), imag: self.imag.[<$op:lower>](rhs.imag) }
                }
            }
            impl<T: Float> [<$op Assign>]<T> for Complex<T> {
                fn [<$op:lower _assign>](&mut self, rhs: T) {
                    self.real = self.real.[<$op:lower>](rhs);
                }
            }
            impl<T: Float> [<$op Assign>]<Complex<T>> for Complex<T> {
                fn [<$op:lower _assign>](&mut self, rhs: Complex<T>) {
                    self.real = self.real.[<$op:lower>](rhs.real);
                    self.imag = self.imag.[<$op:lower>](rhs.imag);
                }
            }
        }
    }
}
impl_add_sub_trait!(Add);
impl_add_sub_trait!(Sub);

macro_rules! impl_componentwise_op_trait {
    ($op:ident) => {
        paste!{
            impl<T: Float> $op<T> for Complex<T> {
                type Output = Complex<T>;
                fn [<$op:lower>](self, rhs: T) -> Self::Output {
                    Complex::<T> {real: self.real.[<$op:lower>](rhs), imag: self.imag.[<$op:lower>](rhs) }
                }
            }
            impl<T: Float> $op<Complex<T>> for T {
                type Output = Complex<T>;
                fn [<$op:lower>](self, rhs: Complex<T>) -> Self::Output {
                    Complex::<T> {real: self.[<$op:lower>](rhs.real), imag: self.[<$op:lower>](rhs.imag) }
                }
            }
            impl<T: Float> [<$op Assign>]<T> for Complex<T> {
                fn [<$op:lower _assign>](&mut self, rhs: T) {
                    self.real = self.real.[<$op:lower>](rhs);
                    self.imag = self.imag.[<$op:lower>](rhs);
                }
            }
        }
    }
}
impl_componentwise_op_trait!(Mul);
impl_componentwise_op_trait!(Div);
impl_componentwise_op_trait!(Rem);
impl_componentwise_op_trait!(Quo);

impl<T: Float> Neg for Complex<T> {
    type Output = Complex<T>;
    fn neg(self) -> Self::Output {
        Complex { real: self.real.neg(), imag: self.imag.neg() }
    }
}


impl<T: Float> Mul<Complex<T>> for Complex<T> {
    type Output = Complex<T>;
    fn mul(self, rhs: Complex<T>) -> Self::Output {
        Complex {
            real: self.real.mul(rhs.real).sub(self.imag.mul(rhs.imag)),
            imag: self.real.mul(rhs.imag).add(self.imag.mul(rhs.real))
        }
    }
}
impl<T: Float> MulAssign<Complex<T>> for Complex<T> {
    fn mul_assign(&mut self, rhs: Complex<T>) {
        *self = Complex {
            real: self.real.mul(rhs.real).sub(self.imag.mul(rhs.imag)),
            imag: self.real.mul(rhs.imag).add(self.imag.mul(rhs.real))
        };
    }
}
impl<T: Float> Div<Complex<T>> for Complex<T> {
    type Output = Complex<T>;
    fn div(self, rhs: Complex<T>) -> Self::Output {
        self.mul(rhs.inv())
    }
}
impl<T: Float> DivAssign<Complex<T>> for Complex<T> {
    fn div_assign(&mut self, rhs: Complex<T>) {
        *self = self.mul(rhs.inv());
    }
}
// To the best of my knowledge, there is no canonical understanding for integer division among complex numbers.
// I therefore chose to simply do it componentwise.
impl<T: Float> Rem<Complex<T>> for Complex<T> {
    type Output = Complex<T>;
    fn rem(self, rhs: Complex<T>) -> Self::Output {
        Complex {real: self.real.rem(rhs.real), imag: self.imag.rem(rhs.imag)}
    }
}
impl<T: Float> RemAssign<Complex<T>> for Complex<T> {
    fn rem_assign(&mut self, rhs: Complex<T>) {
        *self = self.rem(rhs);
    }
}
impl<T: Float> Quo<Complex<T>> for Complex<T> {
    type Output = Complex<T>;
    fn quo(self, rhs: Complex<T>) -> Self::Output {
        Complex {real: self.real.quo(rhs.real), imag: self.imag.quo(rhs.imag)}
    }
}
impl<T: Float> QuoAssign<Complex<T>> for Complex<T> {
    fn quo_assign(&mut self, rhs: Complex<T>) {
        *self = self.quo(rhs);
    }
}

impl<T: Float> Pow<T> for Complex<T> {
    type Output = Complex<T>;
    fn pow(self, rhs: T) -> Self::Output {
        (Complex { real: self.modulus().ln(), imag: self.arg() }.mul(rhs)).exp()
    }
}
impl<T: Float> Pow<Complex<T>> for T {
    type Output = Complex<T>;
    fn pow(self, rhs: Complex<T>) -> Self::Output {
        (rhs.mul(self.abs().ln())).exp()
    }
}
impl<T: Float> Pow<Complex<T>> for Complex<T> {
    type Output = Complex<T>;
    fn pow(self, rhs: Complex<T>) -> Self::Output {
        // z^w = exp(w * ln(z)) = exp(w * (ln(|z|) + i * arg(z)));
        let m = self.modulus();
        if approx_eq(m, T::zero()) {
            if approx_eq(rhs, Complex::zero()) {
                Complex::one()
            } else {
                Complex::zero()
            }
        } else {
            rhs.mul(Complex { real: m.ln(), imag: self.arg() }).exp()
        }
    }
}
impl<T: Float> PowAssign<T> for Complex<T> {
    fn pow_assign(&mut self, rhs: T) {
        *self = (Complex { real: self.modulus().ln(), imag: self.arg() }.mul(rhs)).exp()
    }
}
impl<T: Float> PowAssign<Complex<T>> for Complex<T> {
    fn pow_assign(&mut self, rhs: Complex<T>) {
        *self = (rhs.mul(Complex { real: self.modulus().ln(), imag: self.arg() })).exp()
    }
}

impl<T: Float> std::iter::Sum<Complex<T>> for Complex<T> {
    fn sum<I: Iterator<Item = Complex<T>>>(iter: I) -> Self {
        iter.fold(<Complex::<T> as Scalar>::zero(), |acc, x| acc.add(x))
    }
}