use std::f64::consts;
use std::ops;

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Complex {
    pub real: f64,
    pub imag: f64
}

impl std::fmt::Display for Complex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} + {}*i", self.real, self.imag)
    }
}

impl std::default::Default for Complex {
    fn default() -> Self {
        Complex { real: 0.0, imag: 0.0 }
    }
}

impl ops::Neg for &Complex {
    type Output = Complex;
    fn neg(self) -> Self::Output {
        Complex { real: -self.real, imag: -self.imag }
    }
}
impl ops::Add<&Complex> for &Complex {
    type Output = Complex;
    fn add(self, rhs: &Complex) -> Self::Output {
        Complex { real: self.real + rhs.real, imag: self.imag + rhs.imag }
    }
}
impl ops::AddAssign<&Complex> for Complex {
    fn add_assign(&mut self, rhs: &Complex) {
        self.real += rhs.real;
        self.imag += rhs.imag;
    }
}
impl ops::Sub<&Complex> for &Complex {
    type Output = Complex;
    fn sub(self, rhs: &Complex) -> Self::Output {
        Complex { real: self.real - rhs.real, imag: self.imag - rhs.imag }
    }
}
impl ops::SubAssign<&Complex> for Complex {
    fn sub_assign(&mut self, rhs: &Complex) {
        self.real -= rhs.real;
        self.imag -= rhs.imag;
    }
}
impl ops::Mul<&Complex> for &Complex {
    type Output = Complex;
    fn mul(self, rhs: &Complex) -> Self::Output {
        Complex {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real
        }
    }
}
impl ops::MulAssign<&Complex> for Complex {
    fn mul_assign(&mut self, rhs: &Complex) {
        *self = Complex {
            real: self.real * rhs.real - self.imag * rhs.imag,
            imag: self.real * rhs.imag + self.imag * rhs.real
        };
    }
}
impl ops::Div<&Complex> for &Complex {
    type Output = Complex;
    #[allow(clippy::suspicious_arithmetic_impl)]
    fn div(self, rhs: &Complex) -> Self::Output {
        self * &rhs.inv()
    }
}
impl ops::DivAssign<&Complex> for Complex {
    #[allow(clippy::suspicious_op_assign_impl)]
    fn div_assign(&mut self, rhs: &Complex) {
        *self *= &rhs.inv();
    }
}

impl Complex {
    pub fn conjugate(&self) -> Complex {
        Complex { real: self.real, imag: -self.imag }
    }

    /// Computes the inverse of `self`.
    pub fn inv(&self) -> Complex {
        let x = self.real.powi(2) + self.imag.powi(2);
        Complex { real: self.real / x, imag: -self.imag / x }
    }

    pub fn modulus(&self) -> f64 {
        (self.real.powi(2) + self.imag.powi(2)).sqrt()
    }

    /// Computes the argument (in radian) of the given complex number `a + bi`.
    /// We use the convention `arg(z) ∈ (-π, π]`.
    pub fn arg(&self) -> f64 {
        if self.real == 0.0 {
            consts::PI / 2.0
        } else if self.real > 0.0 {
            (self.imag / self.real).atan()
        } else {
            if self.imag >= 0.0 {
                (self.imag / self.real).atan() + consts::PI
            } else {
                (self.imag / self.real).atan() - consts::PI
            }
        }
    }

    /// Computes `exp(a + bi)` using Euler's formula.
    pub fn exp(&self) -> Complex {
        let x = self.real.exp();
        Complex { real: x * self.imag.cos(), imag: x * self.imag.sin() }
    }

    /// Computes `self ^ exponent`.
    pub fn pow(&self, exponent: &Complex) -> Complex {
        // z^w = exp(w * ln(z)) = exp(w * (ln(|z|) + i * arg(z)));
        (exponent * &Complex { real: self.modulus().ln(), imag: self.arg() }).exp()
    }
}