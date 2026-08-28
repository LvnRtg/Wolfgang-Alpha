//! Implements the struct `Status`.

use std::ops;

use crate::math::Object;

/// Contains a value of type `T` and a (potentially empty) list of warnings.
#[derive(Clone)]
pub struct Status<T> {
    pub value: T,
    pub warnings: Vec<String>
}

pub type ExtResult = Result<Status<Object>, String>;

impl<T> Status<T> {
    /// Returns a `Status` with the given value and no warnings.
    #[inline]
    pub fn ok(value: T) -> Status<T> {
        Status { value, warnings: Vec::<String>::new() }
    }

    /// Applies `f` to each element of `iter`, yielding a `Result<Status<T>, String>` each time. Short-circuits when encountering an `Err`,
    /// otherwise combines the values and warnings to obtain a final `Ok(Status<Vec<T>>)`.
    pub fn from_iter<U, F: FnMut(U) -> Result<Status<T>, String>>(iter: impl Iterator<Item=U>, mut f: F) -> Result<Status<Vec<T>>, String> {
        iter.fold(
            Ok(Status::ok(Vec::<T>::new())),
            |acc, u| {
                match (acc, f(u)) {
                    (Ok(mut acc_ok), Ok(new_ok)) => {
                        acc_ok.push(new_ok);
                        Ok(acc_ok)
                    }
                    (Err(e), _) | (_, Err(e)) => Err(e)
                }
            }
        )
    }

    /// Applies `f` to the contained value and leaves all warnings unchanged.
    pub fn map<U, F: FnOnce(T) -> U>(self, f: F) -> Status<U> {
        Status { value: f(self.value), warnings: self.warnings }
    }
    /// Applies `f` to the contained value. If `f` returns `Ok(x)`, this function returns `Ok(x, self.warnings)`;
    /// otherwise, this function returns `Err` and throws away all warnings.
    pub fn try_map<U, E, F: FnOnce(T) -> Result<U, E>>(self, f: F) -> Result<Status<U>, E> {
        Ok(Status { value: f(self.value)?, warnings: self.warnings })
    }
    /// Acts like `try_map` but flattens the result before returning it.
    pub fn try_map_flatten<U, E, F: FnOnce(T) -> Result<Status<U>, E>>(self, f: F) -> Result<Status<U>, E> {
        let mut warnings = self.warnings;
        Ok(Status { value: f(self.value)?.unpack_into(&mut warnings), warnings })
    }
    /// If `f(self.value) = Ok(u)`, returns `Ok(u)` with the warnings of `self` transferred. Otherwise,
    /// redirects the `Err`.
    pub fn and_then<U, E, F: FnOnce(T) -> Result<U, E>>(self, f: F) -> Result<Status<U>, E> {
        f(self.value).map(|u| Status{value: u, warnings: self.warnings})
    }

    /// Appends `self.warnings` to `warnings` and returns `self.value`, leaving `self` empty.
    pub fn unpack_into(self, warnings: &mut Vec<String>) -> T {
        warnings.extend(self.warnings.into_iter());
        self.value
    }
    /// Appends `self.warnings` to `warnings` and returns `self.value`, leaving `self` empty.
    /// Only allows `warnings` to contain at most `cap` warnings.
    /// 
    /// If additional warnings are supposed to be added, adds a final entry "..." instead.
    pub fn unpack_into_with_cap(self, warnings: &mut Vec<String>, cap: usize) -> T {
        if warnings.len() < cap {
            let add_dots = warnings.len() + self.warnings.len() > cap;
            warnings.extend(self.warnings.into_iter().take(cap - warnings.len()));
            if add_dots {
                warnings.push("...".to_string());
            }
        } else if warnings.len() == cap && self.warnings.len() > 0 {
            warnings.push("...".to_string());
        }
        self.value
    }

    /// Adds `warnings` to `self.warnings` (at the back) while creating a new instance.
    pub fn with_extra_warnings(self, extra_warnings: Vec<String>) -> Self {
        let mut w = self.warnings;
        w.extend(extra_warnings.into_iter());
        Status { value: self.value, warnings: w }
    }

    /// Returns `Status(f(u.value, v.value))` with the combined warnings of `u` and `v`.
    pub fn combine<U, V, F: FnOnce(U, V) -> Result<T, String>>(u: Status<U>, v: Status<V>, f: F) -> Result<Status<T>, String> {
        let mut warnings = u.warnings;
        warnings.extend(v.warnings);
        f(u.value, v.value).map(|value| Status{value, warnings})
    }
    /// Acts like `combine` but accepts three statuses.
    pub fn combine_three<U, V, W, F: FnOnce(U, V, W) -> Result<T, String>>(u: Status<U>, v: Status<V>, w: Status<W>, f: F) -> Result<Status<T>, String> {
        let mut warnings = u.warnings;
        warnings.extend(v.warnings);
        warnings.extend(w.warnings);
        f(u.value, v.value, w.value).map(|value| Status{value, warnings})
    }
    /// Acts like `combine` but flattens the result before returning it.
    pub fn combine_flatten<U, V, F: FnOnce(U, V) -> Result<Status<T>, String>>(lhs: Status<U>, rhs: Status<V>, f: F) -> Result<Status<T>, String> {
        let mut warnings = lhs.warnings;
        warnings.extend(rhs.warnings);
        f(lhs.value, rhs.value).map(|value| Status{value, warnings}.flatten())
    }

    /// Joins all warnings into a single string.
    pub fn warning_str(&self) -> String {
        self.warnings.join("\n")
    }
}

impl<T> Status<Vec<T>> {
    /// Moves `other` into `self`, leaving `other` empty. Appends `self.other` at the back of `self.warnings`.
    pub fn push(&mut self, other: Status<T>) {
        self.value.push(other.value);
        self.warnings.extend(other.warnings);
    }

    /// Merges `other` into `self`, leaving `other` empty.
    pub fn merge(&mut self, other: Status<Vec<T>>) {
        self.value.extend(other.value);
        self.warnings.extend(other.warnings);
    }
}

impl<T> Status<Status<T>> {
    /// Extracts the value of the inner status and appends the warnings of the inner status to the ones of the external status.
    pub fn flatten(self) -> Status<T> {
        let mut warnings = self.warnings;
        warnings.extend(self.value.warnings);
        Status{value: self.value.value, warnings}
    }
}

impl Status<Object> {
    pub fn into_multline(self) -> Vec<String> {
        let mut res = self.value.to_multline();
        res.reserve(self.warnings.len());
        for (i, warning) in self.warnings.into_iter().enumerate() {
            res.push(format!("[WARNING #{}] {}", i+1, warning))
        }
        res
    }
}


// Below implementations are just there to simplify typing. They do not carry much logical weight.
impl<T> Status<T> where T: ops::Neg<Output=Result<T, String>> {
    /// Implemented without `std::ops` because doing so would cause errors when writing `-eval(inner)?` because of type inference.
    pub fn neg(self) -> Result<Status<T>, String> {
        self.try_map(|t| -t)
    }
}
impl<T> Status<T> where T: ops::Not<Output=Result<T, String>> {
    /// Implemented without `std::ops` because doing so would cause errors when writing `!eval(inner)?` because of type inference.
    pub fn not(self) -> Result<Status<T>, String> {
        self.try_map(|t| !t)
    }
}