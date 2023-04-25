use crate::float_trait::Float;
use crate::multicolor::PassbandTrait;

pub use lazy_static::lazy_static;
pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::fmt::Debug;

/// A passband specified by a single wavelength.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MonochromePassband<'a, T> {
    pub name: &'a str,
    pub wavelength: T,
}

impl<'a, T> MonochromePassband<'a, T>
where
    T: Float,
{
    /// Create a new `MonochromePassband`.
    ///
    /// # Arguments
    /// - `wavelength`: The wavelength of the passband, panic if it is not a positive normal number.
    /// - `name`: The name of the passband.
    pub fn new(wavelength: T, name: &'a str) -> Self {
        assert!(
            wavelength.is_normal(),
            "wavelength must be a positive normal number"
        );
        assert!(
            wavelength.is_sign_positive(),
            "wavelength must be a positive normal number"
        );
        Self { wavelength, name }
    }
}

impl<'a, T> PartialEq for MonochromePassband<'a, T>
where
    T: Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.wavelength.eq(&other.wavelength)
    }
}

impl<'a, T> Eq for MonochromePassband<'a, T> where T: Float {}

impl<'a, T> PartialOrd for MonochromePassband<'a, T>
where
    T: Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (self.wavelength).partial_cmp(&other.wavelength)
    }
}

impl<'a, T> Ord for MonochromePassband<'a, T>
where
    T: Float,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<'a, T> PassbandTrait for MonochromePassband<'a, T>
where
    T: Float,
{
    fn name(&self) -> &str {
        self.name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_monochrome_passband() {
        let passband = MonochromePassband::new(1.0, "test");
        assert_eq!(passband.name(), "test");
        assert_eq!(passband.wavelength, 1.0);
    }
}
