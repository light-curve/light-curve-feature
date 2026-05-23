use crate::PassbandTrait;

use schemars::JsonSchema;
use serde::{Deserialize, Deserializer, Serialize};
use std::fmt::{Debug, Display};

/// A passband identified by a label of any type that implements [`Display`].
///
/// The passband name exposed via [`PassbandTrait::name`] is the [`Display`] representation of the
/// label, computed once at construction and stored alongside the label. This allows the label to be
/// any type — integers, enums, or custom types — without requiring it to be string-like.
///
/// # Example
///
/// ```rust
/// use light_curve_feature::multicolor::{LabeledPassband, PassbandTrait};
///
/// let band = LabeledPassband::new(42_u8);
/// assert_eq!(band.name(), "42");
/// assert_eq!(band.label, 42_u8);
///
/// let band = LabeledPassband::new("r");
/// assert_eq!(band.name(), "r");
/// ```
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, JsonSchema)]
pub struct LabeledPassband<N> {
    pub label: N,
    #[serde(skip)]
    name: String,
}

impl<N: Display> LabeledPassband<N> {
    pub fn new(label: N) -> Self {
        let name = label.to_string();
        Self { label, name }
    }
}

impl<N: Display> From<N> for LabeledPassband<N> {
    fn from(label: N) -> Self {
        Self::new(label)
    }
}

impl<N> PassbandTrait for LabeledPassband<N>
where
    N: Display + Debug + Clone + Send + Sync + Ord + Serialize + JsonSchema,
{
    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Deserialize)]
struct LabeledPassbandHelper<N> {
    label: N,
}

impl<'de, N: Deserialize<'de> + Display> Deserialize<'de> for LabeledPassband<N> {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let LabeledPassbandHelper { label } = LabeledPassbandHelper::deserialize(deserializer)?;
        Ok(Self::new(label))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_label_name() {
        let band = LabeledPassband::new(42_u8);
        assert_eq!(band.name(), "42");
        assert_eq!(band.label, 42_u8);
    }

    #[test]
    fn string_label_name() {
        let band = LabeledPassband::new(String::from("r"));
        assert_eq!(band.name(), "r");
    }

    #[test]
    fn from_impl() {
        let band: LabeledPassband<u8> = LabeledPassband::from(7_u8);
        assert_eq!(band.name(), "7");
    }

    #[test]
    fn serde_roundtrip_integer() {
        let band = LabeledPassband::new(3_u32);
        let json = serde_json::to_string(&band).unwrap();
        let band2: LabeledPassband<u32> = serde_json::from_str(&json).unwrap();
        assert_eq!(band, band2);
        assert_eq!(band2.name(), "3");
    }

    #[test]
    fn serde_roundtrip_string() {
        let band = LabeledPassband::new(String::from("g"));
        let json = serde_json::to_string(&band).unwrap();
        let band2: LabeledPassband<String> = serde_json::from_str(&json).unwrap();
        assert_eq!(band, band2);
        assert_eq!(band2.name(), "g");
    }
}
