use crate::transformers::transformer::*;

use conv::prelude::*;
use std::hash::{Hash, Hasher};

macro_const! {
    const DOC: &str = r#"
Decimal logarithm of a value clipped to a minimum value
"#;
}

#[doc = DOC!()]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
pub struct ClippedLgTransformer<T> {
    pub min_value: T,
}

impl<T> Hash for ClippedLgTransformer<T>
where
    T: Float,
{
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Convert to f64 and use its bit representation for hashing
        let min_value_f64: f64 = self.min_value.value_into().unwrap();
        min_value_f64.to_bits().hash(state);
    }
}

impl<T> ClippedLgTransformer<T>
where
    T: Float,
{
    pub fn new(min_value: T) -> Self {
        Self { min_value }
    }

    /// Default is f64::MIN_POSITIVE.log10()
    pub fn default_zero_value() -> T {
        f64::MIN_POSITIVE.log10().approx().unwrap()
    }

    pub const fn doc() -> &'static str {
        DOC
    }

    #[inline]
    fn transform_one(&self, x: T) -> T {
        if x < T::min_positive_value() {
            self.min_value
        } else {
            x.log10()
        }
    }

    #[inline]
    fn transform_one_name(&self, name: &str) -> String {
        format!("clipped_lg_{name}")
    }

    #[inline]
    fn transform_one_description(&self, desc: &str) -> String {
        format!(
            "Maximum of {:.3} or decimal logarithm of {}",
            self.min_value, desc
        )
    }
}

impl<T> Default for ClippedLgTransformer<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(Self::default_zero_value())
    }
}

impl<T> TransformerPropsTrait for ClippedLgTransformer<T>
where
    T: Float,
{
    #[inline]
    fn is_size_valid(&self, _size: usize) -> bool {
        true
    }

    #[inline]
    fn size_hint(&self, size: usize) -> usize {
        size
    }

    fn names(&self, names: &[&str]) -> Vec<String> {
        names
            .iter()
            .map(|name| self.transform_one_name(name))
            .collect()
    }

    fn descriptions(&self, desc: &[&str]) -> Vec<String> {
        desc.iter()
            .map(|name| self.transform_one_description(name))
            .collect()
    }
}

impl<T> TransformerTrait<T> for ClippedLgTransformer<T>
where
    T: Float,
{
    fn transform(&self, x: Vec<T>) -> Vec<T> {
        x.into_iter().map(|x| self.transform_one(x)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    check_transformer!(ClippedLgTransformer<f64>);
}
