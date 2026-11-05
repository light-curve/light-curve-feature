use crate::transformers::transformer::*;

use crate::float_trait::{float_total_eq, hash_float};
use conv::prelude::*;
use std::hash::{Hash, Hasher};

macro_const! {
    const DOC: &str = r#"
Decimal logarithm of a value clipped to a minimum value
"#;
}

#[doc = DOC!()]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ClippedLgTransformer<T> {
    pub min_value: T,
}

impl<T: Float> PartialEq for ClippedLgTransformer<T> {
    fn eq(&self, other: &Self) -> bool {
        float_total_eq(self.min_value, other.min_value)
    }
}

impl<T: Float> Eq for ClippedLgTransformer<T> {}

impl<T: Float> Hash for ClippedLgTransformer<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        hash_float(self.min_value, state);
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
