pub(super) use crate::float_trait::Float;

use enum_dispatch::enum_dispatch;
pub(super) use macro_const::macro_const;
pub(super) use schemars::JsonSchema;
pub(super) use serde::de::DeserializeOwned;
pub(super) use serde::{Deserialize, Serialize};
pub(super) use std::fmt::Debug;

#[enum_dispatch]
pub trait TransformerPropsTrait {
    /// Is the size of the input vector valid?
    fn is_size_valid(&self, size: usize) -> bool;

    /// What is the size of the output vector for a given input vector size?
    fn size_hint(&self, size: usize) -> usize;

    /// Transform the names of the input features.
    fn names(&self, names: &[&str]) -> Vec<String>;

    /// Transform the descriptions of the input features.
    fn descriptions(&self, desc: &[&str]) -> Vec<String>;
}

#[enum_dispatch]
pub trait TransformerTrait<T: Float>:
    TransformerPropsTrait + Clone + Debug + Send + Serialize + DeserializeOwned + JsonSchema
{
    /// Transform the input vector.
    fn transform(&self, x: Vec<T>) -> Vec<T>;
}

#[enum_dispatch(TransformerTrait<T>, TransformerPropsTrait)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound = "T: Float")]
#[non_exhaustive]
pub enum Transformer<T: Float> {
    Arcsinh(super::arcsinh::ArcsinhTransformer),
    BazinFit(super::bazin_fit::BazinFitTransformer<T>),
    ClippedLg(super::clipped_lg::ClippedLgTransformer<T>),
    Composed(super::composed::ComposedTransformer<Self>),
    Identity(super::identity::IdentityTransformer),
    LinexpFit(super::linexp_fit::LinexpFitTransformer<T>),
    Ln1p(super::ln1p::Ln1pTransformer),
    Lg(super::lg::LgTransformer),
    Sqrt(super::sqrt::SqrtTransformer),
    VillarFit(super::villar_fit::VillarFitTransformer<T>),
}
