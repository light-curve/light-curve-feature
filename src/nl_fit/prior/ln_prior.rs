use crate::nl_fit::data::NormalizedData;
use crate::nl_fit::prior::ln_prior_1d::{LnPrior1D, LnPrior1DTrait};

use enum_dispatch::enum_dispatch;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Core trait for evaluating the natural logarithm of a prior
///
/// This trait is implemented by types that can evaluate ln(prior) for a given set of parameters.
/// Unlike [LnPriorTrait], this trait does not require serialization, making it suitable for
/// use with closures and other non-serializable types.
pub trait LnPriorEvaluator<const NPARAMS: usize>: Clone {
    fn ln_prior(&self, params: &[f64; NPARAMS]) -> f64;
}

/// Trait for serializable prior evaluators
///
/// This trait extends [LnPriorEvaluator] with serialization requirements. It is used for
/// prior types that need to be serialized/deserialized, such as the [LnPrior] enum.
///
/// Use [LnPriorEvaluator] directly when you don't need serialization (e.g., for closures
/// or temporary prior objects). Use this trait when you need to serialize the prior
/// configuration.
#[enum_dispatch]
pub trait LnPriorTrait<const NPARAMS: usize>:
    LnPriorEvaluator<NPARAMS> + Debug + Serialize + DeserializeOwned
{
}

/// Natural logarithm of prior for non-linear curve-fit problem
#[enum_dispatch(LnPriorTrait<NPARAMS>)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LnPrior<const NPARAMS: usize> {
    None(NoneLnPrior),
    IndComponents(IndComponentsLnPrior<NPARAMS>),
}

impl<const NPARAMS: usize> LnPriorEvaluator<NPARAMS> for LnPrior<NPARAMS> {
    fn ln_prior(&self, params: &[f64; NPARAMS]) -> f64 {
        match self {
            LnPrior::None(p) => p.ln_prior(params),
            LnPrior::IndComponents(p) => p.ln_prior(params),
        }
    }
}

impl<const NPARAMS: usize> LnPrior<NPARAMS> {
    pub fn none() -> Self {
        NoneLnPrior {}.into()
    }

    pub fn ind_components(components: [LnPrior1D; NPARAMS]) -> Self {
        IndComponentsLnPrior { components }.into()
    }

    pub fn into_func(self) -> impl 'static + Clone + Fn(&[f64; NPARAMS]) -> f64 {
        move |params| self.ln_prior(params)
    }

    pub fn into_func_with_transformation<'a, F>(
        self,
        transform: F,
    ) -> impl 'a + Clone + Fn(&[f64; NPARAMS]) -> f64
    where
        F: 'a + Clone + Fn(&[f64; NPARAMS]) -> [f64; NPARAMS],
    {
        move |params| self.ln_prior(&transform(params))
    }

    pub fn as_func(&self) -> impl '_ + Fn(&[f64; NPARAMS]) -> f64 {
        |params| self.ln_prior(params)
    }

    pub fn as_func_with_transformation<'a, F>(
        &'a self,
        transform: F,
    ) -> impl 'a + Clone + Fn(&[f64; NPARAMS]) -> f64
    where
        F: 'a + Clone + Fn(&[f64; NPARAMS]) -> [f64; NPARAMS],
    {
        move |params| self.ln_prior(&transform(params))
    }

    /// Create a transformed prior that applies parameter transformation using FitParametersInternalExternalTrait
    ///
    /// This method creates a wrapper that stores references to the prior and normalization data,
    /// allowing it to be fully debuggable. The transformation is applied using the trait's
    /// `convert_to_external` method from the `FitParametersInternalExternalTrait` trait.
    pub fn with_fit_parameters_transformation<'a, T>(
        &'a self,
        norm_data: &'a NormalizedData<f64>,
    ) -> TransformedLnPrior<'a, T, NPARAMS>
    where
        T: crate::nl_fit::evaluator::FitParametersInternalExternalTrait<NPARAMS>,
    {
        TransformedLnPrior {
            prior: self.clone(),
            norm_data,
            _phantom: std::marker::PhantomData,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub struct NoneLnPrior {}

impl<const NPARAMS: usize> LnPriorEvaluator<NPARAMS> for NoneLnPrior {
    fn ln_prior(&self, _params: &[f64; NPARAMS]) -> f64 {
        0.0
    }
}

impl<const NPARAMS: usize> LnPriorTrait<NPARAMS> for NoneLnPrior {}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(
    into = "IndComponentsLnPriorSerde",
    try_from = "IndComponentsLnPriorSerde"
)]
pub struct IndComponentsLnPrior<const NPARAMS: usize> {
    pub components: [LnPrior1D; NPARAMS],
}

impl<const NPARAMS: usize> LnPriorEvaluator<NPARAMS> for IndComponentsLnPrior<NPARAMS> {
    fn ln_prior(&self, params: &[f64; NPARAMS]) -> f64 {
        params
            .iter()
            .zip(self.components.iter())
            .map(|(&x, ln_prior)| ln_prior.ln_prior_1d(x))
            .sum()
    }
}

impl<const NPARAMS: usize> LnPriorTrait<NPARAMS> for IndComponentsLnPrior<NPARAMS> {}

impl<const NPARAMS: usize> JsonSchema for IndComponentsLnPrior<NPARAMS> {
    fn is_referenceable() -> bool {
        false
    }

    fn schema_name() -> String {
        IndComponentsLnPriorSerde::schema_name()
    }

    fn json_schema(r#gen: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        IndComponentsLnPriorSerde::json_schema(r#gen)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "IndComponentsLnPrior")]
struct IndComponentsLnPriorSerde {
    components: Vec<LnPrior1D>,
}

impl<const NPARAMS: usize> From<IndComponentsLnPrior<NPARAMS>> for IndComponentsLnPriorSerde {
    fn from(value: IndComponentsLnPrior<NPARAMS>) -> Self {
        Self {
            components: value.components.into(),
        }
    }
}

impl<const NPARAMS: usize> TryFrom<IndComponentsLnPriorSerde> for IndComponentsLnPrior<NPARAMS> {
    type Error = &'static str;

    fn try_from(value: IndComponentsLnPriorSerde) -> Result<Self, Self::Error> {
        Ok(Self {
            components: value
                .components
                .try_into()
                .map_err(|_| "wrong size of the IndComponentsLnPrior.components")?,
        })
    }
}

/// A prior with parameter transformation using FitParametersInternalExternalTrait
///
/// This type wraps a [`LnPrior`] and a reference to `NormalizedData`, applying parameter
/// transformation using the `convert_to_external` method from `FitParametersInternalExternalTrait`.
/// This allows the prior to be evaluated in the external parameter space while being
/// fully debuggable.
///
/// Note: This type stores a reference to `NormalizedData` which is runtime data, so it cannot
/// be serialized. However, the prior itself can be serialized separately.
#[derive(Debug)]
pub struct TransformedLnPrior<'a, T, const NPARAMS: usize>
where
    T: crate::nl_fit::evaluator::FitParametersInternalExternalTrait<NPARAMS>,
{
    prior: LnPrior<NPARAMS>,
    norm_data: &'a NormalizedData<f64>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T, const NPARAMS: usize> Clone for TransformedLnPrior<'a, T, NPARAMS>
where
    T: crate::nl_fit::evaluator::FitParametersInternalExternalTrait<NPARAMS>,
{
    fn clone(&self) -> Self {
        Self {
            prior: self.prior.clone(),
            norm_data: self.norm_data,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T, const NPARAMS: usize> LnPriorEvaluator<NPARAMS> for TransformedLnPrior<'a, T, NPARAMS>
where
    T: crate::nl_fit::evaluator::FitParametersInternalExternalTrait<NPARAMS>,
{
    fn ln_prior(&self, params: &[f64; NPARAMS]) -> f64 {
        let transformed = T::convert_to_external(self.norm_data, params);
        self.prior.ln_prior(&transformed)
    }
}
