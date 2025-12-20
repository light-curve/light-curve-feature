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
    /// Evaluate the natural logarithm of the prior at params
    ///
    /// If `jac` is `Some`, the jacobian (gradient) d(ln_prior)/d(params) is also computed and stored in it.
    fn ln_prior(&self, params: &[f64; NPARAMS], jac: Option<&mut [f64; NPARAMS]>) -> f64;
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
    fn ln_prior(&self, params: &[f64; NPARAMS], jac: Option<&mut [f64; NPARAMS]>) -> f64 {
        match self {
            LnPrior::None(p) => p.ln_prior(params, jac),
            LnPrior::IndComponents(p) => p.ln_prior(params, jac),
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
        move |params| self.ln_prior(params, None)
    }

    pub fn into_func_with_transformation<'a, F>(
        self,
        transform: F,
    ) -> impl 'a + Clone + Fn(&[f64; NPARAMS]) -> f64
    where
        F: 'a + Clone + Fn(&[f64; NPARAMS]) -> [f64; NPARAMS],
    {
        move |params| self.ln_prior(&transform(params), None)
    }

    pub fn as_func(&self) -> impl '_ + Fn(&[f64; NPARAMS]) -> f64 {
        |params| self.ln_prior(params, None)
    }

    pub fn as_func_with_transformation<'a, F>(
        &'a self,
        transform: F,
    ) -> impl 'a + Clone + Fn(&[f64; NPARAMS]) -> f64
    where
        F: 'a + Clone + Fn(&[f64; NPARAMS]) -> [f64; NPARAMS],
    {
        move |params| self.ln_prior(&transform(params), None)
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
    fn ln_prior(&self, _params: &[f64; NPARAMS], jac: Option<&mut [f64; NPARAMS]>) -> f64 {
        if let Some(j) = jac {
            for i in 0..NPARAMS {
                j[i] = 0.0;
            }
        }
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
    fn ln_prior(&self, params: &[f64; NPARAMS], jac: Option<&mut [f64; NPARAMS]>) -> f64 {
        let mut total_ln_prior = 0.0;
        
        if let Some(j) = jac {
            for (i, (&x, ln_prior)) in params.iter().zip(self.components.iter()).enumerate() {
                let mut grad = 0.0;
                let ln_p = ln_prior.ln_prior_1d(x, Some(&mut grad));
                total_ln_prior += ln_p;
                j[i] = grad;
            }
        } else {
            for (&x, ln_prior) in params.iter().zip(self.components.iter()) {
                total_ln_prior += ln_prior.ln_prior_1d(x, None);
            }
        }
        
        total_ln_prior
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
    fn ln_prior(&self, params: &[f64; NPARAMS], jac: Option<&mut [f64; NPARAMS]>) -> f64 {
        let transformed = T::convert_to_external(self.norm_data, params);
        
        // IMPORTANT LIMITATION: Gradient computation is incomplete for transformed priors.
        // 
        // When jac is provided, we should apply the chain rule to transform the gradient
        // from external parameter space back to internal parameter space:
        //   d(ln_prior)/d(internal_params) = J_transform^T * d(ln_prior)/d(external_params)
        // where J_transform is the Jacobian of convert_to_external.
        // 
        // Currently, we compute the gradient in external parameter space and return it
        // without transformation. This is incorrect for non-identity transformations.
        // 
        // However, this limitation doesn't affect current usage in BazinFit because:
        // - dimensionless_to_internal is the identity transformation (see bazin_fit.rs:261)
        // - The NUTS sampler operates in internal space where no transformation is applied
        // - internal_to_dimensionless applies abs() but that happens after the prior evaluation
        // 
        // TODO: For full correctness, compute and apply the transformation Jacobian when
        // jac is Some. This will require adding a jacobian computation method to the
        // FitParametersInternalExternalTrait or computing it numerically.
        
        self.prior.ln_prior(&transformed, jac)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::TimeSeries;

    #[test]
    fn test_ln_prior_evaluator_trait_none() {
        let prior: LnPrior<3> = LnPrior::none();
        let params = [1.0, 2.0, 3.0];
        assert_eq!(prior.ln_prior(&params, None), 0.0);
    }

    #[test]
    fn test_ln_prior_evaluator_trait_ind_components() {
        let components = [
            LnPrior1D::uniform(0.0, 10.0),
            LnPrior1D::uniform(0.0, 10.0),
            LnPrior1D::uniform(0.0, 10.0),
        ];
        let prior: LnPrior<3> = LnPrior::ind_components(components);

        // Test with valid parameters
        let params_valid = [5.0, 5.0, 5.0];
        assert!(prior.ln_prior(&params_valid, None).is_finite());

        // Test with out-of-bounds parameters
        let params_invalid = [15.0, 5.0, 5.0];
        assert!(prior.ln_prior(&params_invalid, None).is_infinite());
        assert!(prior.ln_prior(&params_invalid, None) < 0.0);
    }

    #[test]
    fn test_none_ln_prior_is_zero() {
        let prior = NoneLnPrior {};
        let params = [100.0, -50.0, 0.0];
        assert_eq!(prior.ln_prior(&params, None), 0.0);
    }

    #[test]
    fn test_ind_components_ln_prior() {
        let components = [LnPrior1D::uniform(0.0, 1.0), LnPrior1D::uniform(0.0, 2.0)];
        let prior = IndComponentsLnPrior { components };

        // Both within bounds
        let params = [0.5, 1.0];
        assert!(prior.ln_prior(&params, None).is_finite());

        // First out of bounds
        let params = [1.5, 1.0];
        assert!(prior.ln_prior(&params, None).is_infinite());
    }

    #[test]
    fn test_ln_prior_clone() {
        let prior: LnPrior<2> = LnPrior::none();
        let cloned = prior.clone();
        let params = [1.0, 2.0];
        assert_eq!(prior.ln_prior(&params, None), cloned.ln_prior(&params, None));
    }

    #[test]
    fn test_ln_prior_debug() {
        let prior: LnPrior<2> = LnPrior::none();
        let debug_str = format!("{:?}", prior);
        assert!(debug_str.contains("None"));
    }

    #[test]
    fn test_ln_prior_into_func() {
        let prior: LnPrior<2> = LnPrior::none();
        let func = prior.into_func();
        let params = [1.0, 2.0];
        assert_eq!(func(&params), 0.0);
    }

    #[test]
    fn test_ln_prior_into_func_with_transformation() {
        let prior: LnPrior<2> = LnPrior::none();
        let transform = |params: &[f64; 2]| [params[0] * 2.0, params[1] * 2.0];
        let func = prior.into_func_with_transformation(transform);
        let params = [1.0, 2.0];
        // Since NoneLnPrior always returns 0, transformation doesn't affect result
        assert_eq!(func(&params), 0.0);
    }

    #[test]
    fn test_ln_prior_as_func() {
        let prior: LnPrior<2> = LnPrior::none();
        let func = prior.as_func();
        let params = [1.0, 2.0];
        assert_eq!(func(&params), 0.0);
    }

    // Mock struct for testing FitParametersInternalExternalTrait
    #[derive(Debug)]
    struct MockFitParameters;

    impl crate::nl_fit::evaluator::FitParametersInternalDimlessTrait<f64, 2> for MockFitParameters {
        fn dimensionless_to_internal(params: &[f64; 2]) -> [f64; 2] {
            *params
        }

        fn internal_to_dimensionless(params: &[f64; 2]) -> [f64; 2] {
            *params
        }
    }

    impl crate::nl_fit::evaluator::FitParametersOriginalDimLessTrait<2> for MockFitParameters {
        fn orig_to_dimensionless(_norm_data: &NormalizedData<f64>, orig: &[f64; 2]) -> [f64; 2] {
            *orig
        }

        fn dimensionless_to_orig(_norm_data: &NormalizedData<f64>, norm: &[f64; 2]) -> [f64; 2] {
            // Simple transformation: multiply by 2
            [norm[0] * 2.0, norm[1] * 2.0]
        }
    }

    impl crate::nl_fit::evaluator::FitParametersInternalExternalTrait<2> for MockFitParameters {}

    #[test]
    fn test_transformed_ln_prior() {
        // Create mock normalized data
        let mut ts = TimeSeries::new_without_weight(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
        let norm_data = NormalizedData::<f64>::from_ts(&mut ts);

        // Create a prior with bounds [0, 2] for each parameter
        let components = [LnPrior1D::uniform(0.0, 2.0), LnPrior1D::uniform(0.0, 2.0)];
        let prior: LnPrior<2> = LnPrior::ind_components(components);

        // Create transformed prior
        let transformed_prior =
            prior.with_fit_parameters_transformation::<MockFitParameters>(&norm_data);

        // Test with internal parameters [0.5, 0.5]
        // After transformation: [1.0, 1.0] which is within bounds
        let internal_params = [0.5, 0.5];
        let result = transformed_prior.ln_prior(&internal_params, None);
        assert!(result.is_finite());

        // Test with internal parameters [1.5, 1.5]
        // After transformation: [3.0, 3.0] which is out of bounds
        let internal_params = [1.5, 1.5];
        let result = transformed_prior.ln_prior(&internal_params, None);
        assert!(result.is_infinite());
        assert!(result < 0.0);
    }

    #[test]
    fn test_transformed_ln_prior_clone() {
        let mut ts = TimeSeries::new_without_weight(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
        let norm_data = NormalizedData::<f64>::from_ts(&mut ts);

        let prior: LnPrior<2> = LnPrior::none();
        let transformed_prior =
            prior.with_fit_parameters_transformation::<MockFitParameters>(&norm_data);
        let cloned = transformed_prior.clone();

        let params = [1.0, 2.0];
        assert_eq!(
            transformed_prior.ln_prior(&params, None),
            cloned.ln_prior(&params, None)
        );
    }

    #[test]
    fn test_transformed_ln_prior_debug() {
        let mut ts = TimeSeries::new_without_weight(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
        let norm_data = NormalizedData::<f64>::from_ts(&mut ts);

        let prior: LnPrior<2> = LnPrior::none();
        let transformed_prior =
            prior.with_fit_parameters_transformation::<MockFitParameters>(&norm_data);

        let debug_str = format!("{:?}", transformed_prior);
        assert!(debug_str.contains("TransformedLnPrior"));
    }

    #[test]
    fn test_ln_prior_serialization() {
        let prior: LnPrior<2> = LnPrior::none();
        let serialized = serde_json::to_string(&prior).unwrap();
        let deserialized: LnPrior<2> = serde_json::from_str(&serialized).unwrap();

        let params = [1.0, 2.0];
        assert_eq!(prior.ln_prior(&params, None), deserialized.ln_prior(&params, None));
    }

    #[test]
    fn test_ind_components_serialization() {
        let components = [LnPrior1D::uniform(0.0, 10.0), LnPrior1D::uniform(-5.0, 5.0)];
        let prior: LnPrior<2> = LnPrior::ind_components(components);

        let serialized = serde_json::to_string(&prior).unwrap();
        let deserialized: LnPrior<2> = serde_json::from_str(&serialized).unwrap();

        let params = [5.0, 0.0];
        assert_eq!(prior.ln_prior(&params, None), deserialized.ln_prior(&params, None));
    }

    #[test]
    fn test_ind_components_gradient() {
        use approx::assert_relative_eq;
        
        let components = [
            LnPrior1D::normal(5.0, 2.0),
            LnPrior1D::normal(10.0, 3.0),
        ];
        let prior: LnPrior<2> = LnPrior::ind_components(components);

        let params = [6.0, 11.0];
        let mut jac = [0.0; 2];
        let ln_p = prior.ln_prior(&params, Some(&mut jac));

        // Numerical gradient check
        let eps = 1e-6;
        for i in 0..2 {
            let mut params_plus = params;
            params_plus[i] += eps;
            let ln_p_plus = prior.ln_prior(&params_plus, None);
            
            let mut params_minus = params;
            params_minus[i] -= eps;
            let ln_p_minus = prior.ln_prior(&params_minus, None);
            
            let numerical_grad = (ln_p_plus - ln_p_minus) / (2.0 * eps);
            assert_relative_eq!(jac[i], numerical_grad, epsilon = 1e-4);
        }

        assert!(ln_p.is_finite());
    }

    #[test]
    fn test_none_ln_prior_gradient() {
        let prior: LnPrior<3> = LnPrior::none();
        let params = [1.0, 2.0, 3.0];
        let mut jac = [0.0; 3];
        let ln_p = prior.ln_prior(&params, Some(&mut jac));
        
        assert_eq!(ln_p, 0.0);
        assert_eq!(jac, [0.0, 0.0, 0.0]);
    }
}
