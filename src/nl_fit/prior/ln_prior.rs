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
pub trait LnPriorEvaluator: Clone {
    /// Evaluate the natural logarithm of the prior at params
    ///
    /// If `jac` is `Some`, the jacobian (gradient) d(ln_prior)/d(params) is also computed and stored in it.
    /// `jac`, when given, must be the same length as `params`.
    fn ln_prior(&self, params: &[f64], jac: Option<&mut [f64]>) -> f64;
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
pub trait LnPriorTrait: LnPriorEvaluator + Debug + Serialize + DeserializeOwned {}

/// Natural logarithm of prior for non-linear curve-fit problem
#[enum_dispatch(LnPriorTrait)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LnPrior {
    None(NoneLnPrior),
    IndComponents(IndComponentsLnPrior),
}

impl LnPriorEvaluator for LnPrior {
    fn ln_prior(&self, params: &[f64], jac: Option<&mut [f64]>) -> f64 {
        match self {
            LnPrior::None(p) => p.ln_prior(params, jac),
            LnPrior::IndComponents(p) => p.ln_prior(params, jac),
        }
    }
}

impl LnPrior {
    pub fn none() -> Self {
        NoneLnPrior {}.into()
    }

    pub fn ind_components(components: impl Into<Vec<LnPrior1D>>) -> Self {
        IndComponentsLnPrior {
            components: components.into(),
        }
        .into()
    }

    pub fn into_func(self) -> impl 'static + Clone + Fn(&[f64]) -> f64 {
        move |params| self.ln_prior(params, None)
    }

    pub fn into_func_with_transformation<'a, F>(
        self,
        transform: F,
    ) -> impl 'a + Clone + Fn(&[f64]) -> f64
    where
        F: 'a + Clone + Fn(&[f64]) -> Vec<f64>,
    {
        move |params| self.ln_prior(&transform(params), None)
    }

    pub fn as_func(&self) -> impl '_ + Fn(&[f64]) -> f64 {
        |params| self.ln_prior(params, None)
    }

    pub fn as_func_with_transformation<'a, F>(
        &'a self,
        transform: F,
    ) -> impl 'a + Clone + Fn(&[f64]) -> f64
    where
        F: 'a + Clone + Fn(&[f64]) -> Vec<f64>,
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
    ) -> TransformedLnPrior<'a, T>
    where
        T: crate::nl_fit::evaluator::FitParametersInternalExternalTrait,
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

impl LnPriorEvaluator for NoneLnPrior {
    fn ln_prior(&self, _params: &[f64], jac: Option<&mut [f64]>) -> f64 {
        if let Some(j) = jac {
            j.iter_mut().for_each(|x| *x = 0.0);
        }
        0.0
    }
}

impl LnPriorTrait for NoneLnPrior {}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename = "IndComponentsLnPrior")]
pub struct IndComponentsLnPrior {
    pub components: Vec<LnPrior1D>,
}

impl LnPriorEvaluator for IndComponentsLnPrior {
    fn ln_prior(&self, params: &[f64], jac: Option<&mut [f64]>) -> f64 {
        let mut total_ln_prior = 0.0;

        if let Some(jac) = jac {
            for ((grad_out, &x), ln_prior) in jac
                .iter_mut()
                .zip(params.iter())
                .zip(self.components.iter())
            {
                let mut grad = 0.0;
                let ln_p = ln_prior.ln_prior_1d(x, Some(&mut grad));
                total_ln_prior += ln_p;
                *grad_out = grad;
            }
        } else {
            for (&x, ln_prior) in params.iter().zip(self.components.iter()) {
                total_ln_prior += ln_prior.ln_prior_1d(x, None);
            }
        }

        total_ln_prior
    }
}

impl LnPriorTrait for IndComponentsLnPrior {}

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
pub struct TransformedLnPrior<'a, T>
where
    T: crate::nl_fit::evaluator::FitParametersInternalExternalTrait,
{
    prior: LnPrior,
    norm_data: &'a NormalizedData<f64>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> Clone for TransformedLnPrior<'a, T>
where
    T: crate::nl_fit::evaluator::FitParametersInternalExternalTrait,
{
    fn clone(&self) -> Self {
        Self {
            prior: self.prior.clone(),
            norm_data: self.norm_data,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<'a, T> LnPriorEvaluator for TransformedLnPrior<'a, T>
where
    T: crate::nl_fit::evaluator::FitParametersInternalExternalTrait,
{
    fn ln_prior(&self, params: &[f64], jac: Option<&mut [f64]>) -> f64 {
        let transformed = T::convert_to_external(self.norm_data, params);

        match jac {
            Some(jac) => {
                // Evaluate the prior in external parameter space and get the gradient
                let ln_p = self.prior.ln_prior(&transformed, Some(jac));

                // Apply the chain rule to transform the gradient from external to internal space:
                // ∂(ln_prior)/∂(internal_i) = ∂(ln_prior)/∂(external_i) × ∂(external_i)/∂(internal_i)
                let jacobian = T::jacobian_internal_to_external(self.norm_data, params);
                for (grad, jac_elem) in jac.iter_mut().zip(jacobian.iter()) {
                    *grad *= jac_elem;
                }

                ln_p
            }
            None => self.prior.ln_prior(&transformed, None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TimeSeries;

    #[test]
    fn test_ln_prior_evaluator_trait_none() {
        let prior: LnPrior = LnPrior::none();
        let params = [1.0, 2.0, 3.0];
        assert_eq!(prior.ln_prior(&params, None), 0.0);
    }

    #[test]
    fn test_ln_prior_evaluator_trait_ind_components() {
        let components = vec![
            LnPrior1D::uniform(0.0, 10.0),
            LnPrior1D::uniform(0.0, 10.0),
            LnPrior1D::uniform(0.0, 10.0),
        ];
        let prior: LnPrior = LnPrior::ind_components(components);

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
        let components = vec![LnPrior1D::uniform(0.0, 1.0), LnPrior1D::uniform(0.0, 2.0)];
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
        let prior: LnPrior = LnPrior::none();
        let cloned = prior.clone();
        let params = [1.0, 2.0];
        assert_eq!(
            prior.ln_prior(&params, None),
            cloned.ln_prior(&params, None)
        );
    }

    #[test]
    fn test_ln_prior_debug() {
        let prior: LnPrior = LnPrior::none();
        let debug_str = format!("{:?}", prior);
        assert!(debug_str.contains("None"));
    }

    #[test]
    fn test_ln_prior_into_func() {
        let prior: LnPrior = LnPrior::none();
        let func = prior.into_func();
        let params = [1.0, 2.0];
        assert_eq!(func(&params), 0.0);
    }

    #[test]
    fn test_ln_prior_into_func_with_transformation() {
        let prior: LnPrior = LnPrior::none();
        let transform = |params: &[f64]| vec![params[0] * 2.0, params[1] * 2.0];
        let func = prior.into_func_with_transformation(transform);
        let params = [1.0, 2.0];
        // Since NoneLnPrior always returns 0, transformation doesn't affect result
        assert_eq!(func(&params), 0.0);
    }

    #[test]
    fn test_ln_prior_as_func() {
        let prior: LnPrior = LnPrior::none();
        let func = prior.as_func();
        let params = [1.0, 2.0];
        assert_eq!(func(&params), 0.0);
    }

    // Mock struct for testing FitParametersInternalExternalTrait
    #[derive(Debug)]
    struct MockFitParameters;

    impl crate::nl_fit::evaluator::FitParametersInternalDimlessTrait<f64> for MockFitParameters {
        fn dimensionless_to_internal(
            params: &[f64],
        ) -> smallvec::SmallVec<[f64; crate::nl_fit::evaluator::MAX_INLINE_PARAMS]> {
            smallvec::SmallVec::from_slice(params)
        }

        fn internal_to_dimensionless(
            params: &[f64],
        ) -> smallvec::SmallVec<[f64; crate::nl_fit::evaluator::MAX_INLINE_PARAMS]> {
            smallvec::SmallVec::from_slice(params)
        }
    }

    impl crate::nl_fit::evaluator::FitParametersOriginalDimLessTrait for MockFitParameters {
        fn orig_to_dimensionless(
            _norm_data: &NormalizedData<f64>,
            orig: &[f64],
        ) -> smallvec::SmallVec<[f64; crate::nl_fit::evaluator::MAX_INLINE_PARAMS]> {
            smallvec::SmallVec::from_slice(orig)
        }

        fn dimensionless_to_orig(
            _norm_data: &NormalizedData<f64>,
            norm: &[f64],
        ) -> smallvec::SmallVec<[f64; crate::nl_fit::evaluator::MAX_INLINE_PARAMS]> {
            // Simple transformation: multiply by 2
            norm.iter().map(|&x| x * 2.0).collect()
        }
    }

    impl crate::nl_fit::evaluator::FitParametersInternalExternalTrait for MockFitParameters {
        fn jacobian_internal_to_external(
            _norm_data: &NormalizedData<f64>,
            internal: &[f64],
        ) -> smallvec::SmallVec<[f64; crate::nl_fit::evaluator::MAX_INLINE_PARAMS]> {
            // For MockFitParameters:
            // - internal_to_dimensionless is identity, so derivative is 1
            // - dimensionless_to_orig multiplies by 2, so derivative is 2
            // Combined: 1 * 2 = 2 for each component
            smallvec::smallvec![2.0; internal.len()]
        }
    }

    #[test]
    fn test_transformed_ln_prior() {
        // Create mock normalized data
        let mut ts = TimeSeries::new_without_weight(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
        let norm_data = NormalizedData::<f64>::from_ts(&mut ts);

        // Create a prior with bounds [0, 2] for each parameter
        let components = vec![LnPrior1D::uniform(0.0, 2.0), LnPrior1D::uniform(0.0, 2.0)];
        let prior: LnPrior = LnPrior::ind_components(components);

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

        let prior: LnPrior = LnPrior::none();
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

        let prior: LnPrior = LnPrior::none();
        let transformed_prior =
            prior.with_fit_parameters_transformation::<MockFitParameters>(&norm_data);

        let debug_str = format!("{:?}", transformed_prior);
        assert!(debug_str.contains("TransformedLnPrior"));
    }

    #[test]
    fn test_ln_prior_serialization() {
        let prior: LnPrior = LnPrior::none();
        let serialized = serde_json::to_string(&prior).unwrap();
        let deserialized: LnPrior = serde_json::from_str(&serialized).unwrap();

        let params = [1.0, 2.0];
        assert_eq!(
            prior.ln_prior(&params, None),
            deserialized.ln_prior(&params, None)
        );
    }

    #[test]
    fn test_ind_components_serialization() {
        let components = vec![LnPrior1D::uniform(0.0, 10.0), LnPrior1D::uniform(-5.0, 5.0)];
        let prior: LnPrior = LnPrior::ind_components(components);

        let serialized = serde_json::to_string(&prior).unwrap();
        let deserialized: LnPrior = serde_json::from_str(&serialized).unwrap();

        let params = [5.0, 0.0];
        assert_eq!(
            prior.ln_prior(&params, None),
            deserialized.ln_prior(&params, None)
        );
    }

    #[test]
    fn test_ind_components_gradient() {
        use approx::assert_relative_eq;

        let components = vec![LnPrior1D::normal(5.0, 2.0), LnPrior1D::normal(10.0, 3.0)];
        let prior: LnPrior = LnPrior::ind_components(components.clone());

        let params = [6.0, 11.0];
        let mut jac = [0.0; 2];
        let ln_p = prior.ln_prior(&params, Some(&mut jac));

        // Verify gradient matches individual component gradients
        // For normal(mu, sigma): d(ln_prior)/d(x) = -(x - mu) / sigma^2
        let expected_jac_0 = -(params[0] - 5.0) / (2.0 * 2.0); // normal(5.0, 2.0)
        let expected_jac_1 = -(params[1] - 10.0) / (3.0 * 3.0); // normal(10.0, 3.0)

        assert_relative_eq!(jac[0], expected_jac_0, epsilon = 1e-10);
        assert_relative_eq!(jac[1], expected_jac_1, epsilon = 1e-10);

        // Verify ln_prior value is sum of component values
        let expected_ln_p =
            components[0].ln_prior_1d(params[0], None) + components[1].ln_prior_1d(params[1], None);
        assert_relative_eq!(ln_p, expected_ln_p, epsilon = 1e-10);

        assert!(ln_p.is_finite());
    }

    #[test]
    fn test_none_ln_prior_gradient() {
        let prior: LnPrior = LnPrior::none();
        let params = [1.0, 2.0, 3.0];
        let mut jac = [0.0; 3];
        let ln_p = prior.ln_prior(&params, Some(&mut jac));

        assert_eq!(ln_p, 0.0);
        assert_eq!(jac, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_transformed_ln_prior_gradient() {
        use approx::assert_relative_eq;

        // Create mock normalized data
        let mut ts = TimeSeries::new_without_weight(vec![1.0, 2.0, 3.0], vec![1.0, 2.0, 3.0]);
        let norm_data = NormalizedData::<f64>::from_ts(&mut ts);

        // Create a prior with normal distribution in external space
        let components = vec![LnPrior1D::normal(1.0, 0.5), LnPrior1D::normal(2.0, 0.5)];
        let prior: LnPrior = LnPrior::ind_components(components);

        // Create transformed prior
        let transformed_prior =
            prior.with_fit_parameters_transformation::<MockFitParameters>(&norm_data);

        // Test at internal params [0.5, 1.0]
        // MockFitParameters transforms: external = internal * 2
        // So external = [1.0, 2.0]
        let internal = [0.5, 1.0];
        let external = [1.0, 2.0]; // internal * 2

        // Get gradient from transformed prior
        let mut actual_jac = [0.0; 2];
        let ln_p = transformed_prior.ln_prior(&internal, Some(&mut actual_jac));

        // Compute expected gradient manually:
        // 1. Get prior gradient in external space
        let mut external_jac = [0.0; 2];
        let expected_ln_p = prior.ln_prior(&external, Some(&mut external_jac));

        // 2. Apply chain rule: d(ln_prior)/d(internal) = d(ln_prior)/d(external) * d(external)/d(internal)
        // MockFitParameters.jacobian_internal_to_external returns [2.0, 2.0]
        let jacobian = [2.0, 2.0];
        let expected_jac = [external_jac[0] * jacobian[0], external_jac[1] * jacobian[1]];

        // Verify ln_prior value
        assert_relative_eq!(ln_p, expected_ln_p, epsilon = 1e-10);

        // Verify gradient matches chain rule
        assert_relative_eq!(actual_jac[0], expected_jac[0], epsilon = 1e-10);
        assert_relative_eq!(actual_jac[1], expected_jac[1], epsilon = 1e-10);

        assert!(ln_p.is_finite());
    }
}
