use crate::data::TimeSeries;
use crate::float_trait::Float;
use crate::nl_fit::{CurveFitAlgorithm, LikeFloat, LnPrior, data::NormalizedData};

use arrayvec::ArrayVec;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Inline capacity for [`FitParametersInternalDimlessTrait`]'s per-point-hot transforms: covers
/// every current fixed-`NPARAMS` feature (Bazin=5, Linexp=4, Villar=7) without spilling to the
/// heap. Larger parameter counts still work, just via a heap allocation past this size.
pub const MAX_INLINE_PARAMS: usize = 8;

pub trait FitModelTrait<T, U>
where
    T: Float + Into<U>,
    U: LikeFloat,
{
    fn model(t: T, param: &[U]) -> U
    where
        T: Float + Into<U>,
        U: LikeFloat;
}

pub trait FitFunctionTrait<T: Float>:
    FitModelTrait<T, T> + FitParametersInternalDimlessTrait<T>
{
    fn f(t: T, values: &[T]) -> T {
        let internal = Self::dimensionless_to_internal(values);
        Self::model(t, &internal)
    }
}

pub trait FitDerivalivesTrait<T: Float> {
    fn derivatives(t: T, param: &[T], jac: &mut [T]);
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq)]
pub struct FitInitsBoundsArrays {
    pub init: Vec<f64>,
    pub lower: Vec<f64>,
    pub upper: Vec<f64>,
}

impl FitInitsBoundsArrays {
    pub fn new(init: Vec<f64>, lower: Vec<f64>, upper: Vec<f64>) -> Self {
        Self { init, lower, upper }
    }
}

#[derive(Debug, Clone, Deserialize, Serialize, JsonSchema, PartialEq)]
pub struct OptionFitInitsBoundsArrays {
    pub init: Vec<Option<f64>>,
    pub lower: Vec<Option<f64>>,
    pub upper: Vec<Option<f64>>,
}

impl OptionFitInitsBoundsArrays {
    pub fn new(init: Vec<Option<f64>>, lower: Vec<Option<f64>>, upper: Vec<Option<f64>>) -> Self {
        Self { init, lower, upper }
    }

    pub fn unwrap_with(&self, x: &FitInitsBoundsArrays) -> FitInitsBoundsArrays {
        let unwrap_slice = |opt: &[Option<f64>], with: &[f64]| -> Vec<f64> {
            opt.iter().zip(with).map(|(o, &w)| o.unwrap_or(w)).collect()
        };
        FitInitsBoundsArrays {
            init: unwrap_slice(&self.init, &x.init),
            lower: unwrap_slice(&self.lower, &x.lower),
            upper: unwrap_slice(&self.upper, &x.upper),
        }
    }
}

impl From<FitInitsBoundsArrays> for OptionFitInitsBoundsArrays {
    fn from(item: FitInitsBoundsArrays) -> Self {
        Self {
            init: item.init.into_iter().map(Some).collect(),
            lower: item.lower.into_iter().map(Some).collect(),
            upper: item.upper.into_iter().map(Some).collect(),
        }
    }
}

pub trait FitInitsBoundsTrait<T: Float> {
    fn init_and_bounds_from_ts(&self, ts: &mut TimeSeries<T>) -> FitInitsBoundsArrays;
}

pub trait FitParametersInternalDimlessTrait<U: LikeFloat> {
    fn dimensionless_to_internal(params: &[U]) -> ArrayVec<U, MAX_INLINE_PARAMS>;

    fn internal_to_dimensionless(params: &[U]) -> ArrayVec<U, MAX_INLINE_PARAMS>;
}

pub trait FitParametersOriginalDimLessTrait {
    fn orig_to_dimensionless(
        norm_data: &NormalizedData<f64>,
        orig: &[f64],
    ) -> ArrayVec<f64, MAX_INLINE_PARAMS>;

    fn dimensionless_to_orig(
        norm_data: &NormalizedData<f64>,
        norm: &[f64],
    ) -> ArrayVec<f64, MAX_INLINE_PARAMS>;
}

pub trait FitParametersInternalExternalTrait:
    FitParametersInternalDimlessTrait<f64> + FitParametersOriginalDimLessTrait
{
    fn convert_to_internal(
        norm_data: &NormalizedData<f64>,
        orig: &[f64],
    ) -> ArrayVec<f64, MAX_INLINE_PARAMS> {
        Self::dimensionless_to_internal(&Self::orig_to_dimensionless(norm_data, orig))
    }

    fn convert_to_external(
        norm_data: &NormalizedData<f64>,
        params: &[f64],
    ) -> ArrayVec<f64, MAX_INLINE_PARAMS> {
        Self::dimensionless_to_orig(norm_data, &Self::internal_to_dimensionless(params))
    }

    /// Compute the diagonal Jacobian of the internal → external transformation.
    ///
    /// Returns `∂(external_i)/∂(internal_i)` for each parameter `i`.
    ///
    /// This is used to correctly transform prior gradients from external to internal
    /// parameter space via the chain rule:
    ///
    /// ```text
    /// ∂(ln_prior)/∂(internal_i) = ∂(ln_prior)/∂(external_i) × ∂(external_i)/∂(internal_i)
    /// ```
    ///
    /// The Jacobian is the composition of two transformations:
    /// 1. `internal_to_dimensionless`: often applies `abs()` to positive-only parameters
    /// 2. `dimensionless_to_orig`: linear scaling by `t_std`, `m_std`, etc.
    ///
    /// Since both transformations are element-wise, the full Jacobian is diagonal.
    fn jacobian_internal_to_external(
        norm_data: &NormalizedData<f64>,
        internal: &[f64],
    ) -> ArrayVec<f64, MAX_INLINE_PARAMS>;
}

pub trait FitFeatureEvaluatorGettersTrait {
    fn get_algorithm(&self) -> &CurveFitAlgorithm;

    fn ln_prior_from_ts<T: Float>(&self, ts: &mut TimeSeries<T>) -> LnPrior;
}
