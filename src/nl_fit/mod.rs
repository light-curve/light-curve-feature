//! Non-linear curve fitting infrastructure for light curve feature extraction.
//!
//! # Overview
//!
//! This module provides the infrastructure for fitting parametric models (like Bazin, Villar,
//! LinExp functions) to astronomical light curves using various optimization algorithms.
//! The fitting process uses weighted least squares with optional Bayesian priors.
//!
//! # Parameter Spaces
//!
//! The fitting infrastructure uses three distinct parameter spaces to improve numerical
//! conditioning and allow for constrained optimization:
//!
//! ## 1. External (Original) Parameters
//!
//! These are the physical parameters as understood by users and as reported in the output.
//! For example, in `BazinFit`:
//! - `A`: amplitude in flux/magnitude units
//! - `B`: baseline in flux/magnitude units
//! - `t0`: reference time in days (or original time units)
//! - `tau_rise`, `tau_fall`: timescales in days
//!
//! **Priors are specified in this space**, as users think in physical units.
//!
//! ## 2. Dimensionless Parameters
//!
//! External parameters normalized by data scale factors for numerical stability.
//! The transformation is defined by [`FitParametersOriginalDimLessTrait`]:
//!
//! ```text
//! orig_to_dimensionless(norm_data, orig) → dimensionless
//! dimensionless_to_orig(norm_data, dimensionless) → orig
//! ```
//!
//! For example, time parameters are divided by `t_std`, magnitude parameters by `m_std`.
//! This ensures all parameters are O(1) regardless of the input data scale.
//!
//! ## 3. Internal Parameters
//!
//! The actual parameters seen by the optimizer. Some models require parameter constraints
//! (e.g., positive timescales) which are enforced via transformations. The transformation
//! is defined by [`FitParametersInternalDimlessTrait`]:
//!
//! ```text
//! dimensionless_to_internal(dimensionless) → internal
//! internal_to_dimensionless(internal) → dimensionless
//! ```
//!
//! For example, `BazinFit` uses `internal_to_dimensionless` to apply `abs()` to amplitude
//! and timescale parameters, ensuring they remain positive.
//!
//! ## Full Transformation Chain
//!
//! The complete transformation from external to internal (and vice versa) is:
//!
//! ```text
//! External ←→ Dimensionless ←→ Internal
//!     ↑           ↑               ↑
//!     |           |               |
//!   priors    normalized       optimizer
//!   output     for O(1)        works here
//!              values
//! ```
//!
//! Convenience methods combine these:
//! - [`FitParametersInternalExternalTrait::convert_to_internal`]: external → internal
//! - [`FitParametersInternalExternalTrait::convert_to_external`]: internal → external
//!
//! # Fitting Workflow
//!
//! The [`fit_eval!`](crate::fit_eval) macro implements the standard fitting workflow:
//!
//! 1. **Normalize data**: Create [`NormalizedData`](data::NormalizedData) from the time series,
//!    which stores `(t - t_mean) / t_std`, `(m - m_mean) / m_std`, and scaled weights.
//!
//! 2. **Transform initial guess and bounds** to internal parameter space using
//!    `convert_to_internal`.
//!
//! 3. **Create transformed prior**: Wrap the user-specified prior (in external space) with
//!    [`TransformedLnPrior`](prior::ln_prior::TransformedLnPrior) that applies parameter
//!    transformation before evaluation.
//!
//! 4. **Run optimizer**: Call the selected [`CurveFitAlgorithm`] with:
//!    - Normalized data (t, m, inv_err arrays)
//!    - Initial guess in internal space
//!    - Bounds in internal space
//!    - Model function (operates in internal space)
//!    - Model derivatives (operates in internal space)
//!    - Transformed prior (takes internal params, transforms to external, evaluates prior)
//!
//! 5. **Transform results** back to external space using `convert_to_external`.
//!
//! # Model and Derivatives
//!
//! Models implement [`FitModelTrait`](evaluator::FitModelTrait):
//!
//! ```text
//! fn model(t: T, internal_params: &[U; NPARAMS]) -> U
//! ```
//!
//! The model receives **internal** parameters and typically transforms them to dimensionless
//! (via `internal_to_dimensionless`) to compute the model value. This is because the
//! normalized data has `t` in dimensionless units.
//!
//! Derivatives (via [`FitDerivalivesTrait`](evaluator::FitDerivalivesTrait)) must return:
//!
//! ```text
//! ∂model/∂(internal_params)
//! ```
//!
//! This requires applying the chain rule through the `internal_to_dimensionless` transformation.
//!
//! # Prior Gradients and the Chain Rule
//!
//! **IMPORTANT**: Priors are specified in external parameter space, but the optimizer works
//! in internal parameter space. When computing the gradient of the log-posterior, we need:
//!
//! ```text
//! ∂(ln_prior)/∂(internal) = ∂(ln_prior)/∂(external) × ∂(external)/∂(internal)
//! ```
//!
//! The Jacobian `∂(external)/∂(internal)` comes from the composition of two transformations:
//!
//! 1. `internal_to_dimensionless`: For `BazinFit`, this is `diag(sign(a), 1, 1, sign(τ_r), sign(τ_f))`
//!    due to the `abs()` transformation.
//!
//! 2. `dimensionless_to_orig`: This is `diag(m_std, m_std, t_std, t_std, t_std)` plus offsets
//!    (which don't affect the Jacobian).
//!
//! The full Jacobian is the product of these diagonal matrices.
//!
//! ## Current Status
//!
//! **BUG**: The current implementation in [`TransformedLnPrior`](prior::ln_prior::TransformedLnPrior)
//! does NOT apply the Jacobian to the prior gradient. The comment there is incorrect—it claims
//! that `internal_to_dimensionless` is identity for current use cases, but this is false for
//! `BazinFit`, `VillarFit`, and `LinExpFit` which all use `abs()` transformations.
//!
//! The gradient returned by `ln_prior()` for `TransformedLnPrior` is:
//! ```text
//! ∂(ln_prior)/∂(external)  ← WRONG, should be ∂(ln_prior)/∂(internal)
//! ```
//!
//! This causes incorrect gradient-based optimization (NUTS sampler) when priors are used
//! with models that have non-identity `internal_to_dimensionless` transformations.
//!
//! # Curve Fit Algorithms
//!
//! - [`McmcCurveFit`]: Ensemble MCMC sampler. Does not use derivatives. Supports priors.
//! - [`NutsCurveFit`]: NUTS Hamiltonian Monte Carlo. Uses derivatives. Supports priors.
//!   **Currently affected by the gradient bug above.**
//! - [`LmsderCurveFit`] (requires `gsl`): Levenberg-Marquardt. Uses derivatives. Ignores priors.
//! - [`CeresCurveFit`] (requires `ceres`): Trust-region. Uses derivatives. Ignores priors.
//!
//! [`FitParametersOriginalDimLessTrait`]: evaluator::FitParametersOriginalDimLessTrait
//! [`FitParametersInternalDimlessTrait`]: evaluator::FitParametersInternalDimlessTrait
//! [`FitParametersInternalExternalTrait`]: evaluator::FitParametersInternalExternalTrait

mod bounds;

#[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
pub mod ceres;
#[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
pub use ceres::CeresCurveFit;

#[cfg(any(
    feature = "gsl",
    any(feature = "ceres-source", feature = "ceres-system")
))]
mod constants;

pub mod curve_fit;
pub use curve_fit::{CurveFitAlgorithm, CurveFitResult, CurveFitTrait};

pub mod data;

pub mod evaluator;

#[cfg(feature = "gsl")]
pub mod lmsder;
#[cfg(feature = "gsl")]
pub use lmsder::LmsderCurveFit;

pub mod mcmc;
pub use mcmc::McmcCurveFit;

#[cfg(feature = "nuts")]
pub mod nuts;
#[cfg(feature = "nuts")]
pub use nuts::NutsCurveFit;

pub mod prior;
pub use prior::ln_prior::LnPrior;
pub use prior::ln_prior_1d::LnPrior1D;

#[cfg(test)]
pub trait HyperdualFloat: hyperdual::Float {
    fn half() -> Self;
    fn two() -> Self;
}
#[cfg(test)]
impl<T> HyperdualFloat for T
where
    T: hyperdual::Float,
{
    #[inline]
    fn half() -> Self {
        Self::from(0.5).unwrap()
    }

    #[inline]
    fn two() -> Self {
        Self::from(2.0).unwrap()
    }
}
#[cfg(not(test))]
pub trait HyperdualFloat: crate::Float {}
#[cfg(not(test))]
impl<T> HyperdualFloat for T where T: crate::Float {}

pub trait LikeFloat:
    HyperdualFloat + std::ops::AddAssign<Self> + std::ops::MulAssign<Self> + Sized
{
    fn logistic(x: Self) -> Self {
        (Self::one() + Self::exp(-x)).recip()
    }
}

impl<T> LikeFloat for T where
    T: HyperdualFloat + std::ops::AddAssign<Self> + std::ops::MulAssign<Self>
{
}
