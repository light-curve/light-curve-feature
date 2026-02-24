use crate::nl_fit::bounds::within_bounds;
use crate::nl_fit::curve_fit::{CurveFitAlgorithm, CurveFitResult, CurveFitTrait};
use crate::nl_fit::data::Data;
use crate::nl_fit::prior::ln_prior::LnPriorEvaluator;

use ndarray::Zip;
use nuts_rs::{Chain, CpuLogpFunc, CpuMath, DiagGradNutsSettings, LogpError, Settings};
use nuts_storable::HasDims;
use rand::SeedableRng;
use rand::rngs::StdRng;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt::Debug;
use std::rc::Rc;

/// NUTS (No-U-Turn Sampler) for non-linear least squares
///
/// A Hamiltonian Monte Carlo sampler that uses gradients to efficiently explore the parameter space.
/// It samples `num_tune + num_draws` iterations and chooses the guess corresponding to the minimum
/// sum of squared deviations (maximum likelihood). Optionally, if `fine_tuning_algorithm` is `Some`,
/// it sends this best guess to the next optimization as an initial guess and returns its result.
///
/// This method supports both boundaries and priors and requires the function derivatives.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename = "Nuts")]
pub struct NutsCurveFit {
    pub num_tune: u32,
    pub num_draws: u32,
    pub fine_tuning_algorithm: Option<Box<CurveFitAlgorithm>>,
}

impl NutsCurveFit {
    pub fn new(
        num_tune: u32,
        num_draws: u32,
        fine_tuning_algorithm: Option<CurveFitAlgorithm>,
    ) -> Self {
        Self {
            num_tune,
            num_draws,
            fine_tuning_algorithm: fine_tuning_algorithm.map(|x| x.into()),
        }
    }

    #[inline]
    pub fn default_num_tune() -> u32 {
        200
    }

    #[inline]
    pub fn default_num_draws() -> u32 {
        200
    }

    #[inline]
    pub fn default_fine_tuning_algorithm() -> Option<CurveFitAlgorithm> {
        None
    }
}

impl Default for NutsCurveFit {
    fn default() -> Self {
        Self::new(
            Self::default_num_tune(),
            Self::default_num_draws(),
            Self::default_fine_tuning_algorithm(),
        )
    }
}

#[derive(Debug)]
enum NutsLogpError {
    NonRecoverable,
}

impl std::fmt::Display for NutsLogpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            NutsLogpError::NonRecoverable => write!(f, "Non-recoverable error in logp calculation"),
        }
    }
}

impl std::error::Error for NutsLogpError {}

impl LogpError for NutsLogpError {
    fn is_recoverable(&self) -> bool {
        false
    }
}

struct LogpFunc<F, DF, LP, const NPARAMS: usize> {
    ts: Rc<Data<f64>>,
    model: F,
    derivatives: DF,
    ln_prior: LP,
    lower: [f64; NPARAMS],
    upper: [f64; NPARAMS],
}

impl<F, DF, LP, const NPARAMS: usize> HasDims for LogpFunc<F, DF, LP, NPARAMS> {
    fn dim_sizes(&self) -> HashMap<String, u64> {
        HashMap::from([
            ("unconstrained_parameter".to_string(), NPARAMS as u64),
            ("dim".to_string(), NPARAMS as u64),
        ])
    }
}

impl<F, DF, LP, const NPARAMS: usize> CpuLogpFunc for LogpFunc<F, DF, LP, NPARAMS>
where
    F: Clone + Fn(f64, &[f64; NPARAMS]) -> f64,
    DF: Clone + Fn(f64, &[f64; NPARAMS], &mut [f64; NPARAMS]),
    LP: LnPriorEvaluator<NPARAMS>,
{
    type LogpError = NutsLogpError;
    type FlowParameters = ();
    type ExpandedVector = Vec<f64>;

    fn dim(&self) -> usize {
        NPARAMS
    }

    fn logp(&mut self, params: &[f64], grad: &mut [f64]) -> Result<f64, Self::LogpError> {
        // Check boundaries
        let params_array: [f64; NPARAMS] = params
            .try_into()
            .map_err(|_| NutsLogpError::NonRecoverable)?;

        if !within_bounds(&params_array, &self.lower, &self.upper) {
            return Ok(f64::NEG_INFINITY);
        }

        // Calculate log-likelihood (negative chi-squared)
        let mut residual = 0.0;
        let mut grad_array = [0.0; NPARAMS];

        // Compute -chi^2/2 and its gradient
        Zip::from(&self.ts.t)
            .and(&self.ts.m)
            .and(&self.ts.inv_err)
            .for_each(|&t, &m, &inv_err| {
                let model_val = (self.model)(t, &params_array);
                let diff = model_val - m;
                residual += (inv_err * diff).powi(2);

                // Gradient of chi^2 with respect to parameters
                let mut model_grad = [0.0; NPARAMS];
                (self.derivatives)(t, &params_array, &mut model_grad);
                for i in 0..NPARAMS {
                    grad_array[i] += 2.0 * inv_err.powi(2) * diff * model_grad[i];
                }
            });

        let lnlike = -0.5 * residual;

        // Add prior and compute its gradient
        let mut prior_grad = [0.0; NPARAMS];
        let lnprior = self.ln_prior.ln_prior(&params_array, Some(&mut prior_grad));

        // Gradient is d(lnlike + lnprior)/d(params)
        // = d(lnlike)/d(params) + d(lnprior)/d(params)
        // = -0.5 * d(chi^2)/d(params) + d(lnprior)/d(params)
        for i in 0..NPARAMS {
            grad[i] = -0.5 * grad_array[i] + prior_grad[i];
        }

        Ok(lnlike + lnprior)
    }

    fn expand_vector<R: rand::RngExt + ?Sized>(
        &mut self,
        _rng: &mut R,
        array: &[f64],
    ) -> Result<Self::ExpandedVector, nuts_rs::CpuMathError> {
        Ok(array.to_vec())
    }
}

impl CurveFitTrait for NutsCurveFit {
    fn curve_fit<F, DF, LP, const NPARAMS: usize>(
        &self,
        ts: Rc<Data<f64>>,
        x0: &[f64; NPARAMS],
        bounds: (&[f64; NPARAMS], &[f64; NPARAMS]),
        model: F,
        derivatives: DF,
        ln_prior: LP,
    ) -> CurveFitResult<f64, NPARAMS>
    where
        F: 'static + Clone + Fn(f64, &[f64; NPARAMS]) -> f64,
        DF: 'static + Clone + Fn(f64, &[f64; NPARAMS], &mut [f64; NPARAMS]),
        LP: LnPriorEvaluator<NPARAMS>,
    {
        let nsamples = ts.t.len();

        let logp_func = LogpFunc {
            ts: ts.clone(),
            model: model.clone(),
            derivatives: derivatives.clone(),
            ln_prior: ln_prior.clone(),
            lower: *bounds.0,
            upper: *bounds.1,
        };

        let math = CpuMath::new(logp_func);

        let settings = DiagGradNutsSettings {
            num_tune: self.num_tune as u64,
            num_draws: self.num_draws as u64,
            ..Default::default()
        };

        let mut rng = StdRng::seed_from_u64(0);
        let mut sampler = settings.new_chain(0, math, &mut rng);

        // Set initial position
        sampler.set_position(&x0[..]).unwrap_or_else(|e| {
            panic!(
                "Failed to set initial position for NUTS sampler. \
                     This may be due to invalid parameters or boundary violations. \
                     Initial parameters: {:?}, Error: {:?}",
                x0, e
            )
        });

        // Collect samples
        let mut best_x = *x0;
        let mut best_lnprob = f64::NEG_INFINITY;

        for _ in 0..(self.num_tune + self.num_draws) {
            match sampler.expanded_draw() {
                Ok((draw, _expanded, stats, _progress)) => {
                    // Convert draw to array
                    let params: [f64; NPARAMS] = Vec::from(draw)
                        .try_into()
                        .expect("Failed to convert draw to array");

                    // Use the log probability from the sampler stats
                    let lnprob = stats.logp;

                    if lnprob > best_lnprob {
                        best_x = params;
                        best_lnprob = lnprob;
                    }
                }
                Err(_) => {
                    // If sampling fails, continue with the best parameters found so far
                    break;
                }
            }
        }

        match self.fine_tuning_algorithm.as_ref() {
            Some(algo) => algo.curve_fit(ts, &best_x, bounds, model, derivatives, ln_prior),
            None => {
                // Calculate chi-squared for the best parameters
                let mut residual = 0.0;
                Zip::from(&ts.t)
                    .and(&ts.m)
                    .and(&ts.inv_err)
                    .for_each(|&t, &m, &inv_err| {
                        residual += (inv_err * (model(t, &best_x) - m)).powi(2);
                    });
                let reduced_chi2 = residual / ((nsamples - NPARAMS) as f64);

                CurveFitResult {
                    x: best_x,
                    reduced_chi2,
                    success: true,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nl_fit::data::NormalizedData;
    use crate::nl_fit::evaluator::{
        FitParametersInternalDimlessTrait, FitParametersInternalExternalTrait,
        FitParametersOriginalDimLessTrait,
    };
    use crate::nl_fit::prior::ln_prior::LnPrior;
    use crate::nl_fit::prior::ln_prior_1d::LnPrior1D;
    use crate::time_series::TimeSeries;
    use approx::assert_relative_eq;
    use ndarray::Array1;
    use std::rc::Rc;

    /// A simple constant model for testing: f(t) = |c|
    ///
    /// Uses sign transformation: dimensionless = |internal|
    /// This tests that the Jacobian (sign factor) is correctly applied.
    ///
    /// For simplicity, this model operates directly in dimensionless space,
    /// with external = dimensionless (no additional scaling).
    struct SimpleConstantModel;

    impl FitParametersInternalDimlessTrait<f64, 1> for SimpleConstantModel {
        fn dimensionless_to_internal(params: &[f64; 1]) -> [f64; 1] {
            *params
        }

        fn internal_to_dimensionless(params: &[f64; 1]) -> [f64; 1] {
            [params[0].abs()] // Apply abs() transformation
        }
    }

    impl FitParametersOriginalDimLessTrait<1> for SimpleConstantModel {
        fn orig_to_dimensionless(_norm_data: &NormalizedData<f64>, orig: &[f64; 1]) -> [f64; 1] {
            // No scaling - external = dimensionless for this simple model
            *orig
        }

        fn dimensionless_to_orig(_norm_data: &NormalizedData<f64>, norm: &[f64; 1]) -> [f64; 1] {
            // No scaling - external = dimensionless for this simple model
            *norm
        }
    }

    impl FitParametersInternalExternalTrait<1> for SimpleConstantModel {
        fn jacobian_internal_to_external(
            _norm_data: &NormalizedData<f64>,
            internal: &[f64; 1],
        ) -> [f64; 1] {
            // external = |internal|
            // d(external)/d(internal) = sign(internal)
            [internal[0].signum()]
        }
    }

    /// Test the prior gradient fix with an analytical solution.
    ///
    /// Problem setup:
    /// - Model: f(t) = |c| (constant with abs transformation)
    /// - Data: n observations all equal to y_target
    /// - Prior: c ~ N(μ_prior, σ_prior²) in external space (positive)
    ///
    /// For a constant model with Gaussian prior and Gaussian likelihood:
    /// - Likelihood: ∝ exp(-n(c - y_target)²/(2σ²))
    /// - Prior: ∝ exp(-(c - μ_prior)²/(2σ_prior²))
    ///
    /// Posterior mean: (n*y_target/σ² + μ_prior/σ_prior²) / (n/σ² + 1/σ_prior²)
    #[test]
    fn test_nuts_with_prior_analytical_solution() {
        // Create simple normalized data directly
        // All observations have the same value in dimensionless space
        const N: usize = 20;
        let y_target = 1.0; // Target value in dimensionless space
        let obs_std = 0.5; // Observation standard deviation

        let t_arr = Array1::from_vec((0..N).map(|i| i as f64).collect());
        let m_arr = Array1::from_elem(N, y_target);
        let inv_err_arr = Array1::from_elem(N, 1.0 / obs_std); // inv_err = 1/σ

        let data = Rc::new(Data {
            t: t_arr,
            m: m_arr,
            inv_err: inv_err_arr,
        });

        // Create a dummy TimeSeries just to get NormalizedData for the prior transformation
        let t_vec: Vec<f64> = (0..3).map(|i| i as f64).collect();
        let m_vec: Vec<f64> = vec![1.0, 1.0, 1.0];
        let w_vec: Vec<f64> = vec![1.0, 1.0, 1.0];
        let mut ts = TimeSeries::new(&t_vec, &m_vec, &w_vec);
        let norm_data = NormalizedData::<f64>::from_ts(&mut ts);

        // Prior parameters in external space
        let prior_mean = 1.5; // Prior pulls toward 1.5
        let prior_std = 0.3;
        let prior_var = prior_std * prior_std;
        let obs_var = obs_std * obs_std;

        // Analytical posterior parameters for Gaussian-Gaussian conjugate case
        let posterior_var = 1.0 / ((N as f64) / obs_var + 1.0 / prior_var);
        let posterior_mean =
            posterior_var * ((N as f64) * y_target / obs_var + prior_mean / prior_var);

        // Create prior in external space
        let prior: LnPrior<1> = LnPrior::ind_components([LnPrior1D::normal(prior_mean, prior_std)]);

        // Transform prior to work with internal parameters
        let transformed_prior =
            prior.with_fit_parameters_transformation::<SimpleConstantModel>(&norm_data);

        // Model: f(t) = |internal[0]|
        let model = |_t: f64, params: &[f64; 1]| -> f64 { params[0].abs() };

        // Derivatives: d(|x|)/dx = sign(x)
        let derivatives = |_t: f64, params: &[f64; 1], jac: &mut [f64; 1]| {
            jac[0] = params[0].signum();
        };

        // Initial guess: start at a reasonable positive value
        let init_internal = [y_target];

        // Bounds: allow both positive and negative internal values
        let lower_internal = [0.01]; // Keep external positive
        let upper_internal = [5.0];

        // Run NUTS
        let nuts = NutsCurveFit::new(500, 500, None);
        let result = nuts.curve_fit(
            data,
            &init_internal,
            (&lower_internal, &upper_internal),
            model,
            derivatives,
            transformed_prior,
        );

        // Convert result to external space
        let result_external = SimpleConstantModel::convert_to_external(&norm_data, &result.x);

        println!("Data mean: {}", y_target);
        println!("Prior mean: {}", prior_mean);
        println!("Posterior mean (analytical): {}", posterior_mean);
        println!("NUTS result (external): {:?}", result_external);
        println!("Posterior std (analytical): {}", posterior_var.sqrt());

        // Check that the result is within 3 posterior standard deviations
        let tolerance = 3.0 * posterior_var.sqrt();
        assert!(
            (result_external[0] - posterior_mean).abs() < tolerance,
            "NUTS result {} is too far from analytical posterior mean {} (tolerance: {})",
            result_external[0],
            posterior_mean,
            tolerance
        );

        // Check that result is between data mean and prior mean
        // (posterior should be a weighted average)
        let min_val = y_target.min(prior_mean);
        let max_val = y_target.max(prior_mean);
        assert!(
            result_external[0] >= min_val - tolerance && result_external[0] <= max_val + tolerance,
            "NUTS result {} should be between data mean {} and prior mean {} (with tolerance {})",
            result_external[0],
            y_target,
            prior_mean,
            tolerance
        );
    }

    /// Test that the gradient is correctly computed for the transformed prior
    /// with sign transformation, using numerical differentiation.
    #[test]
    fn test_transformed_prior_gradient_with_sign() {
        // Create dummy normalized data
        let t_vec: Vec<f64> = vec![1.0, 2.0, 3.0];
        let m_vec: Vec<f64> = vec![1.0, 1.0, 1.0];
        let w_vec: Vec<f64> = vec![1.0, 1.0, 1.0];
        let mut ts = TimeSeries::new(&t_vec, &m_vec, &w_vec);
        let norm_data = NormalizedData::<f64>::from_ts(&mut ts);

        // Prior in external space: N(1.0, 0.5²)
        let prior: LnPrior<1> = LnPrior::ind_components([LnPrior1D::normal(1.0, 0.5)]);
        let transformed_prior =
            prior.with_fit_parameters_transformation::<SimpleConstantModel>(&norm_data);

        // Test with both positive and negative internal parameters
        // The key insight: for external = |internal|, the gradient should flip sign
        // when internal is negative
        for &internal_val in &[-2.0, -1.0, -0.5, 0.5, 1.0, 2.0] {
            let params = [internal_val];
            let mut analytical_grad = [0.0];
            let ln_p = transformed_prior.ln_prior(&params, Some(&mut analytical_grad));

            if !ln_p.is_finite() {
                continue;
            }

            // Numerical gradient using central difference
            let eps = 1e-6;
            let ln_p_plus = transformed_prior.ln_prior(&[internal_val + eps], None);
            let ln_p_minus = transformed_prior.ln_prior(&[internal_val - eps], None);
            let numerical_grad = (ln_p_plus - ln_p_minus) / (2.0 * eps);

            println!(
                "internal={:.2}: analytical_grad={:.6}, numerical_grad={:.6}",
                internal_val, analytical_grad[0], numerical_grad
            );

            assert_relative_eq!(
                analytical_grad[0],
                numerical_grad,
                epsilon = 1e-4,
                max_relative = 1e-3
            );

            // Verify the sign relationship:
            // For external = |internal|, if the prior gradient w.r.t. external is g,
            // then the gradient w.r.t. internal should be g * sign(internal)
            // This means the gradient should have opposite signs for opposite internal values
        }

        // Additional check: gradients at symmetric points (away from mode) should have opposite signs
        let mut grad_pos = [0.0];
        let mut grad_neg = [0.0];
        // Use 0.5 instead of 1.0 (the mode) to get non-zero gradients
        transformed_prior.ln_prior(&[0.5], Some(&mut grad_pos));
        transformed_prior.ln_prior(&[-0.5], Some(&mut grad_neg));

        println!("Gradient at internal=0.5: {}", grad_pos[0]);
        println!("Gradient at internal=-0.5: {}", grad_neg[0]);

        // The gradients should have opposite signs (because of the sign(internal) factor)
        // At internal=0.5: external=0.5, prior gradient w.r.t external is positive (pulling toward 1.0)
        // So gradient w.r.t internal = positive * sign(0.5) = positive
        // At internal=-0.5: external=0.5, prior gradient w.r.t external is positive
        // So gradient w.r.t internal = positive * sign(-0.5) = negative
        assert!(
            grad_pos[0] * grad_neg[0] < 0.0,
            "Gradients at symmetric points should have opposite signs: {} vs {}",
            grad_pos[0],
            grad_neg[0]
        );
    }

    /// Test that the prior value is the same for +internal and -internal
    /// (since external = |internal| makes them equivalent)
    #[test]
    fn test_transformed_prior_symmetry() {
        let t_vec: Vec<f64> = vec![1.0, 2.0, 3.0];
        let m_vec: Vec<f64> = vec![1.0, 1.0, 1.0];
        let w_vec: Vec<f64> = vec![1.0, 1.0, 1.0];
        let mut ts = TimeSeries::new(&t_vec, &m_vec, &w_vec);
        let norm_data = NormalizedData::<f64>::from_ts(&mut ts);

        let prior: LnPrior<1> = LnPrior::ind_components([LnPrior1D::normal(1.0, 0.5)]);
        let transformed_prior =
            prior.with_fit_parameters_transformation::<SimpleConstantModel>(&norm_data);

        // Test that ln_prior(+x) == ln_prior(-x) for the same |x|
        for &x in &[0.5, 1.0, 1.5, 2.0] {
            let ln_p_pos = transformed_prior.ln_prior(&[x], None);
            let ln_p_neg = transformed_prior.ln_prior(&[-x], None);

            assert_relative_eq!(ln_p_pos, ln_p_neg, epsilon = 1e-10);
        }
    }
}
