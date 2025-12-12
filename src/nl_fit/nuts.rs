use crate::nl_fit::bounds::within_bounds;
use crate::nl_fit::curve_fit::{CurveFitAlgorithm, CurveFitResult, CurveFitTrait};
use crate::nl_fit::data::Data;

use ndarray::Zip;
use nuts_rs::{Chain, CpuLogpFunc, CpuMath, DiagGradNutsSettings, LogpError, Settings};
use nuts_storable::HasDims;
use rand::rng;
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
    LP: Clone + Fn(&[f64; NPARAMS]) -> f64,
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

        // Add prior
        let lnprior = (self.ln_prior)(&params_array);

        // Gradient is d(lnlike + lnprior)/d(params)
        // We have grad_array = d(-chi^2)/d(params) = -2 * d(lnlike)/d(params)
        // So d(lnlike)/d(params) = -grad_array / 2
        // Note: This implementation does not include the gradient of the prior.
        // For non-uniform priors, this may lead to less efficient sampling,
        // but the sampler will still converge to the correct distribution.
        for i in 0..NPARAMS {
            grad[i] = -grad_array[i] / 2.0;
        }

        Ok(lnlike + lnprior)
    }

    fn expand_vector<R: rand::Rng + ?Sized>(
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
        LP: Clone + Fn(&[f64; NPARAMS]) -> f64,
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

        let mut settings = DiagGradNutsSettings::default();
        settings.num_tune = self.num_tune as u64;
        settings.num_draws = self.num_draws as u64;

        let mut rng = rng();
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
