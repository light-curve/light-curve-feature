use crate::nl_fit::curve_fit::{CurveFitAlgorithm, CurveFitResult, CurveFitTrait};
use crate::nl_fit::data::Data;

use cobyla::{Func, RhoBeg, StopTols, minimize};
use ndarray::Zip;
use ordered_float::NotNan;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::rc::Rc;

/// COBYLA (Constrained Optimization BY Linear Approximations) non-linear least-squares wrapper
///
/// COBYLA is a derivative-free optimization algorithm that can handle constraints. Unlike LMSDER
/// and Ceres, it doesn't require function derivatives (Jacobian). It supports boundaries through
/// constraints but doesn't support priors directly. COBYLA is particularly useful as a drop-in
/// replacement for MCMC when you need constraint-aware optimization without derivatives.
///
/// Optionally, if `fine_tuning_algorithm` is `Some`, it sends the best guess from COBYLA to the
/// next optimization as an initial guess and returns its result. This allows chaining optimizers
/// similar to MCMC's fine-tuning capability.
///
/// The algorithm works by building linear approximations to the objective and constraint functions
/// and is described in M.J.D. Powell's 1994 paper "A direct search optimization method that models
/// the objective and constraint functions by linear interpolation".
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename = "Cobyla")]
pub struct CobylaCurveFit {
    pub niterations: u32,
    pub rhobeg: NotNan<f64>,
    pub ftol_rel: NotNan<f64>,
    pub fine_tuning_algorithm: Option<Box<CurveFitAlgorithm>>,
}

impl CobylaCurveFit {
    /// Create a new [CobylaCurveFit].
    ///
    /// # Arguments
    /// - `niterations`: maximum number of function evaluations
    /// - `rhobeg`: initial change to parameters (controls initial simplex size)
    /// - `ftol_rel`: relative tolerance on function value for convergence
    /// - `fine_tuning_algorithm`: optional algorithm to refine COBYLA's result
    pub fn new(
        niterations: u32,
        rhobeg: f64,
        ftol_rel: f64,
        fine_tuning_algorithm: Option<CurveFitAlgorithm>,
    ) -> Self {
        assert!(niterations > 0, "niterations must be positive");
        assert!(rhobeg > 0.0, "rhobeg must be positive");
        assert!(rhobeg.is_finite(), "rhobeg must be finite");
        assert!(ftol_rel >= 0.0, "ftol_rel must be non-negative");
        assert!(ftol_rel.is_finite(), "ftol_rel must be finite");
        Self {
            niterations,
            rhobeg: NotNan::new(rhobeg).expect("rhobeg must be finite and not NaN"),
            ftol_rel: NotNan::new(ftol_rel).expect("ftol_rel must be finite and not NaN"),
            fine_tuning_algorithm: fine_tuning_algorithm.map(|x| x.into()),
        }
    }

    #[inline]
    pub fn default_niterations() -> u32 {
        1000
    }

    #[inline]
    pub fn default_rhobeg() -> f64 {
        0.5
    }

    #[inline]
    pub fn default_ftol_rel() -> f64 {
        1e-6
    }

    #[inline]
    pub fn default_fine_tuning_algorithm() -> Option<CurveFitAlgorithm> {
        None
    }
}

impl Default for CobylaCurveFit {
    fn default() -> Self {
        Self::new(
            Self::default_niterations(),
            Self::default_rhobeg(),
            Self::default_ftol_rel(),
            Self::default_fine_tuning_algorithm(),
        )
    }
}

impl CurveFitTrait for CobylaCurveFit {
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

        // Objective function: sum of squared weighted residuals (chi-squared)
        let objective = {
            let ts = ts.clone();
            let model = model.clone();
            move |x: &[f64], _user_data: &mut ()| -> f64 {
                // Safety: COBYLA guarantees that x has the same length as x0 (NPARAMS)
                let params: [f64; NPARAMS] = x.try_into().unwrap();
                let mut chi2 = 0.0;
                Zip::from(&ts.t)
                    .and(&ts.m)
                    .and(&ts.inv_err)
                    .for_each(|&t, &m, &inv_err| {
                        let residual = (model(t, &params) - m) * inv_err;
                        chi2 += residual * residual;
                    });
                chi2
            }
        };

        // Convert bounds to the format expected by COBYLA
        let cobyla_bounds: Vec<(f64, f64)> = bounds
            .0
            .iter()
            .zip(bounds.1.iter())
            .map(|(&lower, &upper)| (lower, upper))
            .collect();

        // No additional constraints beyond bounds
        let constraints: Vec<&dyn Func<()>> = vec![];

        // Set up stopping tolerances
        let stop_tol = StopTols {
            ftol_rel: self.ftol_rel.into(),
            ..StopTols::default()
        };

        // Run COBYLA optimization
        let result = minimize(
            objective,
            x0,
            &cobyla_bounds,
            &constraints,
            (),
            self.niterations as usize,
            RhoBeg::All(self.rhobeg.into()),
            Some(stop_tol),
        );

        match result {
            Ok((status, x_vec, chi2)) => {
                // Safety: COBYLA returns a vector with the same length as x0 (NPARAMS)
                let x: [f64; NPARAMS] = x_vec.try_into().unwrap();
                let reduced_chi2 = chi2 / ((nsamples - NPARAMS) as f64);
                let success = matches!(
                    status,
                    cobyla::SuccessStatus::Success
                        | cobyla::SuccessStatus::FtolReached
                        | cobyla::SuccessStatus::XtolReached
                );
                let cobyla_result = CurveFitResult {
                    x,
                    reduced_chi2,
                    success,
                };

                // Apply fine-tuning algorithm if provided
                match &self.fine_tuning_algorithm {
                    Some(fine_tuning_algorithm) => fine_tuning_algorithm.curve_fit(
                        ts,
                        &cobyla_result.x,
                        bounds,
                        model,
                        derivatives,
                        ln_prior,
                    ),
                    None => cobyla_result,
                }
            }
            Err((_status, x_vec, chi2)) => {
                // Safety: COBYLA returns a vector with the same length as x0 (NPARAMS)
                let x: [f64; NPARAMS] = x_vec.try_into().unwrap();
                let reduced_chi2 = chi2 / ((nsamples - NPARAMS) as f64);
                CurveFitResult {
                    x,
                    reduced_chi2,
                    success: false,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use ndarray::Array1;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    fn nonlinear_func(t: f64, param: &[f64; 3]) -> f64 {
        param[1] * f64::exp(-param[0] * t) * t.powi(2) + param[2]
    }

    fn nonlinear_func_derivatives(_t: f64, _param: &[f64; 3], _derivatives: &mut [f64; 3]) {
        // COBYLA doesn't use derivatives, but we need to provide a dummy function
    }

    fn nonlinear_func_dump_ln_prior(_param: &[f64; 3]) -> f64 {
        0.0
    }

    #[test]
    fn nonlinear() {
        const N: usize = 300;
        const NOISE: f64 = 0.5;

        let param_true = [0.75, 2.0, 0.5];
        let param_init = [1.0, 1.0, 1.0];

        let mut rng = StdRng::seed_from_u64(0);

        let t = Array1::linspace(0.0, 10.0, N);
        let y = t.mapv(|x| {
            let eps: f64 = rng.sample(StandardNormal);
            nonlinear_func(x, &param_true) + NOISE * eps
        });
        let inv_err: Array1<_> = vec![1.0 / NOISE; N].into();
        let ts = Rc::new(Data { t, m: y, inv_err });

        let fitter = CobylaCurveFit::new(2000, 0.5, 1e-6, None);
        let result = fitter.curve_fit(
            ts,
            &param_init,
            (&[0.0, 0.0, -10.0], &[10.0, 10.0, 10.0]),
            nonlinear_func,
            nonlinear_func_derivatives,
            nonlinear_func_dump_ln_prior,
        );

        println!("COBYLA result: {:?}", result.x);
        println!("Success: {}", result.success);
        println!("Reduced chi2: {}", result.reduced_chi2);

        // COBYLA is derivative-free and may not be as precise as gradient-based methods,
        // so we use a more relaxed tolerance
        assert!(result.success, "Optimization should succeed");
        assert_abs_diff_eq!(
            &result.x[..],
            &param_true[..],
            epsilon = NOISE * 2.0 / (N as f64).sqrt()
        );
    }

    #[test]
    fn simple_quadratic() {
        // Test with a simple quadratic function
        const N: usize = 50;

        let param_true = [1.0, 2.0, 0.5];
        let param_init = [0.5, 1.0, 0.0];

        let t = Array1::linspace(0.0, 5.0, N);
        let y = t.mapv(|x: f64| param_true[0] + param_true[1] * x + param_true[2] * x.powi(2));
        let inv_err: Array1<_> = vec![1.0; N].into();
        let ts = Rc::new(Data { t, m: y, inv_err });

        let quadratic_func = |t: f64, p: &[f64; 3]| p[0] + p[1] * t + p[2] * t.powi(2);
        let dummy_derivatives = |_t: f64, _p: &[f64; 3], _d: &mut [f64; 3]| {};
        let dummy_prior = |_p: &[f64; 3]| 0.0;

        let fitter = CobylaCurveFit::new(2000, 0.5, 1e-9, None);
        let result = fitter.curve_fit(
            ts,
            &param_init,
            (&[f64::NEG_INFINITY; 3], &[f64::INFINITY; 3]),
            quadratic_func,
            dummy_derivatives,
            dummy_prior,
        );

        println!("Quadratic result: {:?}", result.x);
        println!("Expected: {:?}", param_true);
        println!("Success: {}", result.success);
        println!("Reduced chi2: {}", result.reduced_chi2);

        // COBYLA is derivative-free and may not converge as tightly as gradient-based methods
        // Check that the solution is close enough for practical purposes
        assert_abs_diff_eq!(&result.x[..], &param_true[..], epsilon = 0.5);
        // For a perfect fit (no noise), chi2 should be very small
        assert!(
            result.reduced_chi2 < 0.1,
            "Chi2 should be small for perfect fit"
        );
    }
}
