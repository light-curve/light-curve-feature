use crate::nl_fit::constants::PARAMETER_TOLERANCE;
use crate::nl_fit::curve_fit::{CurveFitResult, CurveFitTrait};
use crate::nl_fit::data::Data;

use ceres_solver::{CurveFitProblem1D, CurveFunctionType, LossFunction, SolverOptions};
use ndarray::Zip;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::rc::Rc;

/// Ceres-Solver non-linear least-squares wrapper
///
/// Requires `ceres` Cargo feature
///
/// Non-linear squares-based light-curve fitters. It requires the function Jacobean. It supports
/// boundaries, but not priors.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Ceres")]
pub struct CeresCurveFit {
    niterations: u16,
    loss_factor: Option<f64>,
}

impl CeresCurveFit {
    /// Create a new [CeresCurveFit].
    ///
    /// # Arguments
    /// - `niterations`: number of iterations
    /// - `loss_factor`: if specified, use Huber loss function with the given factor to transform
    ///   the squared norm of the residuals. This is useful to reduce the influence of outliers.
    pub fn new(niterations: u16, loss_factor: Option<f64>) -> Self {
        if let Some(loss_factor) = loss_factor {
            assert!(loss_factor > 0.0, "loss_factor must be positive");
        }
        Self {
            niterations,
            loss_factor,
        }
    }

    #[inline]
    pub fn default_niterations() -> u16 {
        10
    }

    #[inline]
    pub fn default_loss_factor() -> Option<f64> {
        None
    }
}

impl Default for CeresCurveFit {
    fn default() -> Self {
        Self::new(Self::default_niterations(), Self::default_loss_factor())
    }
}

impl CurveFitTrait for CeresCurveFit {
    fn curve_fit<F, DF, LP, const NPARAMS: usize>(
        &self,
        ts: Rc<Data<f64>>,
        x0: &[f64; NPARAMS],
        bounds: (&[f64; NPARAMS], &[f64; NPARAMS]),
        model: F,
        derivatives: DF,
        _ln_prior: LP,
    ) -> CurveFitResult<f64, NPARAMS>
    where
        F: 'static + Clone + Fn(f64, &[f64; NPARAMS]) -> f64,
        DF: 'static + Clone + Fn(f64, &[f64; NPARAMS], &mut [f64; NPARAMS]),
        LP: Clone + Fn(&[f64; NPARAMS]) -> f64,
    {
        let func: CurveFunctionType = {
            let model = model.clone();
            Box::new(move |t, parameters, y, jacobians| {
                let parameters = parameters.try_into().unwrap();
                *y = model(t, parameters);
                if !y.is_finite() {
                    *y = f64::MAX.sqrt();
                    return false;
                }
                if let Some(jacobians) = jacobians {
                    let jacobians: &mut [_; NPARAMS] = jacobians.try_into().unwrap();
                    let der = {
                        let mut der = [0.0; NPARAMS];
                        derivatives(t, parameters, &mut der);
                        der
                    };
                    for (input, output) in der.into_iter().zip(jacobians.iter_mut()) {
                        if let Some(output) = output {
                            if !input.is_finite() {
                                return false;
                            }
                            *output = input;
                        }
                    }
                }
                true
            })
        };

        let lower_bounds: Vec<_> = bounds.0.iter().map(|&v| Some(v)).collect();
        let upper_bounds: Vec<_> = bounds.1.iter().map(|&v| Some(v)).collect();

        let options = SolverOptions::builder()
            .parameter_tolerance(PARAMETER_TOLERANCE)
            .max_num_iterations(self.niterations as i32)
            .build()
            .unwrap();

        let mut problem_builder = CurveFitProblem1D::builder()
            .x(ts.t.as_slice().unwrap())
            .y(ts.m.as_slice().unwrap())
            .inverse_error(ts.inv_err.as_slice().unwrap())
            .func(func)
            .parameters(x0)
            .lower_bounds(&lower_bounds)
            .upper_bounds(&upper_bounds);
        if let Some(loss_factor) = self.loss_factor {
            problem_builder = problem_builder.loss(LossFunction::cauchy(loss_factor));
        };
        let solution = problem_builder.build().unwrap().solve(&options);
        let x = solution.parameters.try_into().unwrap();
        let success = solution.summary.is_solution_usable();

        let reduced_chi2 = Zip::from(&ts.t)
            .and(&ts.m)
            .and(&ts.inv_err)
            .fold(0.0, |acc, &t, &m, &inv_err| {
                acc + ((model(t, &x) - m) * inv_err).powi(2)
            })
            / (ts.t.len() - NPARAMS) as f64;
        CurveFitResult {
            x,
            reduced_chi2,
            success,
        }
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    use approx::assert_abs_diff_eq;
    use ndarray::Array1;
    use rand::prelude::*;
    use rand_distr::StandardNormal;

    fn nonlinear_func(t: f64, param: &[f64; 3]) -> f64 {
        param[1] * f64::exp(-param[0] * t) * t.powi(2) + param[2]
    }

    fn nonlinear_func_derivatives(t: f64, param: &[f64; 3], derivatives: &mut [f64; 3]) {
        derivatives[0] = -param[1] * f64::exp(-param[2] * t) * t.powi(3);
        derivatives[1] = f64::exp(-param[0] * t) * t.powi(2);
        derivatives[2] = 1.0;
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
        println!(
            "t = {:?}\ny = {:?}\ninv_err = {:?}",
            t.as_slice().unwrap(),
            y.as_slice().unwrap(),
            inv_err.as_slice().unwrap()
        );
        let ts = Rc::new(Data { t, m: y, inv_err });

        let fitter = CeresCurveFit::new(14, None);
        let result = fitter.curve_fit(
            ts,
            &param_init,
            (&[f64::NEG_INFINITY; 3], &[f64::INFINITY; 3]),
            nonlinear_func,
            nonlinear_func_derivatives,
            nonlinear_func_dump_ln_prior,
        );

        // curve_fit(lambda x, a, b, c: b * np.exp(-a * x) * x**2 + c, xdata=t, ydata=y, sigma=1/np.array(inv_err), p0=[1, 1, 1], xtol=1e-6)
        let desired = [0.76007721, 2.0225076, 0.49238112];

        // Not as good as for LMSDER
        assert_abs_diff_eq!(
            &result.x[..],
            &param_true[..],
            epsilon = NOISE / (N as f64).sqrt()
        );
        assert_abs_diff_eq!(&result.x[..], &desired[..], epsilon = 0.04);
    }
}
