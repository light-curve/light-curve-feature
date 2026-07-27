//! Levenberg-Marquardt fitting of the [`RainbowModel`].
//!
//! Every physical parameter is fit through an unconstrained "internal" optimizer variable `u`,
//! reparametrized to always land inside the parameter's `(lower, upper)` box bounds via a
//! logistic (sigmoid) map: `external = lower + (upper - lower) / (1 + exp(-u))`. This is the same
//! idea Python's `iminuit`/MINUIT uses internally for bounded parameters (and the same spirit as
//! this crate's own [`nl_fit`](crate::nl_fit) module's internal/external parameter-space split),
//! but expressed as a single transform instead of separate data-normalization and sign-only
//! reparametrization steps: since `u`'s scale only depends on where the true value sits within
//! its bound interval (not on the parameter's physical magnitude), no separate data-normalization
//! ("dimensionless") layer is needed for optimizer conditioning.
//!
//! This is why Rainbow does not use [`nl_fit`](crate::nl_fit)/[`CurveFitAlgorithm`](crate::nl_fit::CurveFitAlgorithm):
//! that machinery is built around a compile-time-fixed `const NPARAMS: usize` (fixed-size
//! `[f64; NPARAMS]` arrays throughout), whereas Rainbow's parameter count varies with the chosen
//! bolometric/temperature/spectral term combination. The optimizer here is `levenberg-marquardt`
//! (a pure-Rust, `nalgebra`-based Levenberg-Marquardt implementation) working with dynamically
//! (`Dyn`)-sized vectors/matrices instead.
//!
//! Bounds themselves are physical-unit and mostly data-dependent (e.g. `rise_time`'s upper bound
//! scales with the light curve's time span); they're computed once per fit from
//! `super::terms::{Bolometric, Temperature, Spectral}::bounds()`.

use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use nalgebra::{DMatrix, DVector, Dyn, OMatrix, storage::Owned};

use super::model::RainbowModel;

/// `(external, d(external)/du)` for the optimizer's unconstrained internal value `u`,
/// logistically mapped into `(lower, upper)`.
fn bounded_transform(u: f64, lower: f64, upper: f64) -> (f64, f64) {
    let s = 1.0 / (1.0 + (-u).exp());
    let external = lower + (upper - lower) * s;
    let dext_du = (upper - lower) * s * (1.0 - s);
    (external, dext_du)
}

/// Inverse of [`bounded_transform`] (the logit function), used to convert a physical-unit
/// initial guess into the optimizer's internal space. Clamped away from the exact bounds to
/// avoid `+-inf`.
fn inverse_bounded_transform(external: f64, lower: f64, upper: f64) -> f64 {
    let frac = ((external - lower) / (upper - lower)).clamp(1e-9, 1.0 - 1e-9);
    (frac / (1.0 - frac)).ln()
}

fn internal_to_external(u: &DVector<f64>, bounds: &[(f64, f64)]) -> (Vec<f64>, Vec<f64>) {
    let n = bounds.len();
    let mut external = vec![0.0; n];
    let mut dext_du = vec![0.0; n];
    for k in 0..n {
        let (lower, upper) = bounds[k];
        let (e, d) = bounded_transform(u[k], lower, upper);
        external[k] = e;
        dext_du[k] = d;
    }
    (external, dext_du)
}

fn external_to_internal(external: &[f64], bounds: &[(f64, f64)]) -> DVector<f64> {
    DVector::from_iterator(
        bounds.len(),
        external
            .iter()
            .zip(bounds)
            .map(|(&e, &(lower, upper))| inverse_bounded_transform(e, lower, upper)),
    )
}

/// Parameter covariance matrix from the Gauss-Newton approximation `inv(JᵀJ)` at the converged
/// solution, where `J` is `d(residual)/d(external param)` (residuals already weighted by
/// `1/flux_err`). Mirrors Python's `_lsq_covariance`: the same construction scipy's `curve_fit`
/// uses with `absolute_sigma=True` -- no rescaling by `reduced_chi2`.
///
/// Cheap: this is one extra pass over the data building an `n_points x n_params` Jacobian (the
/// same per-point gradient the fit already computes every iteration) plus inverting an
/// `n_params x n_params` matrix (at most ~10x10 for the richest term combinations), done once
/// after convergence.
///
/// Falls back to the Moore-Penrose pseudo-inverse when `JᵀJ` is singular or non-positive-definite
/// -- that signals an unidentified parameter direction (a genuine degeneracy, e.g. two
/// parameters that trade off perfectly given this data), not a failed fit, so a best-effort
/// covariance is still returned. Caveat (same as Python's): the pseudo-inverse reports a
/// *near-zero* variance for a fully unconstrained direction, which understates the true
/// uncertainty there -- the fitted parameters and chi2 are unaffected, only that direction's
/// error bar is unreliable.
fn covariance_from_jacobian(
    model: &RainbowModel,
    t: &[f64],
    flux_err: &[f64],
    band_idx: &[usize],
    params: &[f64],
) -> DMatrix<f64> {
    let n = t.len();
    let n_params = params.len();
    let mut j = DMatrix::<f64>::zeros(n, n_params);
    for i in 0..n {
        let (_, grad) = model.model_and_gradient(t[i], band_idx[i], params);
        for k in 0..n_params {
            j[(i, k)] = grad[k] / flux_err[i];
        }
    }
    let jtj = j.transpose() * &j;

    if let Some(inv) = jtj.clone().try_inverse()
        && inv.diagonal().iter().all(|&x| x.is_finite() && x > 0.0)
    {
        return inv;
    }
    let svd = jtj.svd(true, true);
    svd.pseudo_inverse(1e-12)
        .unwrap_or_else(|_| DMatrix::from_element(n_params, n_params, f64::NAN))
}

struct RainbowFitProblem<'a> {
    model: &'a RainbowModel,
    t: &'a [f64],
    flux: &'a [f64],
    flux_err: &'a [f64],
    band_idx: &'a [usize],
    bounds: Vec<(f64, f64)>,
    u: DVector<f64>,
}

impl<'a> LeastSquaresProblem<f64, Dyn, Dyn> for RainbowFitProblem<'a> {
    type ResidualStorage = Owned<f64, Dyn>;
    type JacobianStorage = Owned<f64, Dyn, Dyn>;
    type ParameterStorage = Owned<f64, Dyn>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.u = x.clone();
    }

    fn params(&self) -> DVector<f64> {
        self.u.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let (params, _) = internal_to_external(&self.u, &self.bounds);
        let n = self.t.len();
        let mut r = DVector::zeros(n);
        for i in 0..n {
            let model_flux = self.model.model(self.t[i], self.band_idx[i], &params);
            r[i] = (model_flux - self.flux[i]) / self.flux_err[i];
        }
        Some(r)
    }

    fn jacobian(&self) -> Option<OMatrix<f64, Dyn, Dyn>> {
        let (params, dext_du) = internal_to_external(&self.u, &self.bounds);
        let n = self.t.len();
        let n_params = params.len();
        let mut j = OMatrix::<f64, Dyn, Dyn>::zeros(n, n_params);
        for i in 0..n {
            let (_, grad) = self
                .model
                .model_and_gradient(self.t[i], self.band_idx[i], &params);
            for k in 0..n_params {
                j[(i, k)] = grad[k] * dext_du[k] / self.flux_err[i];
            }
        }
        Some(j)
    }
}

/// Result of fitting [`RainbowModel`] to a light curve.
pub(crate) struct FitResult {
    /// Fitted physical parameters, in [`RainbowModel::param_names`] order.
    pub(crate) params: Vec<f64>,
    /// 1-sigma uncertainty on each parameter (same order as `params`), from the Gauss-Newton
    /// covariance approximation. See [`covariance_from_jacobian`] for the method and its caveats
    /// around genuinely degenerate parameters. (The full covariance matrix, off-diagonal
    /// correlations included, is computed as an intermediate value but not currently surfaced
    /// here -- a flat per-parameter matrix doesn't fit this crate's flat named-scalar feature
    /// output convention. Easy to add later if a use for it comes up.)
    pub(crate) errors: Vec<f64>,
    pub(crate) reduced_chi2: f64,
    pub(crate) success: bool,
}

/// Jointly fits all bands' flux observations `(t, flux, flux_err, band_idx)` to `model`.
///
/// `band_idx[i]` must index into `model`'s bands. Mismatched array lengths are a caller bug and
/// panic; too few data points to constrain the model (a normal data-quality condition, not a bug
/// -- real light curves are sometimes too sparse) instead returns a [`FitResult`] with
/// `success: false` and NaN-filled outputs, so callers don't need to pre-check `t.len()`
/// themselves.
pub(crate) fn fit(
    model: &RainbowModel,
    t: &[f64],
    flux: &[f64],
    flux_err: &[f64],
    band_idx: &[usize],
) -> FitResult {
    assert_eq!(t.len(), flux.len());
    assert_eq!(t.len(), flux_err.len());
    assert_eq!(t.len(), band_idx.len());

    let n_params = model.n_params();
    if t.len() <= n_params {
        return FitResult {
            params: vec![f64::NAN; n_params],
            errors: vec![f64::NAN; n_params],
            reduced_chi2: f64::NAN,
            success: false,
        };
    }

    let bounds = model.bounds(t, flux, flux_err, band_idx);
    let guess = model.initial_guess(t, flux, flux_err, band_idx);
    let u0 = external_to_internal(&guess, &bounds);

    let problem = RainbowFitProblem {
        model,
        t,
        flux,
        flux_err,
        band_idx,
        bounds,
        u: u0,
    };

    let (solved, report) = LevenbergMarquardt::new().minimize(problem);
    let (params, _) = internal_to_external(&solved.u, &solved.bounds);

    let dof = (t.len() as isize - n_params as isize).max(1) as f64;
    let chi2 = 2.0 * report.objective_function;
    let reduced_chi2 = chi2 / dof;

    let cov = covariance_from_jacobian(model, t, flux_err, band_idx, &params);
    let errors: Vec<f64> = cov.diagonal().iter().map(|&v| v.max(0.0).sqrt()).collect();

    FitResult {
        params,
        errors,
        reduced_chi2,
        success: report.termination.was_successful(),
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::Band;
    use super::super::terms::{Bolometric, Spectral, Temperature};
    use super::*;
    use levenberg_marquardt::differentiate_numerically;

    #[test]
    fn analytic_jacobian_matches_numerical_differentiation() {
        let bands = vec![
            Band {
                name: "g".to_string(),
                wavelength_cm: 4770.0e-8,
            },
            Band {
                name: "r".to_string(),
                wavelength_cm: 6231.0e-8,
            },
        ];
        let model = RainbowModel::new(
            Bolometric::Bazin,
            Temperature::Sigmoid,
            Spectral::Planck,
            bands,
            false,
        );

        let t: Vec<f64> = (0..20).map(|i| i as f64 * 3.0).collect();
        let band_idx: Vec<usize> = (0..20).map(|i| i % 2).collect();
        let flux: Vec<f64> = t.iter().map(|&x| 50.0 + 10.0 * (x / 20.0).sin()).collect();
        let flux_err: Vec<f64> = vec![1.0; 20];

        let bounds = model.bounds(&t, &flux, &flux_err, &band_idx);
        let truth_external = [30.0, 100.0, 5.0, 20.0, 9000.0, 0.1, 6.0];
        let u = external_to_internal(&truth_external, &bounds);

        let mut problem = RainbowFitProblem {
            model: &model,
            t: &t,
            flux: &flux,
            flux_err: &flux_err,
            band_idx: &band_idx,
            bounds,
            u,
        };

        let analytic = problem.jacobian().unwrap();
        let numeric = differentiate_numerically(&mut problem).unwrap();

        assert_eq!(analytic.shape(), numeric.shape());
        for i in 0..analytic.nrows() {
            for j in 0..analytic.ncols() {
                let a = analytic[(i, j)];
                let n = numeric[(i, j)];
                assert!(
                    (a - n).abs() <= 1e-4 * n.abs().max(1.0),
                    "jacobian[{i},{j}]: analytic={a}, numeric={n}"
                );
            }
        }
    }

    #[test]
    fn too_few_points_returns_failure_instead_of_panicking() {
        let bands = vec![
            Band {
                name: "g".to_string(),
                wavelength_cm: 4770.0e-8,
            },
            Band {
                name: "r".to_string(),
                wavelength_cm: 6231.0e-8,
            },
            Band {
                name: "i".to_string(),
                wavelength_cm: 7625.0e-8,
            },
        ];
        let model = RainbowModel::new(
            Bolometric::Bazin,
            Temperature::Sigmoid,
            Spectral::Planck,
            bands,
            false,
        );

        let t = vec![0.0, 1.0, 2.0];
        let flux = vec![1.0, 2.0, 1.5];
        let flux_err = vec![0.1, 0.1, 0.1];
        let band_idx = vec![0, 1, 2];
        let result = fit(&model, &t, &flux, &flux_err, &band_idx);
        assert!(!result.success);
        assert!(result.params.iter().all(|p| p.is_nan()));
        assert!(result.reduced_chi2.is_nan());
    }
}
