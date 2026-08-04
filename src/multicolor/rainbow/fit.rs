//! Drives [`RainbowModel`] through this crate's shared [`CurveFitAlgorithm`] (Ceres/MCMC/NUTS).
//!
//! **Band tagging**: every backend's model/derivatives closures take a single scalar `t` per
//! point, with no channel for "which band". Band index is folded into `t` before it reaches the
//! optimizer -- each band gets a disjoint offset range, `t_encoded = (t - t_min) + band_idx *
//! stride`, decoded inside the closures. Standard technique for simultaneous multi-dataset fits
//! (concatenate independent variables into non-overlapping ranges); entirely local to this
//! function, doesn't touch `nl_fit`'s shared types.
//!
//! **Bounds**: `ceres-solver` 0.5.1 silently ignores upper bounds (`CurveFitProblem1DBuilder`
//! wires up `lower_bounds` but has a literal `// TODO: upper bounds`; confirmed with an isolated
//! repro outside this crate). So bounds are enforced by reparametrization rather than delegated
//! to the backend: each parameter is mapped through a logistic transform from an unconstrained
//! `u` into its `(lower, upper)` interval before the model ever sees it, and the optimizer is
//! handed effectively unconstrained bounds (`+-inf`). Holds regardless of which backend's bound
//! support is broken or missing (Lmsder's is nonexistent, see
//! [`algorithm_supports_bounds`](super::algorithm_supports_bounds)).
//!
//! **Iterations**: `CeresCurveFit::default_niterations()` (10) is tuned for simpler single-band
//! features; Rainbow's larger multi-band parameter space needs more (checked against 90 real
//! light curves vs. the Python reference: ~28% converged close at 10 iterations, ~91% at 200) --
//! see the Python bindings' Rainbow-specific Ceres default. Even at 200, a handful of light
//! curves converge to a real but suboptimal local minimum (same failure under Ceres and MCMC
//! alike, so not a backend quirk) -- a known, accepted gap for now.
//!
//! Skips [`NormalizedData`] (unlike [`crate::nl_fit::fit_eval!`] features): the model already
//! works in physical units, matching the Python reference's public API.
//!
//! [`NormalizedData`]: crate::nl_fit::data::NormalizedData

use std::rc::Rc;

use ndarray::Array1;

use crate::nl_fit::{CurveFitAlgorithm, CurveFitResult, CurveFitTrait, LnPrior, data::Data};

use super::model::RainbowModel;

/// Result of fitting [`RainbowModel`] to a light curve.
pub(crate) struct FitResult {
    /// Fitted physical parameters, in [`RainbowModel::param_names`] order.
    pub(crate) params: Vec<f64>,
    pub(crate) reduced_chi2: f64,
    pub(crate) success: bool,
}

/// Encodes `(t, band_idx)` into one `f64` such that `decode(encode(t, b)) == (t, b)` exactly, by
/// giving each band a disjoint offset range. See the module docs.
#[derive(Clone, Copy)]
struct BandTimeCode {
    t_min: f64,
    stride: f64,
}

impl BandTimeCode {
    fn new(t: &[f64]) -> Self {
        let t_min = t.iter().copied().fold(f64::INFINITY, f64::min);
        let t_max = t.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let span = (t_max - t_min).max(1e-6);
        // Margin wide enough that encoded offsets never approach the stride boundary, so
        // floor() below is robust to floating-point rounding.
        let stride = 4.0 * span + 1.0;
        Self { t_min, stride }
    }

    fn encode(&self, t: f64, band_idx: usize) -> f64 {
        (t - self.t_min) + band_idx as f64 * self.stride
    }

    fn decode(&self, t_encoded: f64) -> (f64, usize) {
        let band_idx = (t_encoded / self.stride).floor();
        let t = t_encoded - band_idx * self.stride + self.t_min;
        (t, band_idx as usize)
    }
}

/// `(external, d(external)/du)` for unconstrained `u`, logistically mapped into `(lower,
/// upper)`. See the module docs' "Bounds" section.
fn bounded_transform(u: f64, lower: f64, upper: f64) -> (f64, f64) {
    let s = 1.0 / (1.0 + (-u).exp());
    let external = lower + (upper - lower) * s;
    let dext_du = (upper - lower) * s * (1.0 - s);
    (external, dext_du)
}

/// Inverse of [`bounded_transform`] (the logit function), for converting a physical-unit initial
/// guess into `u`-space. Clamped away from the exact bounds to avoid `+-inf`.
fn inverse_bounded_transform(external: f64, lower: f64, upper: f64) -> f64 {
    let frac = ((external - lower) / (upper - lower)).clamp(1e-9, 1.0 - 1e-9);
    (frac / (1.0 - frac)).ln()
}

fn internal_to_external(u: &[f64], bounds: &[(f64, f64)]) -> (Vec<f64>, Vec<f64>) {
    let mut external = Vec::with_capacity(bounds.len());
    let mut dext_du = Vec::with_capacity(bounds.len());
    for (&uk, &(lower, upper)) in u.iter().zip(bounds) {
        let (e, d) = bounded_transform(uk, lower, upper);
        external.push(e);
        dext_du.push(d);
    }
    (external, dext_du)
}

fn external_to_internal(external: &[f64], bounds: &[(f64, f64)]) -> Vec<f64> {
    external
        .iter()
        .zip(bounds)
        .map(|(&e, &(lower, upper))| inverse_bounded_transform(e, lower, upper))
        .collect()
}

/// Jointly fits all bands' `(t, flux, flux_err, band_idx)` to `model` using `algorithm`.
///
/// Too few points to constrain the model returns `success: false` with NaN outputs rather than
/// panicking -- sparse real light curves are a normal data-quality issue, not a bug.
pub(crate) fn fit(
    model: &RainbowModel,
    algorithm: &CurveFitAlgorithm,
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
            reduced_chi2: f64::NAN,
            success: false,
        };
    }

    let bounds = model.bounds(t, flux, flux_err, band_idx);
    let guess = model.initial_guess(t, flux, flux_err, band_idx);
    let u0 = external_to_internal(&guess, &bounds);

    let code = BandTimeCode::new(t);
    let t_encoded: Vec<f64> = t
        .iter()
        .zip(band_idx)
        .map(|(&ti, &b)| code.encode(ti, b))
        .collect();
    let inv_err: Vec<f64> = flux_err.iter().map(|e| e.recip()).collect();

    let data = Rc::new(Data {
        t: Array1::from(t_encoded),
        m: Array1::from(flux.to_vec()),
        inv_err: Array1::from(inv_err),
    });

    let model_fn = {
        let model = model.clone();
        let bounds = bounds.clone();
        move |t_encoded: f64, u: &[f64]| -> f64 {
            let (t, b) = code.decode(t_encoded);
            let (external, _) = internal_to_external(u, &bounds);
            model.model(t, b, &external)
        }
    };
    let derivatives_fn = {
        let model = model.clone();
        let bounds = bounds.clone();
        move |t_encoded: f64, u: &[f64], jac: &mut [f64]| {
            let (t, b) = code.decode(t_encoded);
            let (external, dext_du) = internal_to_external(u, &bounds);
            let (_, grad) = model.model_and_gradient(t, b, &external);
            for k in 0..jac.len() {
                jac[k] = grad[k] * dext_du[k];
            }
        }
    };

    // `u` is unconstrained; bounds are enforced by the logistic transform above, not by the
    // backend's own (partly broken) bound support. See the module docs.
    let unconstrained_lower = vec![f64::NEG_INFINITY; n_params];
    let unconstrained_upper = vec![f64::INFINITY; n_params];

    let CurveFitResult {
        x: u_fit,
        reduced_chi2,
        success,
    } = algorithm.curve_fit(
        data,
        &u0,
        (&unconstrained_lower, &unconstrained_upper),
        model_fn,
        derivatives_fn,
        LnPrior::none(),
    );

    let (params, _) = internal_to_external(&u_fit, &bounds);

    FitResult {
        params,
        reduced_chi2,
        success,
    }
}

#[cfg(test)]
mod tests {
    use super::super::model::Band;
    use super::super::terms::{Bolometric, Spectral, Temperature};
    use super::*;
    use crate::nl_fit::McmcCurveFit;

    #[test]
    fn band_time_code_roundtrips() {
        let t = [0.0, 12.5, 37.0, -5.0, 100.0];
        let code = BandTimeCode::new(&t);
        for &ti in &t {
            for b in 0..5usize {
                let encoded = code.encode(ti, b);
                let (decoded_t, decoded_b) = code.decode(encoded);
                assert!((decoded_t - ti).abs() < 1e-9, "t: {ti} -> {decoded_t}");
                assert_eq!(decoded_b, b);
            }
        }
    }

    #[test]
    fn bounded_transform_roundtrips_and_stays_within_bounds() {
        let (lower, upper) = (-0.99, 0.99);
        for u in [-10.0, -1.0, 0.0, 1.0, 10.0] {
            let (external, _) = bounded_transform(u, lower, upper);
            assert!(external > lower && external < upper);
            let back = inverse_bounded_transform(external, lower, upper);
            assert!(
                (back - u).abs() < 1e-6,
                "u={u}, external={external}, back={back}"
            );
        }
    }

    #[test]
    fn full_chain_jacobian_matches_finite_difference() {
        // Reproduces fit()'s model_fn/derivatives_fn closures (u -> external -> model) against
        // wide, asymmetric bounds like real Rainbow bounds actually are.
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
            Spectral::ModifiedBlackBody,
            bands,
            false,
        );

        let t = [
            61033.0, 61035.0, 61040.0, 61041.0, 61050.0, 61052.0, 61060.0, 61070.0, 61071.0,
            61098.0,
        ];
        let flux = [
            900.0, 950.0, 3000.0, 3100.0, 16000.0, 15500.0, 8000.0, 6200.0, 6100.0, 5800.0,
        ];
        let flux_err = [50.0; 10];
        let band_idx = [0, 1, 0, 1, 0, 1, 0, 0, 1, 1];

        let bounds = model.bounds(&t, &flux, &flux_err, &band_idx);
        let guess = model.initial_guess(&t, &flux, &flux_err, &band_idx);
        let u0 = external_to_internal(&guess, &bounds);

        let code = BandTimeCode::new(&t);
        let t_encoded = code.encode(t[4], band_idx[4]);

        let model_fn = |u: &[f64]| -> f64 {
            let (external, _) = internal_to_external(u, &bounds);
            let (t, b) = code.decode(t_encoded);
            model.model(t, b, &external)
        };
        let derivatives_fn = |u: &[f64]| -> Vec<f64> {
            let (external, dext_du) = internal_to_external(u, &bounds);
            let (t, b) = code.decode(t_encoded);
            let (_, grad) = model.model_and_gradient(t, b, &external);
            (0..grad.len()).map(|k| grad[k] * dext_du[k]).collect()
        };

        let analytic = derivatives_fn(&u0);
        for k in 0..u0.len() {
            let h = 1e-5 * u0[k].abs().max(1.0);
            let mut plus = u0.clone();
            let mut minus = u0.clone();
            plus[k] += h;
            minus[k] -= h;
            let numeric = (model_fn(&plus) - model_fn(&minus)) / (2.0 * h);
            assert!(
                (analytic[k] - numeric).abs() <= 1e-3 * numeric.abs().max(1.0),
                "param {k}: analytic={}, numeric={}",
                analytic[k],
                numeric
            );
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
        let algorithm: CurveFitAlgorithm = McmcCurveFit::new(64, None).into();
        let result = fit(&model, &algorithm, &t, &flux, &flux_err, &band_idx);
        assert!(!result.success);
        assert!(result.params.iter().all(|p| p.is_nan()));
        assert!(result.reduced_chi2.is_nan());
    }
}
