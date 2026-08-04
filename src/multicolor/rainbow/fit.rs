//! Drives [`RainbowModel`] through this crate's shared [`CurveFitAlgorithm`] (Ceres/MCMC/NUTS),
//! the same optimizer infrastructure every other `*Fit` feature uses.
//!
//! # Band tagging
//!
//! Every [`CurveFitAlgorithm`] backend calls its model/derivatives closures with a bare scalar
//! `t` per data point -- there's no side channel for "which band is this point from". Rather than
//! extend [`Data`]/[`CurveFitTrait`] with a new per-point-tag channel across all four backends
//! (GSL FFI, Ceres FFI, MCMC, NUTS), band index is folded into `t` itself before the point ever
//! reaches the optimizer: each band gets a disjoint offset range,
//! `t_encoded = (t - t_min) + band_idx * stride` with `stride` chosen wider than the full time
//! span, so `band_idx = floor(t_encoded / stride)` and the real `t` recover exactly. This is the
//! same trick used for "global" / simultaneous multi-dataset fits in `lmfit`/`scipy` tutorials
//! (concatenate independent variables into non-overlapping ranges); it's entirely local to this
//! function and touches none of `nl_fit`'s shared types.
//!
//! `t` is fixed input data to every backend (never perturbed by the optimizer), so the encoding
//! is stable across iterations.
//!
//! # Bounds and physical units
//!
//! Unlike [`crate::nl_fit::fit_eval!`]-based features, this skips [`NormalizedData`]: Rainbow's
//! model already works in raw physical units (matching the Python reference), and box bounds are
//! passed straight through in those units to whichever algorithm was chosen -- Ceres/MCMC/NUTS
//! all honor them (Ceres via native box constraints, MCMC/NUTS by rejecting proposals outside
//! `bounds`). See [`super::algorithm_supports_bounds`] for why Lmsder isn't offered here: it
//! ignores the `bounds` argument entirely, and Rainbow has no equivalent of `BazinFit`'s
//! `abs()`-based unconstrained conditioning to fall back on.
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

/// Encodes `(t, band_idx)` into a single `f64` `t_encoded` such that `decode(encode(t, b)) ==
/// (t, b)` exactly, by giving each band a disjoint offset range. See the module docs.
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
        // Generous margin: encoded offsets never approach the stride boundary, so floor() below
        // is robust to floating-point rounding.
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
    let (lower, upper): (Vec<f64>, Vec<f64>) = bounds.into_iter().unzip();
    let guess = model.initial_guess(t, flux, flux_err, band_idx);

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
        move |t_encoded: f64, params: &[f64]| -> f64 {
            let (t, b) = code.decode(t_encoded);
            model.model(t, b, params)
        }
    };
    let derivatives_fn = {
        let model = model.clone();
        move |t_encoded: f64, params: &[f64], jac: &mut [f64]| {
            let (t, b) = code.decode(t_encoded);
            let (_, grad) = model.model_and_gradient(t, b, params);
            jac.copy_from_slice(&grad);
        }
    };

    let CurveFitResult {
        x: params,
        reduced_chi2,
        success,
    } = algorithm.curve_fit(
        data,
        &guess,
        (&lower, &upper),
        model_fn,
        derivatives_fn,
        LnPrior::none(),
    );

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
