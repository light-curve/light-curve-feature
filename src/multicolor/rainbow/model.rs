//! Composition of a chosen bolometric + temperature + spectral term (plus an optional per-band
//! baseline) into the Rainbow flux model:
//! $$
//! \mathrm{flux}(t, \lambda) = \mathrm{bol}(t) \cdot \frac{B(\lambda, T(t))}{\mathrm{norm}(T(t))} \ [+ \ \mathrm{baseline}_\mathrm{band}],
//! \quad \mathrm{norm}(T) = \frac{\sigma_{SB} T^4}{\pi\,\bar\nu},
//! $$
//! where $\bar\nu$ is the mean frequency across the fitted bands' wavelengths. This composition
//! (and the `norm(T)` normalization, which always uses the Stefan-Boltzmann/Planck form
//! regardless of which SED term is chosen -- only the SED's *shape* varies per term, not this
//! amplitude normalization) mirrors Python's
//! `light_curve_py.features.rainbow._base.BaseRainbowFit._lsq_model_{no_,with_}baseline`.
//!
//! This module works entirely in physical `f64` units on a fixed internal [`Band`] list (name +
//! wavelength in cm); it has no awareness of the crate's generic `T: Float` or `P: PassbandTrait`
//! -- [RainbowFit](super::RainbowFit) converts each configured passband's name/wavelength once at
//! construction time and is the only place those generics matter.
//!
//! Since different bolometric/temperature/spectral term combinations (and band counts, when
//! `with_baseline`) have different numbers of (possibly shared, e.g. `reference_time`)
//! parameters, the parameter vector is a runtime-sized `&[f64]` rather than a fixed struct;
//! [`ParamLayout`] records, for each term, which global parameter index its local parameters map
//! to.

use std::collections::HashMap;
use std::f64::consts::PI;

use super::constants::{SIGMA_SB, SPEED_OF_LIGHT};
use super::terms::common::{max_min, median, ptp};
use super::terms::{Bolometric, MAX_LOCAL_PARAMS, Spectral, Temperature};

/// A photometric band: a name and its effective wavelength in cm.
#[derive(Debug, Clone)]
pub(crate) struct Band {
    pub(crate) name: String,
    pub(crate) wavelength_cm: f64,
}

fn baseline_param_name(band_name: &str) -> String {
    format!("baseline_{band_name}")
}

/// Maps each term's local parameter indices (and, if `with_baseline`, each band's baseline
/// parameter) to indices in the model's global (deduplicated-by-name) parameter vector.
struct ParamLayout {
    names: Vec<String>,
    bol_map: Vec<usize>,
    temp_map: Vec<usize>,
    spec_map: Vec<usize>,
    /// `baseline_map[band_idx]` = global parameter index, only populated when `with_baseline`.
    baseline_map: Vec<usize>,
}

impl ParamLayout {
    fn build(
        bolometric: &Bolometric,
        temperature: &Temperature,
        spectral: &Spectral,
        bands: &[Band],
        with_baseline: bool,
    ) -> Self {
        let mut names: Vec<String> = Vec::new();
        let mut name_to_idx: HashMap<String, usize> = HashMap::new();

        let intern = |name: String, names: &mut Vec<String>, name_to_idx: &mut HashMap<String, usize>| -> usize {
            if let Some(&idx) = name_to_idx.get(&name) {
                idx
            } else {
                names.push(name.clone());
                let idx = names.len() - 1;
                name_to_idx.insert(name, idx);
                idx
            }
        };

        let bol_map: Vec<usize> = bolometric
            .params()
            .iter()
            .map(|&n| intern(n.to_string(), &mut names, &mut name_to_idx))
            .collect();
        let temp_map: Vec<usize> = temperature
            .params()
            .iter()
            .map(|&n| intern(n.to_string(), &mut names, &mut name_to_idx))
            .collect();
        let spec_map: Vec<usize> = spectral
            .params()
            .iter()
            .map(|&n| intern(n.to_string(), &mut names, &mut name_to_idx))
            .collect();

        let baseline_map: Vec<usize> = if with_baseline {
            bands
                .iter()
                .map(|b| intern(baseline_param_name(&b.name), &mut names, &mut name_to_idx))
                .collect()
        } else {
            Vec::new()
        };

        Self { names, bol_map, temp_map, spec_map, baseline_map }
    }
}

/// The Rainbow model: a chosen bolometric/temperature/spectral term combination (plus an
/// optional per-band additive baseline) bound to a fixed set of bands.
pub(crate) struct RainbowModel {
    bolometric: Bolometric,
    temperature: Temperature,
    spectral: Spectral,
    with_baseline: bool,
    bands: Vec<Band>,
    /// `speed_of_light / mean(wavelength)`, used to normalize the SED so that its scale is
    /// carried entirely by `amplitude`.
    average_nu: f64,
    layout: ParamLayout,
}

impl RainbowModel {
    pub(crate) fn new(
        bolometric: Bolometric,
        temperature: Temperature,
        spectral: Spectral,
        bands: Vec<Band>,
        with_baseline: bool,
    ) -> Self {
        assert!(!bands.is_empty(), "RainbowModel requires at least one band");
        let mean_wave_cm: f64 = bands.iter().map(|b| b.wavelength_cm).sum::<f64>() / bands.len() as f64;
        let average_nu = SPEED_OF_LIGHT / mean_wave_cm;
        let layout = ParamLayout::build(&bolometric, &temperature, &spectral, &bands, with_baseline);
        Self { bolometric, temperature, spectral, with_baseline, bands, average_nu, layout }
    }

    pub(crate) fn n_params(&self) -> usize {
        self.layout.names.len()
    }

    pub(crate) fn param_names(&self) -> &[String] {
        &self.layout.names
    }

    pub(crate) fn bands(&self) -> &[Band] {
        &self.bands
    }

    /// Evaluate the model flux at time `t` in the given band.
    pub(crate) fn model(&self, t: f64, band_idx: usize, params: &[f64]) -> f64 {
        self.model_and_gradient(t, band_idx, params).0
    }

    /// Evaluate the model flux and its gradient w.r.t. every global parameter (length
    /// `n_params()`), mirroring `_lsq_jac_{no_,with_}baseline`'s chain rule through
    /// `bol(t) * SED(wave, T(t)) / norm(T(t)) [+ baseline_band]`.
    pub(crate) fn model_and_gradient(&self, t: f64, band_idx: usize, params: &[f64]) -> (f64, Vec<f64>) {
        let wave_cm = self.bands[band_idx].wavelength_cm;

        let n_bol = self.bolometric.n_params();
        let n_temp = self.temperature.n_params();
        let n_spec = self.spectral.n_params();

        let mut bol_local = [0.0; MAX_LOCAL_PARAMS];
        for i in 0..n_bol {
            bol_local[i] = params[self.layout.bol_map[i]];
        }
        let mut bol_jac = [0.0; MAX_LOCAL_PARAMS];
        let bol = self.bolometric.value_jac(t, &bol_local[..n_bol], &mut bol_jac[..n_bol]);

        let mut temp_local = [0.0; MAX_LOCAL_PARAMS];
        for i in 0..n_temp {
            temp_local[i] = params[self.layout.temp_map[i]];
        }
        let mut temp_jac = [0.0; MAX_LOCAL_PARAMS];
        let temp = self.temperature.value_jac(t, &temp_local[..n_temp], &mut temp_jac[..n_temp]);

        let mut spec_local = [0.0; MAX_LOCAL_PARAMS];
        for i in 0..n_spec {
            spec_local[i] = params[self.layout.spec_map[i]];
        }
        let mut spec_jac = [0.0; MAX_LOCAL_PARAMS];
        let (spectral_val, dspectral_dt) =
            self.spectral.value_jac(wave_cm, temp, &spec_local[..n_spec], &mut spec_jac[..n_spec]);

        // norm(T) is always the Stefan-Boltzmann/Planck-based normalization, regardless of which
        // spectral term is in use (matches Python: only the SED *shape* varies per term, the
        // amplitude normalization does not).
        let norm = SIGMA_SB * temp.powi(4) / PI / self.average_nu;
        let shape = spectral_val / norm;
        // d(shape)/dT = d(spectral)/dT / norm - 4*shape/T, since d(norm)/dT = 4*norm/T.
        let dshape_dt = dspectral_dt / norm - 4.0 * shape / temp;

        let mut flux = bol * shape;

        let mut grad = vec![0.0; self.n_params()];
        for i in 0..n_bol {
            grad[self.layout.bol_map[i]] += bol_jac[i] * shape;
        }
        for i in 0..n_temp {
            grad[self.layout.temp_map[i]] += bol * dshape_dt * temp_jac[i];
        }
        for i in 0..n_spec {
            grad[self.layout.spec_map[i]] += bol * spec_jac[i] / norm;
        }

        if self.with_baseline {
            let g = self.layout.baseline_map[band_idx];
            flux += params[g];
            grad[g] += 1.0;
        }

        (flux, grad)
    }

    /// Assembles a per-term `Vec` (`bol`/`temp`/`spec`, each in that term's own `params()`
    /// order) into the global parameter vector, first writer wins for shared names (e.g.
    /// `reference_time`: the bolometric term's value is used, the temperature term's is computed
    /// but discarded).
    fn assemble<T: Copy + Default>(&self, bol: Vec<T>, temp: Vec<T>, spec: Vec<T>) -> Vec<T> {
        let mut out = vec![T::default(); self.n_params()];
        let mut filled = vec![false; self.n_params()];
        for (i, v) in bol.into_iter().enumerate() {
            let g = self.layout.bol_map[i];
            out[g] = v;
            filled[g] = true;
        }
        for (i, v) in temp.into_iter().enumerate() {
            let g = self.layout.temp_map[i];
            if !filled[g] {
                out[g] = v;
                filled[g] = true;
            }
        }
        for (i, v) in spec.into_iter().enumerate() {
            let g = self.layout.spec_map[i];
            out[g] = v;
        }
        out
    }

    /// Per-band median flux, used as the baseline's own initial guess and to baseline-subtract
    /// the data before computing every other term's initial guess / bounds (mirrors Python's
    /// `_baseline_initial_guesses` + `m_corr`).
    fn per_band_baseline_estimate(&self, flux: &[f64], band_idx: &[usize]) -> Vec<f64> {
        (0..self.bands.len())
            .map(|b| {
                let band_flux: Vec<f64> =
                    flux.iter().zip(band_idx).filter(|&(_, &bi)| bi == b).map(|(&f, _)| f).collect();
                if band_flux.is_empty() { 0.0 } else { median(&band_flux) }
            })
            .collect()
    }

    /// Heuristic initial guess (physical units), same order as [`Self::param_names`].
    pub(crate) fn initial_guess(&self, t: &[f64], flux: &[f64], flux_err: &[f64], band_idx: &[usize]) -> Vec<f64> {
        let (flux_corrected, baseline_guess) = if self.with_baseline {
            let baseline_guess = self.per_band_baseline_estimate(flux, band_idx);
            let corrected: Vec<f64> = flux.iter().zip(band_idx).map(|(&f, &b)| f - baseline_guess[b]).collect();
            (corrected, baseline_guess)
        } else {
            (flux.to_vec(), Vec::new())
        };

        let bol_guess = self.bolometric.initial_guess(t, &flux_corrected, flux_err);
        let temp_guess = self.temperature.initial_guess(t, &flux_corrected, flux_err);
        let spec_guess = self.spectral.initial_guess();

        // Each term's guess is matched positionally against its own params() list (see
        // layout.{bol,temp,spec}_map); a length mismatch here silently desyncs every subsequent
        // parameter.
        debug_assert_eq!(bol_guess.len(), self.bolometric.n_params());
        debug_assert_eq!(temp_guess.len(), self.temperature.n_params());
        debug_assert_eq!(spec_guess.len(), self.spectral.n_params());

        let mut params = self.assemble(bol_guess, temp_guess, spec_guess);
        for (b, g) in self.layout.baseline_map.iter().enumerate() {
            params[*g] = baseline_guess[b];
        }
        params
    }

    /// Box bounds `(lower, upper)` (physical units), same order as [`Self::param_names`].
    pub(crate) fn bounds(&self, t: &[f64], flux: &[f64], flux_err: &[f64], band_idx: &[usize]) -> Vec<(f64, f64)> {
        let (flux_corrected, baseline_bounds) = if self.with_baseline {
            let baseline_guess = self.per_band_baseline_estimate(flux, band_idx);
            let corrected: Vec<f64> = flux.iter().zip(band_idx).map(|(&f, &b)| f - baseline_guess[b]).collect();
            let bounds: Vec<(f64, f64)> = (0..self.bands.len())
                .map(|b| {
                    let band_flux: Vec<f64> =
                        flux.iter().zip(band_idx).filter(|&(_, &bi)| bi == b).map(|(&f, _)| f).collect();
                    if band_flux.is_empty() {
                        (0.0, 0.0)
                    } else {
                        let (max, min) = max_min(&band_flux);
                        (min - 10.0 * ptp(&band_flux), max)
                    }
                })
                .collect();
            (corrected, bounds)
        } else {
            (flux.to_vec(), Vec::new())
        };

        let bol_bounds = self.bolometric.bounds(t, &flux_corrected, flux_err);
        let temp_bounds = self.temperature.bounds(t, &flux_corrected, flux_err);
        let spec_bounds = self.spectral.bounds();

        debug_assert_eq!(bol_bounds.len(), self.bolometric.n_params());
        debug_assert_eq!(temp_bounds.len(), self.temperature.n_params());
        debug_assert_eq!(spec_bounds.len(), self.spectral.n_params());

        let mut bounds = self.assemble(bol_bounds, temp_bounds, spec_bounds);
        for (b, g) in self.layout.baseline_map.iter().enumerate() {
            bounds[*g] = baseline_bounds[b];
        }
        bounds
    }

    /// Analytic bolometric peak time, if defined for the chosen bolometric term.
    pub(crate) fn peak_time(&self, params: &[f64]) -> Option<f64> {
        let n_bol = self.bolometric.n_params();
        let bol_local: Vec<f64> = (0..n_bol).map(|i| params[self.layout.bol_map[i]]).collect();
        self.bolometric.peak_time(&bol_local)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_bands() -> Vec<Band> {
        vec![
            Band { name: "g".to_string(), wavelength_cm: 4770.0e-8 },
            Band { name: "r".to_string(), wavelength_cm: 6231.0e-8 },
        ]
    }

    fn check_gradient(model: &RainbowModel, params: &[f64], h: &[f64]) {
        let t = 15.0;
        let band_idx = 1;
        let (_, analytic) = model.model_and_gradient(t, band_idx, params);
        for k in 0..params.len() {
            let mut plus = params.to_vec();
            let mut minus = params.to_vec();
            plus[k] += h[k];
            minus[k] -= h[k];
            let f_plus = model.model(t, band_idx, &plus);
            let f_minus = model.model(t, band_idx, &minus);
            let numeric = (f_plus - f_minus) / (2.0 * h[k]);
            assert!(
                (analytic[k] - numeric).abs() <= 1e-4 * numeric.abs().max(1.0),
                "param {k}: analytic={}, numeric={}",
                analytic[k],
                numeric
            );
        }
    }

    #[test]
    fn bazin_sigmoid_planck_gradient_matches_finite_difference() {
        let model =
            RainbowModel::new(Bolometric::Bazin, Temperature::Sigmoid, Spectral::Planck, sample_bands(), false);
        // reference_time, amplitude, rise_time, fall_time, T, T_amplitude, t_color
        let params = [10.0, 5.0, 3.0, 20.0, 9000.0, 0.2, 5.0];
        let h = [1e-4, 1e-5, 1e-5, 1e-5, 1e-2, 1e-6, 1e-5];
        check_gradient(&model, &params, &h);
    }

    #[test]
    fn bazin_sigmoid_planck_with_baseline_gradient_matches_finite_difference() {
        let model = RainbowModel::new(Bolometric::Bazin, Temperature::Sigmoid, Spectral::Planck, sample_bands(), true);
        // reference_time, amplitude, rise_time, fall_time, T, T_amplitude, t_color, baseline_g, baseline_r
        let params = [10.0, 5.0, 3.0, 20.0, 9000.0, 0.2, 5.0, 1.5, -0.7];
        let h = [1e-4, 1e-5, 1e-5, 1e-5, 1e-2, 1e-6, 1e-5, 1e-5, 1e-5];
        check_gradient(&model, &params, &h);
    }

    #[test]
    fn sigmoid_bol_constant_genwien_gradient_matches_finite_difference() {
        let model =
            RainbowModel::new(Bolometric::Sigmoid, Temperature::Constant, Spectral::GenWien, sample_bands(), false);
        // reference_time, amplitude, rise_time, T, spec_k
        let params = [10.0, 5.0, 3.0, 9000.0, 1.3];
        let h = [1e-4, 1e-5, 1e-5, 1e-2, 1e-6];
        check_gradient(&model, &params, &h);
    }

    #[test]
    fn doublexp_delayed_sigmoid_logparabola_gradient_matches_finite_difference() {
        let model = RainbowModel::new(
            Bolometric::Doublexp,
            Temperature::DelayedSigmoid,
            Spectral::LogParabola,
            sample_bands(),
            false,
        );
        // reference_time, amplitude, time1, time2, p, T, T_amplitude, t_color, t_delay, sp_a, sp_b
        let params = [10.0, 5.0, 3.0, 4.0, 1.5, 9000.0, 0.2, 5.0, 1.0, 0.3, -0.2];
        let h = [1e-4, 1e-5, 1e-5, 1e-5, 1e-5, 1e-2, 1e-6, 1e-5, 1e-5, 1e-6, 1e-6];
        check_gradient(&model, &params, &h);
    }

    #[test]
    fn shared_reference_time_deduplicated() {
        let model =
            RainbowModel::new(Bolometric::Bazin, Temperature::Sigmoid, Spectral::Planck, sample_bands(), false);
        assert_eq!(model.n_params(), 7); // not 4+4+0=8; reference_time is shared
        assert_eq!(model.param_names()[0], "reference_time");
    }

    #[test]
    fn constant_temperature_has_no_reference_time_collision() {
        let model =
            RainbowModel::new(Bolometric::Bazin, Temperature::Constant, Spectral::Planck, sample_bands(), false);
        assert_eq!(model.n_params(), 5); // 4 bol + 1 temp (T), no shared name
    }

    #[test]
    fn baseline_params_appended_per_band() {
        let model =
            RainbowModel::new(Bolometric::Bazin, Temperature::Sigmoid, Spectral::Planck, sample_bands(), true);
        assert_eq!(model.n_params(), 9); // 7 + baseline_g + baseline_r
        assert_eq!(model.param_names()[7], "baseline_g");
        assert_eq!(model.param_names()[8], "baseline_r");
    }
}
