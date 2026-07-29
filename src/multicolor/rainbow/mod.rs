//! Multi-band non-linear fit of a bolometric envelope times a temperature-dependent spectral
//! energy distribution (SED) -- the "Rainbow" model (Russeil et al. 2023,
//! [arXiv:2310.02916](https://arxiv.org/abs/2310.02916)):
//! $$
//! \mathrm{flux}(t, \lambda) = \mathrm{bol}(t) \cdot \frac{B(\lambda, T(t))}{\mathrm{norm}(T(t))} \ [+ \ \mathrm{baseline}_\mathrm{band}].
//! $$
//! `bol(t)`, `T(t)`, and `B(lambda, T)` are independently pluggable ([`terms::Bolometric`],
//! [`terms::Temperature`], [`terms::Spectral`] -- see [`RainbowFit::new`]). `norm(T)` is always
//! the Planck-based normalization, so overall flux scale is carried by the bolometric term's
//! `amplitude`.
//!
//! # Why not `nl_fit`
//!
//! [`nl_fit`](crate::nl_fit) (used by `BazinFit`, `VillarFit`, etc.) assumes a compile-time
//! `const NPARAMS`. Rainbow's parameter count depends on which terms are chosen, so it can't be
//! one; [`fit`] instead runs its own `Dyn`-dimensioned Levenberg-Marquardt fit directly, the same
//! way `ParabolaFit` bypasses `nl_fit` for its own closed-form fit. See [`fit`]'s module doc for
//! the bounded-parameter transform.
//!
//! # Uncertainties
//!
//! [`fit::fit`] also returns 1-sigma parameter uncertainties (Gauss-Newton `inv(JᵀJ)`, cheap
//! since it reuses the fit's own gradient), appended to the output as `<name>_sigma` columns
//! instead of going through the generic `Bootstrap` wrapper (which would multiply the cost via
//! resampling). See [`RainbowFit::build_properties`] for the exact layout -- a different
//! convention from every other feature, worth a second look on review.
//!
//! # Open design questions (flagged for review)
//!
//! - **New dependency**: first pure-Rust linear-algebra dependency in the crate
//!   (`levenberg-marquardt` + `nalgebra`), gated behind the opt-in `rainbow` Cargo feature.
//! - **`PassbandTrait::wavelength()`**: new default (`None`) method so `RainbowFit` can require
//!   a wavelength while still fitting into the generic `MultiColorFeature<P, T>` registry.
//! - **Output layout**: see "Uncertainties" above.
//!
//! # Not yet ported from the Python reference
//!
//! - `bolometric = "linexp"`: Python's own docstring flags its guesses/limits as unstable
//!   (confirmed here by a seed sweep -- about half land in a wrong local optimum).
//! - `spectral = "blanketed"`: anchors to the temperature term's own `T` via Python's
//!   `common_temp_spec` cross-term sharing, more involved than the other terms.
//! - Gaussian priors anchoring some parameters (e.g. `T_amplitude`, `beta`) toward 0, and
//!   non-detection/upper-limit support (Python's Tobit-model likelihood).

mod constants;
mod fit;
mod model;
pub mod terms;

use std::collections::BTreeSet;
use std::marker::PhantomData;

use conv::ConvUtil;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::data::MultiColorTimeSeries;
use crate::error::{EvaluatorError, MultiColorEvaluatorError};
use crate::evaluator::{
    EvaluatorInfo, EvaluatorInfoTrait, EvaluatorProperties, FeatureNamesDescriptionsTrait,
};
use crate::float_trait::Float;
use crate::multicolor::multicolor_evaluator::{
    MultiColorEvaluator, MultiColorPassbandSetTrait, PassbandSet,
};
use crate::multicolor::passband::PassbandTrait;

use model::{Band, RainbowModel};
pub use terms::{Bolometric, Spectral, Temperature};

/// Error constructing a [`RainbowFit`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RainbowFitError {
    #[error("RainbowFit requires at least one passband")]
    NoPassbands,

    #[error(
        "passband {0:?} has no known wavelength (PassbandTrait::wavelength() returned None); use MonochromePassband, or a passband type overriding wavelength()"
    )]
    MissingWavelength(String),
}

/// Multi-band fit of the Rainbow model (see the [module-level documentation](self) for the
/// model, why it doesn't use `nl_fit`, and open design questions).
#[derive(Clone, Serialize, Deserialize)]
#[serde(
    into = "RainbowFitParameters<P, T>",
    try_from = "RainbowFitParameters<P, T>",
    bound(
        serialize = "P: PassbandTrait + Serialize, T: Float",
        deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float"
    )
)]
pub struct RainbowFit<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    bolometric: Bolometric,
    temperature: Temperature,
    spectral: Spectral,
    with_baseline: bool,
    passband_set: PassbandSet<P>,
    /// Band indices match `passband_set`'s sorted order (relied on by
    /// `eval_multicolor_no_mcts_check`).
    model: RainbowModel,
    properties: Box<EvaluatorProperties>,
    _marker: PhantomData<T>,
}

impl<P, T> std::fmt::Debug for RainbowFit<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RainbowFit")
            .field("bolometric", &self.bolometric)
            .field("temperature", &self.temperature)
            .field("spectral", &self.spectral)
            .field("with_baseline", &self.with_baseline)
            .field("passband_set", &self.passband_set)
            .finish()
    }
}

impl<P, T> RainbowFit<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    /// Constructs a `RainbowFit` for the given passbands and term choices.
    ///
    /// # Errors
    /// [`RainbowFitError::NoPassbands`] if `passbands` is empty, or
    /// [`RainbowFitError::MissingWavelength`] if any passband's [`PassbandTrait::wavelength`]
    /// returns `None` (only [`crate::multicolor::MonochromePassband`] provides one today).
    pub fn new(
        passbands: impl IntoIterator<Item = P>,
        bolometric: Bolometric,
        temperature: Temperature,
        spectral: Spectral,
        with_baseline: bool,
    ) -> Result<Self, RainbowFitError> {
        let passband_set: BTreeSet<P> = passbands.into_iter().collect();
        if passband_set.is_empty() {
            return Err(RainbowFitError::NoPassbands);
        }

        let bands: Vec<Band> = passband_set
            .iter()
            .map(|p| {
                let wavelength_cm = p
                    .wavelength()
                    .ok_or_else(|| RainbowFitError::MissingWavelength(p.name().into()))?;
                Ok(Band {
                    name: p.name().into(),
                    wavelength_cm,
                })
            })
            .collect::<Result<_, RainbowFitError>>()?;

        let model = RainbowModel::new(bolometric, temperature, spectral, bands, with_baseline);
        let properties = Box::new(Self::build_properties(&model));

        Ok(Self {
            bolometric,
            temperature,
            spectral,
            with_baseline,
            passband_set: PassbandSet(passband_set),
            model,
            properties,
            _marker: PhantomData,
        })
    }

    /// Output layout: fitted parameters, then their 1-sigma uncertainties (see the module-level
    /// documentation's "Uncertainties" section), then the reduced chi2 -- `2*n_params + 1` values
    /// in total.
    fn build_properties(model: &RainbowModel) -> EvaluatorProperties {
        let param_names = model.param_names();
        let mut names = Vec::with_capacity(2 * param_names.len() + 1);
        let mut descriptions = Vec::with_capacity(2 * param_names.len() + 1);

        for name in param_names {
            names.push(format!("rainbow_{name}"));
            descriptions.push(format!("Rainbow fit parameter: {name}"));
        }
        for name in param_names {
            names.push(format!("rainbow_{name}_sigma"));
            descriptions.push(format!(
                "1-sigma uncertainty (Gauss-Newton) of Rainbow fit parameter: {name}"
            ));
        }
        names.push("rainbow_reduced_chi2".to_owned());
        descriptions.push("Reduced chi^2 of the Rainbow fit".to_owned());

        EvaluatorProperties {
            info: EvaluatorInfo {
                size: names.len(),
                // Per-band minimums aren't meaningful for a joint multi-band fit; the real
                // total-vs-parameter-count check happens inside `fit::fit`.
                min_ts_length: 1,
                t_required: true,
                m_required: true,
                w_required: true,
                sorting_required: false,
                // A light curve flat in every band isn't rejected today; it just produces a
                // poorly-constrained fit (known gap, see module docs).
                variability_required: false,
            },
            names,
            descriptions,
        }
    }

    /// Analytic bolometric peak time, if defined for the chosen [`Bolometric`] term. `params`
    /// must be just the fitted parameters (the first `n_params` values of this feature's
    /// output, not the `_sigma`/`reduced_chi2` tail).
    pub fn peak_time(&self, params: &[T]) -> Option<T> {
        let params_f64: Vec<f64> = params.iter().map(|&x| x.value_into().unwrap()).collect();
        self.model.peak_time(&params_f64).map(|t| {
            t.approx_as::<T>().unwrap_or_else(|_| {
                if t.is_sign_negative() {
                    T::min_value()
                } else {
                    T::max_value()
                }
            })
        })
    }
}

impl<P, T> EvaluatorInfoTrait for RainbowFit<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}

impl<P, T> FeatureNamesDescriptionsTrait for RainbowFit<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn get_names(&self) -> Vec<&str> {
        self.properties.names.iter().map(String::as_str).collect()
    }

    fn get_descriptions(&self) -> Vec<&str> {
        self.properties
            .descriptions
            .iter()
            .map(String::as_str)
            .collect()
    }
}

impl<P, T> MultiColorPassbandSetTrait<P> for RainbowFit<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn get_passband_set(&self) -> &PassbandSet<P> {
        &self.passband_set
    }
}

impl<P, T> MultiColorEvaluator<P, T> for RainbowFit<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn eval_multicolor_no_mcts_check<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        // Flatten into flat f64 arrays; band order matches `self.model` (built from
        // `passband_set`'s sorted order in `new`).
        mcts.with_mapping_mut(|_| {});
        let mapping = mcts.mapping().expect("mapping was just ensured");

        let mut t = Vec::new();
        let mut flux = Vec::new();
        let mut flux_err = Vec::new();
        let mut band_idx = Vec::new();

        for (band_i, (p, maybe_ts)) in mapping.iter_passband_set(&self.passband_set).enumerate() {
            let ts = maybe_ts.ok_or_else(|| {
                MultiColorEvaluatorError::wrong_passbands_error(
                    mcts.passbands(),
                    self.passband_set.0.iter(),
                )
            })?;
            let _ = p; // only needed for the error path above
            for i in 0..ts.lenu() {
                let ti: f64 = ts.t.sample[i].value_into().unwrap();
                let mi: f64 = ts.m.sample[i].value_into().unwrap();
                // `w` is inverse-variance weight (`w = 1/sigma^2`), not sigma directly.
                let wi: f64 = ts.w.sample[i].value_into().unwrap();
                t.push(ti);
                flux.push(mi);
                flux_err.push(1.0 / wi.sqrt());
                band_idx.push(band_i);
            }
        }

        let result = fit::fit(&self.model, &t, &flux, &flux_err, &band_idx);
        if !result.success {
            return Err(EvaluatorError::FitDidNotConverge.into());
        }

        let to_t = |x: f64| {
            x.approx_as::<T>().unwrap_or_else(|_| {
                if x.is_sign_negative() {
                    T::min_value()
                } else {
                    T::max_value()
                }
            })
        };
        let mut out: Vec<T> = Vec::with_capacity(self.properties.info.size);
        out.extend(result.params.iter().copied().map(to_t));
        out.extend(result.errors.iter().copied().map(to_t));
        out.push(to_t(result.reduced_chi2));

        Ok(out)
    }
}

/// Serde/JsonSchema surrogate for [`RainbowFit`]: (de)serializes only the user-facing config;
/// `model`/`properties` are derived state, rebuilt via [`RainbowFit::new`].
#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(
    rename = "RainbowFit",
    bound(
        serialize = "P: PassbandTrait + Serialize, T: Float",
        deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float"
    )
)]
struct RainbowFitParameters<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    bolometric: Bolometric,
    temperature: Temperature,
    spectral: Spectral,
    with_baseline: bool,
    passbands: Vec<P>,
    #[serde(skip)]
    _marker: PhantomData<T>,
}

impl<P, T> From<RainbowFit<P, T>> for RainbowFitParameters<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(r: RainbowFit<P, T>) -> Self {
        Self {
            bolometric: r.bolometric,
            temperature: r.temperature,
            spectral: r.spectral,
            with_baseline: r.with_baseline,
            passbands: r.passband_set.0.into_iter().collect(),
            _marker: PhantomData,
        }
    }
}

impl<P, T> TryFrom<RainbowFitParameters<P, T>> for RainbowFit<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    type Error = RainbowFitError;

    fn try_from(p: RainbowFitParameters<P, T>) -> Result<Self, Self::Error> {
        Self::new(
            p.passbands,
            p.bolometric,
            p.temperature,
            p.spectral,
            p.with_baseline,
        )
    }
}

impl<P, T> JsonSchema for RainbowFit<P, T>
where
    P: PassbandTrait + JsonSchema,
    T: Float,
{
    json_schema!(RainbowFitParameters<P, T>, false);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::TimeSeries;
    use crate::multicolor::MonochromePassband;

    use std::collections::BTreeMap;

    fn bands() -> Vec<MonochromePassband<'static, f64>> {
        vec![
            MonochromePassband::new(4770.0e-8, "g"),
            MonochromePassband::new(6231.0e-8, "r"),
        ]
    }

    fn make_rainbow() -> RainbowFit<MonochromePassband<'static, f64>, f64> {
        RainbowFit::new(
            bands(),
            Bolometric::Bazin,
            Temperature::Sigmoid,
            Spectral::Planck,
            false,
        )
        .unwrap()
    }

    #[test]
    fn construction_requires_at_least_one_band() {
        let err = RainbowFit::<MonochromePassband<'static, f64>, f64>::new(
            [],
            Bolometric::Bazin,
            Temperature::Sigmoid,
            Spectral::Planck,
            false,
        )
        .unwrap_err();
        assert_eq!(err, RainbowFitError::NoPassbands);
    }

    #[test]
    fn construction_rejects_wavelength_less_passbands() {
        use crate::multicolor::StringPassband;
        let err = RainbowFit::<StringPassband, f64>::new(
            [StringPassband::from("g")],
            Bolometric::Bazin,
            Temperature::Sigmoid,
            Spectral::Planck,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, RainbowFitError::MissingWavelength(_)));
    }

    #[test]
    fn names_and_size_match_2n_plus_1() {
        let rainbow = make_rainbow();
        // reference_time, amplitude, rise_time, fall_time, T, T_amplitude, t_color = 7 params
        assert_eq!(rainbow.size_hint(), 2 * 7 + 1);
        assert_eq!(rainbow.get_names().len(), rainbow.size_hint());
        assert!(rainbow.get_names().contains(&"rainbow_reference_time"));
        assert!(
            rainbow
                .get_names()
                .contains(&"rainbow_reference_time_sigma")
        );
        assert!(rainbow.get_names().contains(&"rainbow_reduced_chi2"));
    }

    #[test]
    fn recovers_known_parameters_from_noisy_multiband_light_curve() {
        let rainbow = make_rainbow();

        let truth = [58900.0_f64, 120.0, 5.0, 25.0, 11000.0, 0.2, 8.0];
        let mut rng_state: u64 = 42;
        let mut rand01 = move || {
            // xorshift, deterministic, no extra dev-dependency needed here.
            rng_state ^= rng_state << 13;
            rng_state ^= rng_state >> 7;
            rng_state ^= rng_state << 17;
            (rng_state >> 11) as f64 / (1u64 << 53) as f64
        };

        let mut map = BTreeMap::new();
        for band in bands() {
            let n = 60;
            let mut t: Vec<f64> = (0..n).map(|_| truth[0] - 15.0 + rand01() * 90.0).collect();
            t.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let m: Vec<f64> = t
                .iter()
                .map(|&ti| {
                    let flux =
                        rainbow
                            .model
                            .model(ti, if band.name == "g" { 0 } else { 1 }, &truth);
                    let err = (flux.abs() * 0.02).max(0.05);
                    // Box-Muller for approximately Gaussian noise from two uniforms.
                    let u1 = rand01().max(1e-12);
                    let u2 = rand01();
                    let noise =
                        err * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
                    flux + noise
                })
                .collect();
            let w: Vec<f64> = t
                .iter()
                .map(|&ti| {
                    1.0 / (rainbow.model.model(ti, 0, &truth).abs() * 0.02)
                        .max(0.05)
                        .powi(2)
                })
                .collect();
            map.insert(band, TimeSeries::new(t, m, w));
        }

        let mut mcts = MultiColorTimeSeries::from_map(map);
        let result = rainbow.eval_multicolor(&mut mcts).unwrap();

        assert_eq!(result.len(), rainbow.size_hint());
        let reduced_chi2 = result[14];
        assert!(reduced_chi2 < 3.0, "reduced chi2 too high: {reduced_chi2}");

        let rel_err =
            |actual: f64, expected: f64| (actual - expected).abs() / expected.abs().max(1.0);
        assert!(rel_err(result[0], truth[0]) < 0.02); // reference_time
        assert!(rel_err(result[1], truth[1]) < 0.15); // amplitude
        assert!(rel_err(result[4], truth[4]) < 0.2); // T
    }
}
