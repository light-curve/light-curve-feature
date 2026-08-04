//! Temperature-vs-time terms for [RainbowFit](super::super::RainbowFit).

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::common::{max_min, t0_and_weighted_centroid_sigma};

// ---------------------------------------------------------------------
// Constant
// ---------------------------------------------------------------------

fn constant_temperature_jacobian(temperature: f64, jac: &mut [f64]) -> f64 {
    jac[0] = 1.0;
    temperature
}

// ---------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------

fn sigmoid_temperature_jacobian(
    t: f64,
    t0: f64,
    temperature: f64,
    temperature_amplitude: f64,
    t_color: f64,
    jac: &mut [f64],
) -> f64 {
    let dt = t - t0;
    if dt <= -100.0 * t_color {
        jac[0] = 0.0;
        jac[1] = 1.0 + temperature_amplitude;
        jac[2] = temperature;
        jac[3] = 0.0;
        return temperature * (1.0 + temperature_amplitude);
    }
    if dt >= 100.0 * t_color {
        jac[0] = 0.0;
        jac[1] = 1.0 - temperature_amplitude;
        jac[2] = -temperature;
        jac[3] = 0.0;
        return temperature * (1.0 - temperature_amplitude);
    }

    let e = (dt / t_color).exp();
    let inv_1p_e = 1.0 / (1.0 + e);
    let s = inv_1p_e;
    let s_1ms = e * inv_1p_e * inv_1p_e; // s * (1 - s)
    let two_s_m1 = 2.0 * s - 1.0;

    jac[0] = 2.0 * temperature * temperature_amplitude * s_1ms / t_color;
    jac[1] = 1.0 + temperature_amplitude * two_s_m1;
    jac[2] = temperature * two_s_m1;
    jac[3] = 2.0 * temperature * temperature_amplitude * s_1ms * dt / (t_color * t_color);

    temperature * (1.0 + temperature_amplitude * two_s_m1)
}

// ---------------------------------------------------------------------
// Delayed sigmoid
// ---------------------------------------------------------------------

fn delayed_sigmoid_temperature_jacobian(
    t: f64,
    t0: f64,
    temperature: f64,
    temperature_amplitude: f64,
    t_color: f64,
    t_delay: f64,
    jac: &mut [f64],
) -> f64 {
    let mut sigmoid_jac = [0.0; 4];
    let value = sigmoid_temperature_jacobian(
        t,
        t0 + t_delay,
        temperature,
        temperature_amplitude,
        t_color,
        &mut sigmoid_jac,
    );
    jac[0] = sigmoid_jac[0]; // d/d(reference_time)
    jac[1] = sigmoid_jac[1]; // d/d(T)
    jac[2] = sigmoid_jac[2]; // d/d(T_amplitude)
    jac[3] = sigmoid_jac[3]; // d/d(t_color)
    jac[4] = sigmoid_jac[0]; // d/d(t_delay) = d/d(reference_time), since both enter as their sum
    value
}

// ---------------------------------------------------------------------
// Enum wrapper
// ---------------------------------------------------------------------

/// Which parametric function models the temperature-vs-time curve $T(t)$.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum Temperature {
    /// Time-independent temperature: $T(t) = T$. One parameter, `T`. Appropriate when the SED
    /// shape isn't expected to evolve over the observed window (e.g. a short-duration event, or
    /// one where cooling isn't the dominant behavior).
    Constant,
    /// Logistic transition between two temperature plateaus, parametrized by the mid-temperature
    /// $T$ and a dimensionless relative amplitude $T_\mathrm{amplitude} \in (-1, 1)$ so that
    /// $T_\mathrm{max} = T(1 + T_\mathrm{amplitude})$ and $T_\mathrm{min} = T(1 - T_\mathrm{amplitude})$:
    /// $$
    /// T(t) = T\bigl(1 + T_\mathrm{amplitude}(2s - 1)\bigr), \quad s = \frac{1}{1 + \mathrm{e}^{(t-t_0)/\tau_\mathrm{color}}}.
    /// $$
    /// $s$ runs from 1 (early, hot) to 0 (late, cool), so $T(t)$ runs from $T_\mathrm{max}$ down
    /// to $T_\mathrm{min}$; $T_\mathrm{amplitude} = 0$ recovers the constant-temperature case.
    /// Parameters: `reference_time` ($t_0$, shared with the bolometric term when both use it),
    /// `T`, `T_amplitude`, `t_color` ($\tau_\mathrm{color}$). The standard choice for cooling
    /// transients such as supernovae.
    Sigmoid,
    /// Same logistic transition as [`Temperature::Sigmoid`], but with an extra `t_delay`
    /// parameter offsetting the temperature's own reference time from the bolometric one:
    /// $$
    /// T(t) = T\bigl(1 + T_\mathrm{amplitude}(2s - 1)\bigr), \quad s = \frac{1}{1 + \mathrm{e}^{(t-t_0-\tau_\mathrm{delay})/\tau_\mathrm{color}}}.
    /// $$
    /// Parameters: `reference_time` ($t_0$), `T`, `T_amplitude`, `t_color`, `t_delay`
    /// ($\tau_\mathrm{delay}$). Useful when the temperature evolution is expected to lag (or
    /// lead) the bolometric peak rather than track it exactly.
    DelayedSigmoid,
}

const CONSTANT_PARAMS: [&str; 1] = ["T"];
const SIGMOID_TEMP_PARAMS: [&str; 4] = ["reference_time", "T", "T_amplitude", "t_color"];
const DELAYED_SIGMOID_PARAMS: [&str; 5] =
    ["reference_time", "T", "T_amplitude", "t_color", "t_delay"];

impl Temperature {
    pub(crate) fn params(&self) -> &'static [&'static str] {
        match self {
            Temperature::Constant => &CONSTANT_PARAMS,
            Temperature::Sigmoid => &SIGMOID_TEMP_PARAMS,
            Temperature::DelayedSigmoid => &DELAYED_SIGMOID_PARAMS,
        }
    }

    pub(crate) fn n_params(&self) -> usize {
        self.params().len()
    }

    pub(crate) fn value_jac(&self, t: f64, p: &[f64], jac: &mut [f64]) -> f64 {
        match self {
            Temperature::Constant => constant_temperature_jacobian(p[0], jac),
            Temperature::Sigmoid => sigmoid_temperature_jacobian(t, p[0], p[1], p[2], p[3], jac),
            Temperature::DelayedSigmoid => {
                delayed_sigmoid_temperature_jacobian(t, p[0], p[1], p[2], p[3], p[4], jac)
            }
        }
    }

    /// Returns exactly `n_params()` values in [`Self::params`] order. Unlike Python (which
    /// merges initial guesses by *name* and so may omit a term's shared `reference_time`), every
    /// entry here is positional, so `reference_time` guesses are included even though they end
    /// up unused when shared with the bolometric term (see `RainbowModel::initial_guess`, whose
    /// assembly takes the bolometric term's guess for any name both terms declare).
    pub(crate) fn initial_guess(&self, t: &[f64], flux: &[f64], flux_err: &[f64]) -> Vec<f64> {
        match self {
            Temperature::Constant => vec![8000.0],
            Temperature::Sigmoid => {
                let (t0, dt) = t0_and_weighted_centroid_sigma(t, flux, flux_err);
                vec![t0, 10_000.0, 0.0, 2.0 * dt]
            }
            Temperature::DelayedSigmoid => {
                let (t0, dt) = t0_and_weighted_centroid_sigma(t, flux, flux_err);
                vec![t0, 10_000.0, 0.0, 2.0 * dt, 0.0]
            }
        }
    }

    /// Box bounds `(lower, upper)` (physical units), same order as [`Self::params`].
    pub(crate) fn bounds(&self, t: &[f64], flux: &[f64], flux_err: &[f64]) -> Vec<(f64, f64)> {
        const T_BOUNDS: (f64, f64) = (1e3, 2e6);
        const T_AMPLITUDE_BOUNDS: (f64, f64) = (-0.99, 0.99);
        match self {
            Temperature::Constant => vec![T_BOUNDS],
            Temperature::Sigmoid => {
                let (t_max, t_min) = max_min(t);
                let ptp_t = t_max - t_min;
                let (_, dt) = t0_and_weighted_centroid_sigma(t, flux, flux_err);
                vec![
                    (t_min - 10.0 * ptp_t, t_max + 10.0 * ptp_t),
                    T_BOUNDS,
                    T_AMPLITUDE_BOUNDS,
                    (dt / 3.0, 10.0 * ptp_t),
                ]
            }
            Temperature::DelayedSigmoid => {
                let (t_max, t_min) = max_min(t);
                let ptp_t = t_max - t_min;
                let (_, dt) = t0_and_weighted_centroid_sigma(t, flux, flux_err);
                vec![
                    (t_min - 10.0 * ptp_t, t_max + 10.0 * ptp_t),
                    T_BOUNDS,
                    T_AMPLITUDE_BOUNDS,
                    (dt / 3.0, 10.0 * ptp_t),
                    (-ptp_t, ptp_t),
                ]
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_diff_check(
        name: &str,
        n: usize,
        value_jac: impl Fn(&[f64], &mut [f64]) -> f64,
        p: &[f64],
    ) {
        let mut jac = vec![0.0; n];
        value_jac(p, &mut jac);
        let h = 1e-6;
        for k in 0..n {
            let mut plus = p.to_vec();
            let mut minus = p.to_vec();
            plus[k] += h * plus[k].abs().max(1.0);
            minus[k] -= h * minus[k].abs().max(1.0);
            let step = plus[k] - minus[k];
            let mut dummy = vec![0.0; n];
            let f_plus = value_jac(&plus, &mut dummy);
            let f_minus = value_jac(&minus, &mut dummy);
            let numeric = (f_plus - f_minus) / step;
            assert!(
                (jac[k] - numeric).abs() <= 1e-4 * numeric.abs().max(1.0),
                "{name} param {k}: analytic={}, numeric={}",
                jac[k],
                numeric
            );
        }
    }

    #[test]
    fn sigmoid_temperature_jacobian_matches_finite_difference() {
        finite_diff_check(
            "sigmoid_temperature",
            4,
            |p, jac| sigmoid_temperature_jacobian(11.5, p[0], p[1], p[2], p[3], jac),
            &[10.0, 8000.0, 0.3, 2.0],
        );
    }

    #[test]
    fn sigmoid_temperature_plateaus() {
        let (t0, temperature, temperature_amplitude, t_color) = (10.0, 8000.0, 0.3, 2.0);
        let mut jac = [0.0; 4];
        let early = sigmoid_temperature_jacobian(
            t0 - 1000.0,
            t0,
            temperature,
            temperature_amplitude,
            t_color,
            &mut jac,
        );
        let late = sigmoid_temperature_jacobian(
            t0 + 1000.0,
            t0,
            temperature,
            temperature_amplitude,
            t_color,
            &mut jac,
        );
        assert!((early - temperature * 1.3).abs() <= 1e-9 * temperature);
        assert!((late - temperature * 0.7).abs() <= 1e-9 * temperature);
    }

    #[test]
    fn delayed_sigmoid_jacobian_matches_finite_difference() {
        finite_diff_check(
            "delayed_sigmoid",
            5,
            |p, jac| delayed_sigmoid_temperature_jacobian(11.5, p[0], p[1], p[2], p[3], p[4], jac),
            &[10.0, 8000.0, 0.3, 2.0, 1.0],
        );
    }

    #[test]
    fn constant_temperature_is_constant() {
        let mut jac = [0.0; 1];
        assert_eq!(constant_temperature_jacobian(9000.0, &mut jac), 9000.0);
        assert_eq!(jac[0], 1.0);
    }
}
