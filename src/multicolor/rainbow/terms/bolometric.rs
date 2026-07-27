//! Bolometric (flux-vs-time envelope) terms for [RainbowFit](super::super::RainbowFit).
//!
//! `Linexp`, present in the Python reference implementation
//! (`light_curve_py.features.rainbow.bolometric.LinexpBolometricTerm`), is deliberately not
//! ported here: the Python docstring itself flags its guesses/limits as "not very stable", and a
//! seed sweep during development of this port confirmed it -- roughly half of random seeds land
//! the fit in a self-consistent-looking but wrong local optimum. Dropped rather than carried as a
//! known-flaky term; can be revisited if someone wants to stabilize its initial guess.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::common::{max_min, ptp, t0_and_weighted_centroid_sigma};

// ---------------------------------------------------------------------
// Bazin (symmetric, peak-normalized)
// ---------------------------------------------------------------------

fn bazin(t: f64, t0: f64, amplitude: f64, rise_time: f64, fall_time: f64) -> f64 {
    let dt = t - t0;
    if !(dt > -100.0 * rise_time && dt < 100.0 * fall_time) {
        return 0.0;
    }
    let scale = bazin_scale(rise_time, fall_time);
    amplitude * scale / ((-dt / rise_time).exp() + (dt / fall_time).exp())
}

fn bazin_scale(rise_time: f64, fall_time: f64) -> f64 {
    let alpha = fall_time / rise_time;
    let tau = rise_time + fall_time;
    let u = rise_time / tau;
    let v = fall_time / tau;
    alpha.powf(u) + alpha.powf(-v)
}

fn bazin_jacobian(t: f64, t0: f64, amplitude: f64, rise_time: f64, fall_time: f64, jac: &mut [f64]) -> f64 {
    let dt = t - t0;
    if !(dt > -100.0 * rise_time && dt < 100.0 * fall_time) {
        jac[..4].fill(0.0);
        return 0.0;
    }

    let e_r = (-dt / rise_time).exp();
    let e_f = (dt / fall_time).exp();
    let denom = e_r + e_f;

    let alpha = fall_time / rise_time;
    let tau = rise_time + fall_time;
    let u = rise_time / tau;
    let v = fall_time / tau;
    let log_alpha = alpha.ln();
    let a1 = alpha.powf(u);
    let a2 = alpha.powf(-v);
    let scale = a1 + a2;
    let dscale_dr = a1 * (v * log_alpha / tau - u / rise_time) + a2 * (v * log_alpha / tau + v / rise_time);
    let dscale_df = a1 * u * (1.0 / fall_time - log_alpha / tau) + a2 * (-u * log_alpha / tau - v / fall_time);

    let b = amplitude * scale / denom;
    let value = b;

    // d(value)/d(t0)
    jac[0] = (b / denom) * (e_f / fall_time - e_r / rise_time);
    // d(value)/d(amplitude)
    jac[1] = scale / denom;
    // d(value)/d(rise_time)
    jac[2] = b * (dscale_dr / scale - e_r * dt / (rise_time * rise_time * denom));
    // d(value)/d(fall_time)
    jac[3] = b * (dscale_df / scale + e_f * dt / (fall_time * fall_time * denom));

    value
}

// ---------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------

fn sigmoid_bol(t: f64, t0: f64, amplitude: f64, rise_time: f64) -> f64 {
    let dt = t - t0;
    if dt <= -100.0 * rise_time {
        return 0.0;
    }
    amplitude / ((-dt / rise_time).exp() + 1.0)
}

fn sigmoid_bol_jacobian(t: f64, t0: f64, amplitude: f64, rise_time: f64, jac: &mut [f64]) -> f64 {
    let dt = t - t0;
    if dt <= -100.0 * rise_time {
        jac[..3].fill(0.0);
        return 0.0;
    }
    let e = (-dt / rise_time).exp();
    let s = 1.0 / (e + 1.0);
    let s_1ms = e * s * s; // s * (1 - s)

    jac[0] = -amplitude * s_1ms / rise_time;
    jac[1] = s;
    jac[2] = -amplitude * s_1ms * dt / (rise_time * rise_time);

    amplitude * s
}

// ---------------------------------------------------------------------
// Doublexp
// ---------------------------------------------------------------------

fn doublexp_bol(t: f64, t0: f64, amplitude: f64, time1: f64, time2: f64, p: f64) -> f64 {
    let dt = t - t0;
    let v = (-dt / time2).exp();
    let a_inner = -(dt / time1) * (p - v);
    let a_inner = a_inner.min(20.0);
    amplitude * a_inner.exp()
}

fn doublexp_bol_jacobian(
    t: f64,
    t0: f64,
    amplitude: f64,
    time1: f64,
    time2: f64,
    p: f64,
    jac: &mut [f64],
) -> f64 {
    let dt = t - t0;
    let v = (-dt / time2).exp();
    let a_inner = -(dt / time1) * (p - v);
    let maxp = 20.0;
    let clamped = a_inner > maxp;
    let b = amplitude * a_inner.min(maxp).exp();

    jac[1] = b / amplitude;

    if clamped {
        // Beyond the exponent clamp, the model is constant in every parameter but amplitude.
        jac[0] = 0.0;
        jac[2] = 0.0;
        jac[3] = 0.0;
        jac[4] = 0.0;
    } else {
        let da_dt0 = (p - v) / time1 + dt * v / (time1 * time2);
        let da_dtime1 = dt * (p - v) / (time1 * time1);
        let da_dtime2 = (dt * dt) * v / (time1 * time2 * time2);
        let da_dp = -dt / time1;

        jac[0] = b * da_dt0;
        jac[2] = b * da_dtime1;
        jac[3] = b * da_dtime2;
        jac[4] = b * da_dp;
    }

    b
}

/// Principal branch (`W0`) of the Lambert W function for real `x >= -1/e`, via Halley's method.
/// Only needed for [`Bolometric::peak_time`] of the Doublexp term; not exposed as a general
/// utility since it isn't validated outside the range this needs (`x >= 0`, given `p > 0`).
fn lambert_w0(x: f64) -> f64 {
    if x == 0.0 {
        return 0.0;
    }
    let mut w = if x < 10.0 { (x + 1.0).ln().max(1e-4) } else { x.ln() - x.ln().ln() };
    for _ in 0..50 {
        let ew = w.exp();
        let f = w * ew - x;
        let denom = ew * (w + 1.0) - (w + 2.0) * f / (2.0 * w + 2.0);
        if denom == 0.0 {
            break;
        }
        let step = f / denom;
        w -= step;
        if step.abs() < 1e-14 * w.abs().max(1.0) {
            break;
        }
    }
    w
}

// ---------------------------------------------------------------------
// Enum wrapper
// ---------------------------------------------------------------------

/// Which parametric function models the bolometric (band-integrated) flux envelope $\mathrm{bol}(t)$.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum Bolometric {
    /// Symmetric, peak-normalized Bazin function (Bazin et al. 2009,
    /// [DOI:10.1051/0004-6361/200911847](https://doi.org/10.1051/0004-6361/200911847)):
    /// $$
    /// \mathrm{bol}(t) = A \cdot \frac{\alpha}{\mathrm{e}^{-(t-t_0)/\tau_\mathrm{rise}} + \mathrm{e}^{(t-t_0)/\tau_\mathrm{fall}}},
    /// $$
    /// where $\alpha$ is chosen so the peak value equals $A$ exactly:
    /// $$
    /// \alpha = \left(\frac{\tau_\mathrm{fall}}{\tau_\mathrm{rise}}\right)^{\tau_\mathrm{rise}/(\tau_\mathrm{rise}+\tau_\mathrm{fall})}
    ///        + \left(\frac{\tau_\mathrm{fall}}{\tau_\mathrm{rise}}\right)^{-\tau_\mathrm{fall}/(\tau_\mathrm{rise}+\tau_\mathrm{fall})}.
    /// $$
    /// Parameters: `reference_time` ($t_0$), `amplitude` ($A$), `rise_time` ($\tau_\mathrm{rise}$),
    /// `fall_time` ($\tau_\mathrm{fall}$). A good default for transients with a clear rise and
    /// fall (e.g. supernovae).
    Bazin,
    /// Plain logistic rise, with no decline:
    /// $$
    /// \mathrm{bol}(t) = \frac{A}{1 + \mathrm{e}^{-(t-t_0)/\tau_\mathrm{rise}}}.
    /// $$
    /// Parameters: `reference_time` ($t_0$, the inflection point), `amplitude` ($A$, the
    /// asymptotic plateau level), `rise_time` ($\tau_\mathrm{rise}$). Appropriate for sources
    /// that rise and then plateau within the observed window, rather than declining (e.g. AGN,
    /// TDEs observed only near onset).
    Sigmoid,
    /// Asymmetric rise/decline with an adjustable decline sharpness, fitted by symbolic
    /// regression on ZTF SN Ia light curves (Russeil et al. 2024,
    /// [arXiv:2402.04298](https://arxiv.org/abs/2402.04298)):
    /// $$
    /// \mathrm{bol}(t) = A \, \exp\!\left(-\frac{t-t_0}{\tau_1}\left(p - \mathrm{e}^{-(t-t_0)/\tau_2}\right)\right).
    /// $$
    /// Parameters: `reference_time` ($t_0$), `amplitude` ($A$), `time1` ($\tau_1$), `time2`
    /// ($\tau_2$), `p`. Unlike Bazin, decline sharpness is a free parameter ($p$) rather than
    /// tied to the same functional form as the rise, at the cost of two extra parameters.
    Doublexp,
}

const BAZIN_PARAMS: [&str; 4] = ["reference_time", "amplitude", "rise_time", "fall_time"];
const SIGMOID_BOL_PARAMS: [&str; 3] = ["reference_time", "amplitude", "rise_time"];
const DOUBLEXP_PARAMS: [&str; 5] = ["reference_time", "amplitude", "time1", "time2", "p"];

impl Bolometric {
    pub(crate) fn params(&self) -> &'static [&'static str] {
        match self {
            Bolometric::Bazin => &BAZIN_PARAMS,
            Bolometric::Sigmoid => &SIGMOID_BOL_PARAMS,
            Bolometric::Doublexp => &DOUBLEXP_PARAMS,
        }
    }

    pub(crate) fn n_params(&self) -> usize {
        self.params().len()
    }

    /// Evaluates the term and writes its Jacobian (length `n_params()`) into `jac`, returning
    /// the value.
    pub(crate) fn value_jac(&self, t: f64, p: &[f64], jac: &mut [f64]) -> f64 {
        match self {
            Bolometric::Bazin => bazin_jacobian(t, p[0], p[1], p[2], p[3], jac),
            Bolometric::Sigmoid => sigmoid_bol_jacobian(t, p[0], p[1], p[2], jac),
            Bolometric::Doublexp => doublexp_bol_jacobian(t, p[0], p[1], p[2], p[3], p[4], jac),
        }
    }

    /// Heuristic initial guess (physical units), same order as [`Self::params`].
    pub(crate) fn initial_guess(&self, t: &[f64], flux: &[f64], flux_err: &[f64]) -> Vec<f64> {
        let (max_flux, min_flux) = max_min(flux);
        let ptp_flux = max_flux - min_flux;
        match self {
            Bolometric::Bazin => {
                let (t0, dt) = t0_and_weighted_centroid_sigma(t, flux, flux_err);
                vec![t0, 1.5 * max_flux.max(ptp_flux), dt, dt]
            }
            Bolometric::Sigmoid => {
                let peak_t = t[super::common::argmax(flux)];
                vec![peak_t, ptp_flux, 1.0]
            }
            Bolometric::Doublexp => {
                let (t0, dt) = t0_and_weighted_centroid_sigma(t, flux, flux_err);
                vec![t0, max_flux.max(ptp_flux), 2.0 * dt, 2.0 * dt, 1.0]
            }
        }
    }

    /// Box bounds `(lower, upper)` (physical units), same order as [`Self::params`].
    pub(crate) fn bounds(&self, t: &[f64], flux: &[f64], flux_err: &[f64]) -> Vec<(f64, f64)> {
        let (t_max, t_min) = max_min(t);
        let ptp_t = t_max - t_min;
        let ptp_flux = ptp(flux);
        let reference_time_bounds = (t_min - 10.0 * ptp_t, t_max + 10.0 * ptp_t);

        match self {
            Bolometric::Bazin => {
                let (_, dt) = t0_and_weighted_centroid_sigma(t, flux, flux_err);
                vec![
                    reference_time_bounds,
                    (0.0, 20.0 * ptp_flux),
                    (dt / 100.0, 10.0 * ptp_t),
                    (dt / 100.0, 10.0 * ptp_t),
                ]
            }
            Bolometric::Sigmoid => {
                let (_, dt) = t0_and_weighted_centroid_sigma(t, flux, flux_err);
                vec![reference_time_bounds, (0.0, 20.0 * ptp_flux), (dt / 100.0, 10.0 * ptp_t)]
            }
            Bolometric::Doublexp => {
                let (_, dt) = t0_and_weighted_centroid_sigma(t, flux, flux_err);
                vec![
                    reference_time_bounds,
                    (0.0, 10.0 * ptp_flux),
                    (dt / 10.0, 2.0 * ptp_t),
                    (dt / 10.0, 2.0 * ptp_t),
                    (1e-2, 100.0),
                ]
            }
        }
    }

    /// Analytic bolometric peak time, if defined for this term.
    pub(crate) fn peak_time(&self, p: &[f64]) -> Option<f64> {
        match self {
            Bolometric::Bazin => {
                let (t0, _a, rise_time, fall_time) = (p[0], p[1], p[2], p[3]);
                Some(t0 + (fall_time / rise_time).ln() * rise_time * fall_time / (rise_time + fall_time))
            }
            // Peak time is not defined for the sigmoid (monotonic), so it returns the
            // inflection point (mid-time of the rise) instead.
            Bolometric::Sigmoid => Some(p[0]),
            Bolometric::Doublexp => {
                let (t0, _a, _time1, _time2, pp) = (p[0], p[1], p[2], p[3], p[4]);
                Some(t0 + (-lambert_w0(pp * std::f64::consts::E) + 1.0))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_diff_check(name: &str, n: usize, value_jac: impl Fn(&[f64], &mut [f64]) -> f64, p: &[f64]) {
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
    fn bazin_jacobian_matches_finite_difference() {
        finite_diff_check(
            "bazin",
            4,
            |p, jac| bazin_jacobian(12.0, p[0], p[1], p[2], p[3], jac),
            &[10.0, 5.0, 3.0, 20.0],
        );
    }

    #[test]
    fn bazin_peak_equals_amplitude() {
        let (t0, amplitude, rise_time, fall_time): (f64, f64, f64, f64) = (100.0, 5.0, 3.0, 20.0);
        let peak_t = t0 + (fall_time / rise_time).ln() * rise_time * fall_time / (rise_time + fall_time);
        let value = bazin(peak_t, t0, amplitude, rise_time, fall_time);
        assert!((value - amplitude).abs() <= 1e-9 * amplitude);
    }

    #[test]
    fn sigmoid_bol_jacobian_matches_finite_difference() {
        finite_diff_check(
            "sigmoid_bol",
            3,
            |p, jac| sigmoid_bol_jacobian(12.0, p[0], p[1], p[2], jac),
            &[10.0, 5.0, 3.0],
        );
    }

    #[test]
    fn doublexp_bol_jacobian_matches_finite_difference() {
        finite_diff_check(
            "doublexp_bol",
            5,
            |p, jac| doublexp_bol_jacobian(12.0, p[0], p[1], p[2], p[3], p[4], jac),
            &[10.0, 5.0, 3.0, 4.0, 1.5],
        );
    }

    #[test]
    fn lambert_w0_matches_definition() {
        for &x in &[0.1, 1.0, 2.0, 10.0, 100.0, 0.001] {
            let w = lambert_w0(x);
            let check = w * w.exp();
            assert!((check - x).abs() <= 1e-8 * x.max(1.0), "x={x}, w={w}, w*e^w={check}");
        }
    }
}
