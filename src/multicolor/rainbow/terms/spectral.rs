//! Spectral energy distribution (SED) terms for [RainbowFit](super::super::RainbowFit).
//!
//! Python's `BlanketedPlanckSpectralTerm` is deliberately not implemented here: its
//! `lambda_scale` parameter anchors to the temperature term's own (possibly different)
//! characteristic `T` parameter via cross-term parameter sharing (Python's `common_temp_spec`
//! machinery), which is more involved than a straight port of the other terms and is deferred.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::super::constants::{BOLTZMANN_K, PLANCK_H, SPEED_OF_LIGHT};

// ---------------------------------------------------------------------
// Planck (plain blackbody, no free parameters)
// ---------------------------------------------------------------------

fn planck(wave_cm: f64, temperature: f64) -> f64 {
    let nu = SPEED_OF_LIGHT / wave_cm;
    let x = PLANCK_H * nu / (BOLTZMANN_K * temperature);
    // B = (2h/c^2) nu^3 / (e^x - 1), written as nu^3 e^{-x} / (1 - e^{-x}) to stay overflow-safe
    // at large x (cold T / blue wavelengths): e^x/expm1(x) overflows there, whereas e^{-x}
    // underflows to 0 and B -> 0 (the Wien tail), which is the correct limit.
    let neg_expm1 = -(-x).exp_m1();
    (2.0 * PLANCK_H / (SPEED_OF_LIGHT * SPEED_OF_LIGHT)) * nu.powi(3) * (-x).exp() / neg_expm1
}

fn planck_dtemperature(wave_cm: f64, temperature: f64) -> f64 {
    let nu = SPEED_OF_LIGHT / wave_cm;
    let x = PLANCK_H * nu / (BOLTZMANN_K * temperature);
    let neg_expm1 = -(-x).exp_m1();
    let value = (2.0 * PLANCK_H / (SPEED_OF_LIGHT * SPEED_OF_LIGHT)) * nu.powi(3) * (-x).exp() / neg_expm1;
    // d(planck)/dT = planck * x*e^x / (T*expm1(x)) = planck * x / (T * (1 - e^{-x})), the same
    // overflow-safe rewrite as `planck` itself.
    value * x / (temperature * neg_expm1)
}

// ---------------------------------------------------------------------
// Generalized Wien: B(nu) ~ nu^3 * exp(-x^spec_k), x = h*nu/(k*T)
// ---------------------------------------------------------------------

fn genwien_x_value(wave_cm: f64, temperature: f64, spec_k: f64) -> (f64, f64) {
    let nu = SPEED_OF_LIGHT / wave_cm;
    let x = PLANCK_H * nu / (BOLTZMANN_K * temperature);
    let value = (2.0 * PLANCK_H / (SPEED_OF_LIGHT * SPEED_OF_LIGHT)) * nu.powi(3) * (-x.powf(spec_k)).exp();
    (x, value)
}

fn genwien(wave_cm: f64, temperature: f64, spec_k: f64) -> f64 {
    genwien_x_value(wave_cm, temperature, spec_k).1
}

fn genwien_dtemperature(wave_cm: f64, temperature: f64, spec_k: f64) -> f64 {
    let (x, value) = genwien_x_value(wave_cm, temperature, spec_k);
    value * spec_k * x.powf(spec_k) / temperature
}

fn genwien_jacobian(wave_cm: f64, temperature: f64, spec_k: f64, jac: &mut [f64]) -> f64 {
    let (x, value) = genwien_x_value(wave_cm, temperature, spec_k);
    jac[0] = -value * x.powf(spec_k) * x.ln();
    value
}

// ---------------------------------------------------------------------
// Modified blackbody: Planck(wave, T) * (wave/wave_ref)^beta
// ---------------------------------------------------------------------

const MODIFIED_BB_WAVE_REF_CM: f64 = 6000e-8;

fn modified_bb(wave_cm: f64, temperature: f64, beta: f64) -> f64 {
    let tilt = (wave_cm / MODIFIED_BB_WAVE_REF_CM).powf(beta);
    planck(wave_cm, temperature) * tilt
}

fn modified_bb_dtemperature(wave_cm: f64, temperature: f64, beta: f64) -> f64 {
    let tilt = (wave_cm / MODIFIED_BB_WAVE_REF_CM).powf(beta);
    planck_dtemperature(wave_cm, temperature) * tilt
}

fn modified_bb_jacobian(wave_cm: f64, temperature: f64, beta: f64, jac: &mut [f64]) -> f64 {
    let rel = wave_cm / MODIFIED_BB_WAVE_REF_CM;
    let value = planck(wave_cm, temperature) * rel.powf(beta);
    jac[0] = value * rel.ln();
    value
}

// ---------------------------------------------------------------------
// Log-parabola: Planck(wave, T) * exp(sp_a*L + sp_b*L^2), L = ln(wave/wave_ref)
// ---------------------------------------------------------------------

const LOGPARABOLA_WAVE_REF_CM: f64 = 6000e-8;

fn logparabola_l_fac(wave_cm: f64, sp_a: f64, sp_b: f64) -> (f64, f64) {
    let ell = (wave_cm / LOGPARABOLA_WAVE_REF_CM).ln();
    (ell, (sp_a * ell + sp_b * ell * ell).exp())
}

fn logparabola(wave_cm: f64, temperature: f64, sp_a: f64, sp_b: f64) -> f64 {
    let (_ell, fac) = logparabola_l_fac(wave_cm, sp_a, sp_b);
    planck(wave_cm, temperature) * fac
}

fn logparabola_dtemperature(wave_cm: f64, temperature: f64, sp_a: f64, sp_b: f64) -> f64 {
    let (_ell, fac) = logparabola_l_fac(wave_cm, sp_a, sp_b);
    planck_dtemperature(wave_cm, temperature) * fac
}

fn logparabola_jacobian(wave_cm: f64, temperature: f64, sp_a: f64, sp_b: f64, jac: &mut [f64]) -> f64 {
    let (ell, fac) = logparabola_l_fac(wave_cm, sp_a, sp_b);
    let value = planck(wave_cm, temperature) * fac;
    jac[0] = value * ell;
    jac[1] = value * ell * ell;
    value
}

// ---------------------------------------------------------------------
// Enum wrapper
// ---------------------------------------------------------------------

/// Which parametric spectral energy distribution (SED) describes the flux shape across bands at
/// fixed temperature $T$.
///
/// Every variant's amplitude is normalized away in [the model composition](super::super::model)
/// (divided by a Stefan-Boltzmann-based `norm(T)` shared across all variants), so only the SED's
/// *shape* -- not its absolute scale -- matters here; the overall bolometric amplitude is carried
/// entirely by the bolometric term. This also means the physical "bolometric luminosity"
/// interpretation of `amplitude` is only exact for [`Spectral::Planck`]; the other variants trade
/// that physical interpretation for extra shape flexibility.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum Spectral {
    /// Standard blackbody spectrum (Planck's law), no free parameters:
    /// $$
    /// B_\nu(\lambda, T) = \frac{2h}{c^2}\,\nu^3\,\frac{1}{\mathrm{e}^{h\nu/k_BT} - 1}, \quad \nu = c/\lambda.
    /// $$
    /// The physically-motivated default; use this unless the data show a clear deviation from a
    /// pure blackbody.
    Planck,
    /// Generalized-Wien SED, replacing the Planck denominator's exponent with a free power
    /// $\mathrm{spec\_k}$:
    /// $$
    /// B(\nu) \propto \nu^3\,\mathrm{e}^{-x^{\mathrm{spec\_k}}}, \quad x = h\nu/(k_BT).
    /// $$
    /// $\mathrm{spec\_k} \approx 1$ recovers the Wien tail (equal to Planck for cool sources,
    /// where $x$ is large); $\mathrm{spec\_k} > 1$ sharpens the blue cutoff, mimicking the UV
    /// deficit of hot sources. One parameter, `spec_k`. Note the fitted `T` under this term is
    /// *not* a physical temperature once `spec_k` deviates from 1 -- `spec_k` and `T` trade off,
    /// so treat `(T, spec_k)` jointly as an SED-shape descriptor rather than separately
    /// physically meaningful values.
    GenWien,
    /// Planck spectrum tilted by a wavelength power law:
    /// $$
    /// F(\lambda) = B_\nu(\lambda, T) \cdot (\lambda/\lambda_\mathrm{ref})^\beta, \quad \lambda_\mathrm{ref} = 6000\,\text{\AA}.
    /// $$
    /// $\beta = 0$ is exactly Planck, so `T` stays physical; $\beta > 0$ suppresses the blue
    /// (gentle UV blanketing), $\beta < 0$ enhances it. One parameter, `beta`. Because the
    /// deviation is a single tilt of an otherwise-intact Planck core, `beta` and `T` stay close
    /// to orthogonal, making this the best-conditioned of the non-Planck SEDs when a small,
    /// physically-motivated deviation is expected.
    ModifiedBlackBody,
    /// Planck spectrum modulated by a log-parabola in wavelength:
    /// $$
    /// F(\lambda) = B_\nu(\lambda, T) \cdot \mathrm{e}^{\mathrm{sp\_a} L + \mathrm{sp\_b} L^2}, \quad L = \ln(\lambda/\lambda_\mathrm{ref}), \quad \lambda_\mathrm{ref} = 6000\,\text{\AA}.
    /// $$
    /// Two parameters, `sp_a` (tilt) and `sp_b` (curvature), both anchored at the pure-Planck
    /// value 0. The most flexible of the deviation terms -- its curvature captures sharper blue
    /// cutoffs than [`Spectral::ModifiedBlackBody`]'s single tilt can -- at the cost of `(T,
    /// sp_a, sp_b)` being genuinely degenerate for smooth, close-to-blackbody spectra, where `T`
    /// can be biased without a prior pulling `sp_a`/`sp_b` back toward 0 (not yet implemented
    /// here; see the module-level documentation).
    LogParabola,
}

const PLANCK_PARAMS: [&str; 0] = [];
const GENWIEN_PARAMS: [&str; 1] = ["spec_k"];
const MODIFIED_BB_PARAMS: [&str; 1] = ["beta"];
const LOGPARABOLA_PARAMS: [&str; 2] = ["sp_a", "sp_b"];

impl Spectral {
    pub(crate) fn params(&self) -> &'static [&'static str] {
        match self {
            Spectral::Planck => &PLANCK_PARAMS,
            Spectral::GenWien => &GENWIEN_PARAMS,
            Spectral::ModifiedBlackBody => &MODIFIED_BB_PARAMS,
            Spectral::LogParabola => &LOGPARABOLA_PARAMS,
        }
    }

    pub(crate) fn n_params(&self) -> usize {
        self.params().len()
    }

    /// Evaluates the SED and writes its Jacobian w.r.t. this term's own local parameters
    /// (length `n_params()`, NOT including `T`) into `jac`, returning `(value, d(value)/dT)`.
    pub(crate) fn value_jac(&self, wave_cm: f64, temperature: f64, p: &[f64], jac: &mut [f64]) -> (f64, f64) {
        match self {
            Spectral::Planck => (planck(wave_cm, temperature), planck_dtemperature(wave_cm, temperature)),
            Spectral::GenWien => {
                let value = genwien_jacobian(wave_cm, temperature, p[0], jac);
                (value, genwien_dtemperature(wave_cm, temperature, p[0]))
            }
            Spectral::ModifiedBlackBody => {
                let value = modified_bb_jacobian(wave_cm, temperature, p[0], jac);
                (value, modified_bb_dtemperature(wave_cm, temperature, p[0]))
            }
            Spectral::LogParabola => {
                let value = logparabola_jacobian(wave_cm, temperature, p[0], p[1], jac);
                (value, logparabola_dtemperature(wave_cm, temperature, p[0], p[1]))
            }
        }
    }

    pub(crate) fn initial_guess(&self) -> Vec<f64> {
        match self {
            Spectral::Planck => vec![],
            Spectral::GenWien => vec![1.0],
            Spectral::ModifiedBlackBody => vec![0.0],
            Spectral::LogParabola => vec![0.0, 0.0],
        }
    }

    /// Box bounds `(lower, upper)` (physical units), same order as [`Self::params`]. None of
    /// these are data-dependent.
    pub(crate) fn bounds(&self) -> Vec<(f64, f64)> {
        match self {
            Spectral::Planck => vec![],
            Spectral::GenWien => vec![(0.3, 3.0)],
            Spectral::ModifiedBlackBody => vec![(-6.0, 10.0)],
            Spectral::LogParabola => vec![(-6.0, 6.0), (-4.0, 4.0)],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn finite_diff_check_local(name: &str, n: usize, value_jac: impl Fn(&[f64], &mut [f64]) -> f64, p: &[f64]) {
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

    fn finite_diff_check_dt(name: &str, value: impl Fn(f64) -> f64, dvalue: impl Fn(f64) -> f64, t: f64) {
        let h = 1e-2;
        let numeric = (value(t + h) - value(t - h)) / (2.0 * h);
        let analytic = dvalue(t);
        assert!(
            (analytic - numeric).abs() <= 1e-5 * numeric.abs().max(1.0),
            "{name}: analytic={analytic}, numeric={numeric}"
        );
    }

    const WAVE_CM: f64 = 5.0e-5;
    const TEMP: f64 = 9000.0;

    #[test]
    fn planck_dtemperature_matches_finite_difference() {
        finite_diff_check_dt("planck_dT", |t| planck(WAVE_CM, t), |t| planck_dtemperature(WAVE_CM, t), TEMP);
    }

    #[test]
    fn genwien_jacobian_matches_finite_difference() {
        finite_diff_check_local("genwien", 1, |p, jac| genwien_jacobian(WAVE_CM, TEMP, p[0], jac), &[1.3]);
        finite_diff_check_dt(
            "genwien_dT",
            |t| genwien(WAVE_CM, t, 1.3),
            |t| genwien_dtemperature(WAVE_CM, t, 1.3),
            TEMP,
        );
    }

    #[test]
    fn modified_bb_jacobian_matches_finite_difference() {
        finite_diff_check_local("modified_bb", 1, |p, jac| modified_bb_jacobian(WAVE_CM, TEMP, p[0], jac), &[0.5]);
        finite_diff_check_dt(
            "modified_bb_dT",
            |t| modified_bb(WAVE_CM, t, 0.5),
            |t| modified_bb_dtemperature(WAVE_CM, t, 0.5),
            TEMP,
        );
    }

    #[test]
    fn logparabola_jacobian_matches_finite_difference() {
        finite_diff_check_local(
            "logparabola",
            2,
            |p, jac| logparabola_jacobian(WAVE_CM, TEMP, p[0], p[1], jac),
            &[0.3, -0.2],
        );
        finite_diff_check_dt(
            "logparabola_dT",
            |t| logparabola(WAVE_CM, t, 0.3, -0.2),
            |t| logparabola_dtemperature(WAVE_CM, t, 0.3, -0.2),
            TEMP,
        );
    }

    #[test]
    fn modified_bb_beta_zero_equals_planck() {
        let a = modified_bb(WAVE_CM, TEMP, 0.0);
        let b = planck(WAVE_CM, TEMP);
        assert!((a - b).abs() <= 1e-9 * b);
    }
}
