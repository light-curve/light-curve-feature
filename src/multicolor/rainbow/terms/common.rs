//! Shared initial-guess heuristics used by more than one bolometric/temperature term.

/// Weighted centroid time and its spread, used as an initial guess for
/// `reference_time` and various rise/fall/color timescales.
///
/// The centroid is computed only from points above the median flux (i.e. roughly the
/// "bright half" of the light curve), weighted by `flux / flux_err`, so it approximates the
/// peak position without needing to know the model shape in advance. The spread is the
/// flux-weighted standard deviation of time around that centroid, used as a starting guess for
/// timescale parameters (rise/fall/color).
pub fn t0_and_weighted_centroid_sigma(t: &[f64], flux: &[f64], flux_err: &[f64]) -> (f64, f64) {
    let min_flux = flux.iter().cloned().fold(f64::MAX, f64::min);
    let mc: Vec<f64> = flux.iter().map(|m| m - min_flux).collect();

    let median = median(flux);
    let idx: Vec<usize> = (0..t.len()).filter(|&i| flux[i] > median).collect();

    let weight_sum: f64 = idx.iter().map(|&i| flux[i] / flux_err[i]).sum();
    let t0 = idx.iter().map(|&i| t[i] * flux[i] / flux_err[i]).sum::<f64>() / weight_sum;

    let mc_weight_sum: f64 = idx.iter().map(|&i| mc[i] / flux_err[i]).sum();
    let var = idx.iter().map(|&i| (t[i] - t0).powi(2) * mc[i] / flux_err[i]).sum::<f64>() / mc_weight_sum;
    let dt = var.sqrt();

    // Guard against a degenerate spread (e.g. all bright points at the same time): fall back to
    // a quarter of the light curve's time span, which is always positive and finite.
    let dt = if dt.is_finite() && dt > 0.0 {
        dt
    } else {
        let t_max = t.iter().cloned().fold(f64::MIN, f64::max);
        let t_min = t.iter().cloned().fold(f64::MAX, f64::min);
        ((t_max - t_min) / 4.0).max(1.0)
    };

    (t0, dt)
}

pub fn max_min(x: &[f64]) -> (f64, f64) {
    (x.iter().cloned().fold(f64::MIN, f64::max), x.iter().cloned().fold(f64::MAX, f64::min))
}

pub fn ptp(x: &[f64]) -> f64 {
    let (max, min) = max_min(x);
    max - min
}

pub fn argmax(x: &[f64]) -> usize {
    let mut best = 0;
    for i in 1..x.len() {
        if x[i] > x[best] {
            best = i;
        }
    }
    best
}

pub fn median(x: &[f64]) -> f64 {
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    if n % 2 == 1 { sorted[n / 2] } else { 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) }
}
