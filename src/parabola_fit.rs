use crate::data::TimeSeries;
use crate::float_trait::Float;

use ndarray::Zip;

// Weighted least-squares fit of a parabola m(t) = m0 + g (t - t0)^2.
//
// We fit the equivalent quadratic m(t) = A t^2 + B t + C by minimising
// sum_i w_i (m_i - A t_i^2 - B t_i - C)^2, then report the physical parameters
// g = A (curvature) and m0 = C - B^2 / (4 A) (value at the extremum). The
// extremum position t0 = -B / (2 A) is not reported: it is an absolute time,
// which is a poor classification feature and diverges as g -> 0.
//
// Time is centred on its weighted mean before accumulating the moment sums to
// avoid catastrophic cancellation in the t^4 terms for large timestamps (e.g.
// MJD). Centring is an exact change of variable: g and m0 are invariant under
// the shift.

/// Result of [fit_parabola].
pub struct ParabolaFitterResult<T> {
    /// Curvature `g` (coefficient of the quadratic term).
    pub g: T,
    /// Value at the extremum `m0 = C - B^2 / (4 g)`.
    pub m0: T,
    /// Reduced chi-squared, `chi2 / (N - 3)`.
    pub reduced_chi2: T,
}

/// Determinant of a 3x3 matrix.
fn det3<T: Float>(m: &[[T; 3]; 3]) -> T {
    m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
}

/// Solve the symmetric 3x3 system `M x = r` by Cramer's rule.
fn solve3<T: Float>(m: &[[T; 3]; 3], r: &[T; 3]) -> [T; 3] {
    let det = det3(m);
    let mut x = [T::zero(); 3];
    for (col, value) in x.iter_mut().enumerate() {
        let mut mc = *m;
        for (row, mc_row) in mc.iter_mut().enumerate() {
            mc_row[col] = r[row];
        }
        *value = det3(&mc) / det;
    }
    x
}

/// Weighted least-squares fit of the parabola `m = m0 + g (t - t0)^2`.
///
/// Weights are interpreted as inverse squared observation errors, so
/// `reduced_chi2` is normalised by `N - 3`.
#[allow(clippy::many_single_char_names)]
pub fn fit_parabola<T: Float>(ts: &TimeSeries<T>) -> ParabolaFitterResult<T> {
    let n = ts.lenf();

    // Weighted mean of time, used to centre the time axis.
    let (sw, swt) = Zip::from(&ts.t.sample)
        .and(&ts.w.sample)
        .fold((T::zero(), T::zero()), |(sw, swt), &t, &w| {
            (sw + w, swt + w * t)
        });
    let t_mean = swt / sw;

    // Moment sums in centred time tau = t - t_mean give the normal matrix (for
    // the basis [tau^2, tau, 1]) and right-hand side.  The accumulators are
    // confined to this scope so the returned matrix and vector stay immutable.
    // s1 == sum(w * tau) is zero up to round-off, but we accumulate it anyway
    // and solve the general system.
    let (matrix, vector) = {
        let mut s1 = T::zero();
        let mut s2 = T::zero();
        let mut s3 = T::zero();
        let mut s4 = T::zero();
        let mut r0 = T::zero(); // sum(w * tau^2 * m)
        let mut r1 = T::zero(); // sum(w * tau * m)
        let mut r2 = T::zero(); // sum(w * m)
        Zip::from(&ts.t.sample)
            .and(&ts.m.sample)
            .and(&ts.w.sample)
            .for_each(|&t, &m, &w| {
                let tau = t - t_mean;
                let wtau = w * tau;
                let wtau2 = wtau * tau;
                s1 += wtau;
                s2 += wtau2;
                s3 += wtau2 * tau;
                s4 += wtau2 * tau * tau;
                r0 += wtau2 * m;
                r1 += wtau * m;
                r2 += w * m;
            });
        ([[s4, s3, s2], [s3, s2, s1], [s2, s1, sw]], [r0, r1, r2])
    };
    let [a, b, c] = solve3(&matrix, &vector);

    let g = a;
    let m0 = c - b.powi(2) / (T::four() * a);

    let chi2 = Zip::from(&ts.t.sample)
        .and(&ts.m.sample)
        .and(&ts.w.sample)
        .fold(T::zero(), |chi2, &t, &m, &w| {
            let tau = t - t_mean;
            let model = a * tau.powi(2) + b * tau + c;
            chi2 + w * (m - model).powi(2)
        });
    let reduced_chi2 = chi2 / (n - T::three());

    ParabolaFitterResult {
        g,
        m0,
        reduced_chi2,
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use light_curve_common::all_close;

    #[test]
    fn exact_parabola() {
        // m = 2 (t - 3)^2 + 5  =>  g = 2, m0 = 5, exact fit (extremum at t = 3).
        let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let m: Vec<f64> = t.iter().map(|&t| 2.0 * (t - 3.0).powi(2) + 5.0).collect();
        let ts = TimeSeries::new_without_weight(&t, &m);
        let result = fit_parabola(&ts);
        all_close(&[result.g], &[2.0], 1e-10);
        all_close(&[result.m0], &[5.0], 1e-10);
        all_close(&[result.reduced_chi2], &[0.0], 1e-10);
    }

    #[test]
    fn linear_has_no_finite_extremum() {
        // Perfectly linear data has zero curvature, so the parabola extremum is
        // at infinity: g == 0 and m0 is non-finite.  The fit must not panic
        // (float division by zero yields inf, not a panic).
        let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0];
        let m: Vec<f64> = t.iter().map(|&t| 2.0 * t + 1.0).collect();
        let result = fit_parabola(&TimeSeries::new_without_weight(&t, &m));
        assert_eq!(result.g, 0.0);
        assert!(!result.m0.is_finite());
    }

    #[test]
    fn weighted_parabola() {
        let t = [0.5_f64, 1.5, 2.5, 5.0, 7.0, 16.0];
        let m = [-1.0_f64, 3.0, 2.0, 6.0, 10.0, 25.0];
        let w = [2.0_f64, 1.0, 3.0, 10.0, 1.0, 0.4];
        let ts = TimeSeries::new(&t, &m, &w);
        // numpy: a, b, c = np.polyfit(t, m, 2, w=np.sqrt(w))
        //   g = a; m0 = c - b**2 / (4 a)
        //   chi2 = sum(w * (m - (a t^2 + b t + c))**2); reduced = chi2 / (6 - 3)
        let desired_g = 0.016123168785128206;
        let desired_m0 = -31.173339538115183;
        let desired_reduced_chi2 = 1.9657090723904733;
        let result = fit_parabola(&ts);
        all_close(&[result.g], &[desired_g], 1e-6);
        all_close(&[result.m0], &[desired_m0], 1e-4);
        all_close(&[result.reduced_chi2], &[desired_reduced_chi2], 1e-6);
    }
}
