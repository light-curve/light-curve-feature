use crate::evaluator::*;

macro_const! {
    const DOC: &str = r"
Lafler–Kinman string-length statistic

A smoothness measure for a phase-folded light curve. Observations are sorted
by time (i.e. by phase when applied to a phase-folded series) and the
normalised sum of squared successive differences is computed:

$$
\theta = \frac{\sum_{i=0}^{N-1} \bigl(m_{\pi(i+1 \bmod N)} - m_{\pi(i)}\bigr)^2}{2 (N-1) s^2},
$$

where the observations must arrive sorted by phase (time), $s^2 = (N-1)^{-1}\sum_i(m_i-\bar m)^2$
is the sample variance, and the sum wraps around (last observation is followed by the first).

With this normalisation $\langle\theta\rangle \approx 1$ for observations in random
magnitude order; a smooth phase-folded curve yields $\theta \ll 1$.

The input time series must be sorted by phase before evaluation. Intended to be applied to a
phase-folded series (time = phase).

- Depends on: **magnitude**
- Minimum number of observations: **2**
- Number of features: **1**

Lafler, J. & Kinman, T. D. 1965, *ApJS* **11**, 216
[ADS:1965ApJS...11..216L](https://ui.adsabs.harvard.edu/abs/1965ApJS...11..216L/abstract)
";
}

#[doc = DOC!()]
/// ### Example
/// ```
/// use light_curve_feature::*;
///
/// let lk = LaflerKinman::default();
/// // Phase-folded sine wave: phase in [0, 1), magnitude = sin(2π·phase)
/// let n = 32_usize;
/// let phase: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
/// let magn: Vec<f64> = phase.iter().map(|&p| (2.0 * std::f64::consts::PI * p).sin()).collect();
/// let mut ts = TimeSeries::new_without_weight(&phase, &magn);
/// let theta = lk.eval(&mut ts).unwrap()[0];
/// // Smooth phase-folded curve: string length well below 1
/// assert!(theta < 0.5, "theta = {theta}");
/// ```
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub struct LaflerKinman {}

impl LaflerKinman {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    LAFLER_KINMAN_INFO,
    LaflerKinman,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: true,
    variability_required: true,
);

impl FeatureNamesDescriptionsTrait for LaflerKinman {
    fn get_names(&self) -> Vec<&str> {
        vec!["lafler_kinman"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["Lafler-Kinman string-length statistic: smoothness of the phase-sorted light curve"]
    }
}

impl<T> FeatureEvaluator<T> for LaflerKinman
where
    T: Float,
{
    fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        let n = ts.lenu();
        let two_sum_sq_dev = T::two() * ts.m.get_std2() * (ts.lenf() - T::one());
        let m = ts.m.as_slice();

        let mut sum_sq = T::zero();
        for i in 0..n {
            let next = if i + 1 < n { i + 1 } else { 0 };
            let dm = m[next] - m[i];
            sum_sq += dm * dm;
        }

        Ok(vec![sum_sq / two_sum_sq_dev])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(LaflerKinman);

    // θ = Σ(m[i+1]-m[i])² / (2*(N-1)*s²)
    // m=[0,1,0]: sum_sq=2, s²=1/3, denom=4/3 → θ=1.5
    feature_test!(
        zigzag_exact,
        [LaflerKinman::new()],
        [1.5_f32],
        [0.0_f32, 1.0, 2.0],
        [0.0_f32, 1.0, 0.0],
    );

    // N=2: sum_sq=2*(m[1]-m[0])²=2, s²=0.5, denom=2*(N-1)*s²=1 → θ=2
    feature_test!(
        two_points,
        [LaflerKinman::new()],
        [2.0_f32],
        [0.0_f32, 1.0],
        [0.0_f32, 1.0],
    );

    // Alternating [0,1,0,1,...] → θ well above 1 (noisy, not smooth)
    #[test]
    fn alternating_is_noisy() {
        let n = 20_usize;
        let phase: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let magn: Vec<f64> = (0..n).map(|i| (i % 2) as f64).collect();
        let mut ts = TimeSeries::new_without_weight(&phase, &magn);
        let theta = LaflerKinman::new().eval(&mut ts).unwrap()[0];
        assert!(
            theta > 1.0,
            "expected theta > 1 for alternating signal, got {theta}"
        );
    }

    // Smooth sine wave → θ well below 1
    #[test]
    fn smooth_phase_folded_curve() {
        let n = 64_usize;
        let phase: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let magn: Vec<f64> = phase
            .iter()
            .map(|&p| (2.0 * std::f64::consts::PI * p).sin())
            .collect();
        let mut ts = TimeSeries::new_without_weight(&phase, &magn);
        let theta = LaflerKinman::new().eval(&mut ts).unwrap()[0];
        assert!(
            theta < 0.5,
            "expected theta < 0.5 for smooth sine, got {theta}"
        );
    }

    // Smooth is smaller than noisy for the same magnitude distribution
    #[test]
    fn smooth_smaller_than_shuffled() {
        use rand::prelude::*;
        let n = 64_usize;
        let phase: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let magn: Vec<f64> = phase
            .iter()
            .map(|&p| (2.0 * std::f64::consts::PI * p).sin())
            .collect();

        let theta_smooth = {
            let mut ts = TimeSeries::new_without_weight(&phase, &magn);
            LaflerKinman::new().eval(&mut ts).unwrap()[0]
        };

        // Shuffle magnitudes keeping phase order (breaks the smooth structure)
        let mut shuffled = magn.clone();
        let mut rng = StdRng::seed_from_u64(42);
        shuffled.shuffle(&mut rng);
        // Sort phase to keep sorting_required satisfied
        let theta_shuffled = {
            let mut ts = TimeSeries::new_without_weight(&phase, &shuffled);
            LaflerKinman::new().eval(&mut ts).unwrap()[0]
        };

        assert!(
            theta_smooth < theta_shuffled,
            "smooth θ={theta_smooth} should be less than shuffled θ={theta_shuffled}"
        );
    }
}
