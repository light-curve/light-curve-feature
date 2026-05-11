use crate::evaluator::*;
use crate::periodogram::{AverageNyquistFreq, FreqGridStrategy, FreqGridTrait};

macro_const! {
    const DOC: &str = r"
Minimum Lafler–Kinman string-length statistic

The string length at trial period $P$ is

$$
\theta(P) = \frac{\sum_{i=0}^{N-1} \bigl(m_{\pi(i+1 \bmod N)} - m_{\pi(i)}\bigr)^2}{2 (N-1) s^2},
$$

where $\pi$ sorts the observations by phase $\phi_j = \{t_j / P\}$ ($\{\cdot\}$ is the fractional
part), $s^2 = (N-1)^{-1} \sum_i (m_i - \bar m)^2$ is the sample variance of the magnitudes, and
the sum wraps around ($\pi(N) \equiv \pi(0)$).

With this normalisation $\langle\theta\rangle \approx 1$ for observations drawn from a stationary
random process, and strongly periodic signals yield $\theta \ll 1$.

The trial-frequency grid follows the same [FreqGridStrategy] as [Periodogram](crate::Periodogram).

- Depends on: **time**, **magnitude**
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
/// let lk = LaflerKinman::<f64>::default();
/// // A sine wave: times span one period
/// let n = 32_usize;
/// let period = 1.0;
/// let time: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * period).collect();
/// let magn: Vec<f64> = time.iter().map(|&t| (2.0 * std::f64::consts::PI * t).sin()).collect();
/// let mut ts = TimeSeries::new_without_weight(&time, &magn);
/// let theta = lk.eval(&mut ts).unwrap()[0];
/// // Periodic signal: string length well below 1
/// assert!(theta < 0.5, "theta = {theta}");
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(
    from = "LaflerKinmanParameters<T>",
    into = "LaflerKinmanParameters<T>",
    bound = "T: Float"
)]
pub struct LaflerKinman<T>
where
    T: Float,
{
    freq_grid_strategy: FreqGridStrategy<T>,
}

impl<T> LaflerKinman<T>
where
    T: Float,
{
    pub fn new(freq_grid_strategy: impl Into<FreqGridStrategy<T>>) -> Self {
        Self {
            freq_grid_strategy: freq_grid_strategy.into(),
        }
    }

    pub fn default_resolution() -> f32 {
        10.0
    }

    pub fn default_max_freq_factor() -> f32 {
        1.0
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

impl<T> Default for LaflerKinman<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(FreqGridStrategy::dynamic(
            Self::default_resolution(),
            Self::default_max_freq_factor(),
            AverageNyquistFreq,
        ))
    }
}

lazy_info!(
    LAFLER_KINMAN_INFO,
    LaflerKinman<T>,
    T,
    size: 1,
    min_ts_length: 2,
    t_required: true,
    m_required: true,
    w_required: false,
    sorting_required: true,
    variability_required: true,
);

impl<T> FeatureNamesDescriptionsTrait for LaflerKinman<T>
where
    T: Float,
{
    fn get_names(&self) -> Vec<&str> {
        vec!["lafler_kinman"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["minimum Lafler-Kinman string-length statistic over the trial-period grid"]
    }
}

impl<T> FeatureEvaluator<T> for LaflerKinman<T>
where
    T: Float,
{
    fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        let n = ts.lenu();
        let two_sum_sq_dev = T::two() * ts.m.get_std2() * (ts.lenf() - T::one());
        // Clone to owned Vecs: freq_grid borrows &t while the loop borrows m.
        let t: Vec<T> = ts.t.as_slice().to_vec();
        let m: Vec<T> = ts.m.as_slice().to_vec();

        let freq_grid = self.freq_grid_strategy.freq_grid(&t, false);

        let mut indices: Vec<usize> = (0..n).collect();
        let mut phases: Vec<T> = vec![T::zero(); n];

        let mut min_theta = T::infinity();

        for k in 0..freq_grid.size() {
            let freq = freq_grid.get(k);
            if freq.is_zero() {
                continue;
            }
            let period = T::two() * T::PI() / freq;

            for i in 0..n {
                let phi = (t[i] / period).fract();
                phases[i] = if phi < T::zero() { phi + T::one() } else { phi };
            }

            indices.sort_unstable_by(|&a, &b| {
                phases[a]
                    .partial_cmp(&phases[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut sum_sq = T::zero();
            for i in 0..n {
                let next = if i + 1 < n { i + 1 } else { 0 };
                let dm = m[indices[next]] - m[indices[i]];
                sum_sq += dm * dm;
            }

            let theta = sum_sq / two_sum_sq_dev;
            if theta < min_theta {
                min_theta = theta;
            }
        }

        Ok(vec![min_theta])
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "LaflerKinman", bound = "T: Float")]
struct LaflerKinmanParameters<T>
where
    T: Float,
{
    freq_grid_strategy: FreqGridStrategy<T>,
}

impl<T> From<LaflerKinman<T>> for LaflerKinmanParameters<T>
where
    T: Float,
{
    fn from(f: LaflerKinman<T>) -> Self {
        Self {
            freq_grid_strategy: f.freq_grid_strategy,
        }
    }
}

impl<T> From<LaflerKinmanParameters<T>> for LaflerKinman<T>
where
    T: Float,
{
    fn from(p: LaflerKinmanParameters<T>) -> Self {
        Self::new(p.freq_grid_strategy)
    }
}

impl<T> JsonSchema for LaflerKinman<T>
where
    T: Float,
{
    json_schema!(LaflerKinmanParameters<T>, false);
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(LaflerKinman<f64>);

    #[test]
    fn periodic_signal_detected() {
        let n = 64_usize;
        let period = 1.0_f64;
        let t: Vec<f64> = (0..n).map(|i| i as f64 / n as f64 * period).collect();
        let m: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti).sin())
            .collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let theta = LaflerKinman::default().eval(&mut ts).unwrap()[0];
        assert!(
            theta < 0.5,
            "expected theta < 0.5 for periodic signal, got {theta}"
        );
    }
}
