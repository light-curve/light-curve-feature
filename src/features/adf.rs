use crate::evaluator::*;

macro_const! {
    const DOC: &str = r"
Augmented Dickey-Fuller (ADF) test statistic

The ADF test is used to determine whether a time series is stationary.
The test statistic is calculated from the following regression:

$$
\Delta y_t = \alpha + \gamma y_{t-1} + \sum_{i=1}^{p} \theta_i \Delta y_{t-i} + \varepsilon_t,
$$

where $y_t$ is the magnitude at time $t$, $\Delta y_t = y_t - y_{t-1}$ is the first difference,
$\alpha$ is a constant, $\gamma$ is the coefficient of interest,
$p$ is the number of lags (automatically selected), and $\varepsilon_t$ is the error term.

The test statistic is:
$$
\mathrm{ADF} = \frac{\hat{\gamma}}{\mathrm{SE}(\hat{\gamma})},
$$
where $\hat{\gamma}$ is the estimated coefficient of $y_{t-1}$ and $\mathrm{SE}(\hat{\gamma})$ is its standard error.

The implementation uses a simple lag selection based on the number of observations:
$p = \min(10, \lfloor (N-1)^{1/3} \rfloor)$, where $N$ is the number of observations.

More negative values indicate stronger evidence of stationarity.

- Depends on: **magnitude** (sorted by time if time is provided)
- Minimum number of observations: **4**
- Number of features: **1**

[Wikipedia](https://en.wikipedia.org/wiki/Augmented_Dickey–Fuller_test)
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub struct Adf {}

impl Adf {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }

    /// Select optimal number of lags based on sample size
    /// Using the rule: p = min(10, floor((N-1)^(1/3)))
    fn select_lags(n: usize) -> usize {
        if n < 4 {
            return 0;
        }
        let max_lag = ((n - 1) as f64).powf(1.0 / 3.0).floor() as usize;
        max_lag.clamp(1, 10)
    }

    /// Perform OLS regression to compute ADF statistic
    /// Returns the t-statistic for the lagged level coefficient
    fn compute_adf_statistic<T: Float>(y: &[T], lags: usize) -> T {
        let n = y.len();
        let n_obs = n - lags - 1;

        if n_obs < 2 {
            return T::zero();
        }

        // Build regression matrices
        // Y = first differences starting from index lags+1
        // X = [1, y_{t-1}, Delta y_{t-1}, ..., Delta y_{t-p}]

        let mut y_diff = Vec::with_capacity(n_obs);
        let mut y_lag = Vec::with_capacity(n_obs);
        let mut x_lags = vec![Vec::with_capacity(n_obs); lags];

        for i in (lags + 1)..n {
            // Dependent variable: Delta y_t = y_t - y_{t-1}
            y_diff.push(y[i] - y[i - 1]);

            // y_{t-1}
            y_lag.push(y[i - 1]);

            // Lagged differences: Delta y_{t-j} for j=1..lags
            for (j, x_lag_vec) in x_lags.iter_mut().enumerate().take(lags) {
                let lag_idx = i - j - 1;
                x_lag_vec.push(y[lag_idx] - y[lag_idx - 1]);
            }
        }

        let n_obs_f = T::from_usize(n_obs).unwrap();

        // Compute means
        let mean_y_diff = y_diff.iter().copied().sum::<T>() / n_obs_f;
        let mean_y_lag = y_lag.iter().copied().sum::<T>() / n_obs_f;
        let mean_x_lags: Vec<T> = x_lags
            .iter()
            .map(|x| x.iter().copied().sum::<T>() / n_obs_f)
            .collect();

        // Center the variables
        let y_diff_centered: Vec<T> = y_diff.iter().map(|&v| v - mean_y_diff).collect();
        let y_lag_centered: Vec<T> = y_lag.iter().map(|&v| v - mean_y_lag).collect();
        let x_lags_centered: Vec<Vec<T>> = x_lags
            .iter()
            .zip(&mean_x_lags)
            .map(|(x, &mean)| x.iter().map(|&v| v - mean).collect())
            .collect();

        // Compute regression using Frisch-Waugh-Lovell theorem
        // First regress y_lag on x_lags and constant to get residuals
        let mut y_lag_resid = y_lag_centered.clone();

        // For each lag variable, partial out its effect
        for x_lag_centered in &x_lags_centered {
            let x_sum_sq = x_lag_centered.iter().map(|&v| v * v).sum::<T>();
            if x_sum_sq > T::epsilon() {
                let cov = y_lag_resid
                    .iter()
                    .zip(x_lag_centered.iter())
                    .map(|(&r, &x)| r * x)
                    .sum::<T>();
                let coef = cov / x_sum_sq;

                for i in 0..n_obs {
                    y_lag_resid[i] = y_lag_resid[i] - coef * x_lag_centered[i];
                }
            }
        }

        // Similarly partial out x_lags from y_diff
        let mut y_diff_resid = y_diff_centered.clone();

        for x_lag_centered in &x_lags_centered {
            let x_sum_sq = x_lag_centered.iter().map(|&v| v * v).sum::<T>();
            if x_sum_sq > T::epsilon() {
                let cov = y_diff_resid
                    .iter()
                    .zip(x_lag_centered.iter())
                    .map(|(&r, &x)| r * x)
                    .sum::<T>();
                let coef = cov / x_sum_sq;

                for i in 0..n_obs {
                    y_diff_resid[i] = y_diff_resid[i] - coef * x_lag_centered[i];
                }
            }
        }

        // Now compute the coefficient of y_lag on y_diff after partialing out other variables
        let y_lag_resid_sum_sq = y_lag_resid.iter().map(|&v| v * v).sum::<T>();

        if y_lag_resid_sum_sq <= T::epsilon() {
            return T::zero();
        }

        let gamma = y_diff_resid
            .iter()
            .zip(&y_lag_resid)
            .map(|(&d, &l)| d * l)
            .sum::<T>()
            / y_lag_resid_sum_sq;

        // Compute residuals and standard error
        let residuals: Vec<T> = (0..n_obs)
            .map(|i| y_diff_resid[i] - gamma * y_lag_resid[i])
            .collect();

        let residual_sum_sq = residuals.iter().map(|&v| v * v).sum::<T>();
        let df = T::from_usize(n_obs - lags - 2).unwrap().max(T::one());
        let sigma_sq = residual_sum_sq / df;

        let se_gamma = T::sqrt(sigma_sq / y_lag_resid_sum_sq);

        if se_gamma <= T::epsilon() {
            return T::zero();
        }

        // Return t-statistic (ADF statistic)
        gamma / se_gamma
    }
}

lazy_info!(
    ADF_INFO,
    Adf,
    size: 1,
    min_ts_length: 4,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: true,
);

impl FeatureNamesDescriptionsTrait for Adf {
    fn get_names(&self) -> Vec<&str> {
        vec!["adf"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["Augmented Dickey-Fuller test statistic"]
    }
}

impl<T> FeatureEvaluator<T> for Adf
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;

        let y = ts.m.as_slice();
        let n = y.len();
        let lags = Self::select_lags(n);

        let adf_stat = Self::compute_adf_statistic(y, lags);

        Ok(vec![adf_stat])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(Adf);

    feature_test!(
        adf_trending,
        [Adf::new()],
        // Trending series should give a positive or small negative value
        [6.8123717],
        linspace(0.0_f32, 19.0, 20),
        [
            0.0_f32, 0.99833417, 1.9092974, 2.858082, 3.7568025, 4.720585, 5.6353555, 6.620087,
            7.5566044, 8.574961, 9.544021, 10.599999, 11.615806, 12.716676, 13.77692, 14.936632,
            16.041698, 17.245535, 18.3913, 19.637423,
        ],
    );

    #[test]
    fn test_lag_selection() {
        assert_eq!(Adf::select_lags(3), 0);
        assert_eq!(Adf::select_lags(4), 1);
        assert_eq!(Adf::select_lags(10), 2);
        assert_eq!(Adf::select_lags(30), 3);
        assert_eq!(Adf::select_lags(100), 4);
        assert_eq!(Adf::select_lags(1000), 9);
        assert_eq!(Adf::select_lags(2000), 10); // capped at 10
    }

    #[test]
    fn test_adf_returns_finite() {
        let mut rng = StdRng::seed_from_u64(42);
        let t: Vec<f64> = (0..50).map(|x| x as f64).collect();
        let m: Vec<f64> = (0..50).map(|_| rng.sample(StandardNormal)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let result = Adf::new().eval(&mut ts).unwrap();
        assert_eq!(result.len(), 1);
        assert!(result[0].is_finite());
    }
}
