use crate::evaluator::*;
use crate::straight_line_fit::fit_straight_line;

macro_const! {
    const DOC: &str = r"
The slope, its error and noise level of the light curve in the linear fit

Least squares fit of the linear stochastic model with constant Gaussian noise $\Sigma$ assuming
observation errors to be zero:
$$
m_i = c + \mathrm{slope} t_i + \Sigma \varepsilon_i,
$$
where $c$ is a constant,
$\{\varepsilon_i\}$ are standard distributed random variables. $\mathrm{slope}$,
$\sigma_\mathrm{slope}$ and $\Sigma$ are returned.

- Depends on: **time**, **magnitude**
- Minimum number of observations: **3**
- Number of features: **3**
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub struct LinearTrend {}

impl LinearTrend {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    LINEAR_TREND_INFO,
    LinearTrend,
    size: 3,
    min_ts_length: 3,
    t_required: true,
    m_required: true,
    w_required: false,
    sorting_required: true,
);

impl FeatureNamesDescriptionsTrait for LinearTrend {
    fn get_names(&self) -> Vec<&str> {
        vec!["linear_trend", "linear_trend_sigma", "linear_trend_noise"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![
            "linear trend without respect to observation errors",
            "error of slope of linear fit without respect to observation errors",
            "standard deviation of noise for linear fit without respect to observation errors",
        ]
    }
}

impl<T> FeatureEvaluator<T> for LinearTrend
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let result = fit_straight_line(ts, false);
        Ok(vec![
            result.slope,
            T::sqrt(result.slope_sigma2),
            T::sqrt(result.reduced_chi2),
        ])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature_with_hash!(LinearTrend);

    feature_test!(
        linear_trend,
        [LinearTrend::new()],
        [1.38198758, 0.24532195657979344, 2.54157969],
        [1.0_f32, 3.0, 5.0, 7.0, 11.0, 13.0],
        [1.0_f32, 2.0, 3.0, 8.0, 10.0, 19.0],
    );

    /// See [Issue #3](https://github.com/hombit/light-curve/issues/3)
    fn linear_trend_finite(path: &str) {
        let eval = LinearTrend::default();
        let (t, m, _) =
            light_curve_feature_test_util::issue_light_curve_mag::<f32, _>(path).into_triple(None);
        let mut ts = TimeSeries::new_without_weight(t, m);
        let actual = eval.eval(&mut ts).unwrap();
        assert!(actual.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn linear_trend_finite_1() {
        linear_trend_finite("light-curve-3/1.csv");
    }

    #[test]
    fn linear_trend_finite_2() {
        linear_trend_finite("light-curve-3/2.csv");
    }

    #[test]
    fn linear_trend_finite_3() {
        linear_trend_finite("light-curve-3/640202200001881.csv");
    }

    #[test]
    fn linear_trend_finite_4() {
        linear_trend_finite("light-curve-3/742201400001054.csv");
    }

    #[test]
    fn linear_trend_finite_5() {
        linear_trend_finite("light-curve-3/742201400001066.csv");
    }
}
