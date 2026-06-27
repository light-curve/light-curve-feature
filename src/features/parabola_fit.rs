use crate::evaluator::*;
use crate::parabola_fit::fit_parabola;

macro_const! {
    const DOC: &str = r"
Curvature, extremum value, and reduced $\chi^2$ of the parabolic fit

Weighted least squares fit of the quadratic model with Gaussian noise described by observation
errors $\{\delta_i\}$:
$$
m_i = m_0 + g \, (t_i - t_0)^2 + \delta_i \varepsilon_i,
$$
where $g$ is the curvature, $t_0$ is the position of the extremum, $m_0$ is the value at the
extremum, and $\{\varepsilon_i\}$ are standard distributed random variables.

The model is fitted as $m_i = a \, t_i^2 + b \, t_i + c$, and the feature values are recovered as
$g = a$, $m_0 = c - \frac{b^2}{4a}$, and
$\frac{\sum{((m_i - a t_i^2 - b t_i - c) / \delta_i)^2}}{N - 3}$.
The extremum position $t_0 = -\frac{b}{2a}$ is not returned: it is an absolute time, which is a
poor classification feature and diverges as $g \to 0$.

Note that for a (nearly) straight light curve the curvature $g \to 0$ and the extremum
moves to infinity, so $m_0$ is not finite. Constant (plateau) light curves are rejected, as
the fit is degenerate.

- Depends on: **time**, **magnitude**, **magnitude error**
- Minimum number of observations: **4**
- Number of features: **3**
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub struct ParabolaFit {}

impl ParabolaFit {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    PARABOLA_FIT_INFO,
    ParabolaFit,
    size: 3,
    min_ts_length: 4,
    t_required: true,
    m_required: true,
    w_required: true,
    sorting_required: true,
    variability_required: true,
);

impl FeatureNamesDescriptionsTrait for ParabolaFit {
    fn get_names(&self) -> Vec<&str> {
        vec![
            "parabola_fit_g",
            "parabola_fit_m0",
            "parabola_fit_reduced_chi2",
        ]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![
            "curvature of parabolic fit",
            "extremum value of parabolic fit",
            "parabolic fit quality (reduced chi2)",
        ]
    }
}

impl<T> FeatureEvaluator<T> for ParabolaFit
where
    T: Float,
{
    fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        let result = fit_parabola(ts);
        Ok(vec![result.g, result.m0, result.reduced_chi2])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(ParabolaFit);

    feature_test!(
        parabola_fit,
        [ParabolaFit::default()],
        // m = 2 (t - 3)^2 + 5  =>  g = 2, m0 = 5, exact fit (reduced chi2 = 0)
        [2.0, 5.0, 0.0],
        [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0],
        [23.0_f64, 13.0, 7.0, 5.0, 7.0, 13.0],
        [1.0_f64, 1.0, 1.0, 1.0, 1.0, 1.0],
        1e-10,
    );

    #[test]
    fn flat_light_curve_is_rejected() {
        // A constant light curve is degenerate (g = 0, t0 / m0 = NaN); the
        // variability requirement must reject it instead of returning NaNs.
        let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let m = [7.0_f64; 5];
        let w = [1.0_f64; 5];
        let mut ts = TimeSeries::new(&t, &m, &w);
        assert!(matches!(
            ParabolaFit::default().eval(&mut ts),
            Err(EvaluatorError::FlatTimeSeries)
        ));
    }
}
