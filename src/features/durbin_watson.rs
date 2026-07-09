use crate::evaluator::*;
use crate::straight_line_fit::fit_straight_line;
use itertools::Itertools;

macro_const! {
    const DOC: &str = r"
Durbin-Watson statistic for serial autocorrelation of residuals

$$
d \equiv \frac{\sum_{i=1}^{N-1}(e_i - e_{i-1})^2}{\sum_{i=0}^{N-1}e_i^2},
$$
where $e_i = m_i - (\hat{a} + \hat{b}\,t_i)$ are residuals from the ordinary least-squares
linear fit $m = a + b\,t$, with $\hat{a}$ and $\hat{b}$ being the fitted intercept and slope.

The statistic lies in $[0, 4]$:
- $d \approx 0$: strong positive serial autocorrelation
- $d \approx 2$: no serial autocorrelation
- $d \approx 4$: strong negative serial autocorrelation

- Depends on: **time**, **magnitude**
- Minimum number of observations: **3**
- Number of features: **1**

Durbin, Watson 1950, [DOI:10.1093/biomet/37.3-4.409](https://doi.org/10.1093/biomet/37.3-4.409)
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub struct DurbinWatson {}

impl DurbinWatson {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    DURBIN_WATSON_INFO,
    DurbinWatson,
    size: 1,
    min_ts_length: 3,
    t_required: true,
    m_required: true,
    w_required: false,
    sorting_required: true,
    variability_required: false,
);

impl FeatureNamesDescriptionsTrait for DurbinWatson {
    fn get_names(&self) -> Vec<&str> {
        vec!["dw"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["Durbin-Watson statistic for serial autocorrelation of residuals from linear fit"]
    }
}

impl<T> FeatureEvaluator<T> for DurbinWatson
where
    T: Float,
{
    fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        let fit = fit_straight_line(ts, false);
        let residuals: Vec<T> =
            ts.t.as_slice()
                .iter()
                .zip(ts.m.as_slice().iter())
                .map(|(&t, &m)| m - fit.intercept - fit.slope * t)
                .collect();
        let denominator: T = residuals.iter().map(|&e| e * e).sum();
        if denominator.is_zero() {
            return Err(EvaluatorError::ZeroDivision(
                "Durbin-Watson: all residuals from linear fit are zero",
            ));
        }
        let numerator: T = residuals
            .iter()
            .tuple_windows()
            .map(|(&a, &b)| (b - a).powi(2))
            .sum();
        Ok(vec![numerator / denominator])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(DurbinWatson);

    // Expected value computed with:
    //   import numpy as np
    //   t = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    //   m = np.array([2.0, 3.0, 1.0, 4.0, 2.0])
    //   X = np.column_stack([np.ones(len(t)), t])
    //   b = np.linalg.lstsq(X, m, rcond=None)[0]
    //   e = m - X @ b
    //   dw = np.sum(np.diff(e)**2) / np.sum(e**2)
    //   # dw = 3.5372549019607843
    feature_test!(
        durbin_watson,
        [DurbinWatson::new()],
        [3.5372549_f32],
        [1.0_f32, 2.0, 3.0, 4.0, 5.0],
        [2.0_f32, 3.0, 1.0, 4.0, 2.0],
    );
}
