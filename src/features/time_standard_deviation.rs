use crate::evaluator::*;

macro_const! {
    const DOC: &'static str = r"
Standard deviation of time moments

$$
\sigma_t \equiv \frac{\sum_i {(t_i - \langle t \rangle)^2}}{N - 1}.
$$

Note: highly cadence-dependent feature.

- Depends on: **time**
- Minimum number of observations: **2**
- Number of features: **1**
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Deserialize, Serialize, JsonSchema)]
pub struct TimeStandardDeviation {}

impl TimeStandardDeviation {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    TIME_STANDARD_DEVIATION_INFO,
    TimeStandardDeviation,
    size: 1,
    min_ts_length: 2,
    t_required: true,
    m_required: false,
    w_required: false,
    sorting_required: false,
);

impl FeatureNamesDescriptionsTrait for TimeStandardDeviation {
    fn get_names(&self) -> Vec<&str> {
        vec!["time_standard_deviation"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["standard deviation of time moments"]
    }
}

impl<T> FeatureEvaluator<T> for TimeStandardDeviation
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.t.get_std()])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(TimeStandardDeviation);

    feature_test!(
        time_standard_deviation,
        [TimeStandardDeviation::new()],
        [1.5811388300841898],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
    );
}
