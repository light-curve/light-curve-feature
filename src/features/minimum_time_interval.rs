use crate::evaluator::*;
use itertools::Itertools;

macro_const! {
    const DOC: &str = r"
Minimum time interval between consequent observations

$$
\min{(t_{i+1} - t_i)}
$$

Note: highly cadence-dependent feature.

- Depends on: **time**
- Minimum number of observations: **2**
- Number of features: **1**
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Deserialize, Serialize, JsonSchema)]
pub struct MinimumTimeInterval {}

impl MinimumTimeInterval {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    MINIMUM_TIME_INTERVAL_INFO,
    MinimumTimeInterval,
    size: 1,
    min_ts_length: 2,
    t_required: true,
    m_required: false,
    w_required: false,
    sorting_required: true,
);

impl FeatureNamesDescriptionsTrait for MinimumTimeInterval {
    fn get_names(&self) -> Vec<&str> {
        vec!["minimum_time_interval"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["minimum time interval between consequent observations"]
    }
}

impl<T> FeatureEvaluator<T> for MinimumTimeInterval
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let dt =
            ts.t.as_slice()
                .iter()
                .tuple_windows()
                .map(|(&a, &b)| b - a)
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
        Ok(vec![dt])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(MinimumTimeInterval);

    feature_test!(
        minimum_time_interval,
        [MinimumTimeInterval::new()],
        [0.1],
        [0.0_f32, 0.5, 0.6, 1.6, 10.6],
    );
}
