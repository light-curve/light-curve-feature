use crate::evaluator::*;

macro_const! {
    const DOC: &str = r"
Weighted mean magnitude

$$
\bar{m} \equiv \frac{\sum_i m_i / \delta_i^2}{\sum_i 1 / \delta_i^2}.
$$
See [Mean](crate::Mean) for non-weighted mean.

- Depends on: **magnitude**, **magnitude error**
- Minimum number of observations: **1**
- Number of features: **1**
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]
pub struct WeightedMean {}

lazy_info!(
    WEIGHTED_MEAN_INFO,
    WeightedMean,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: true,
    sorting_required: false,
);

impl WeightedMean {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

impl FeatureNamesDescriptionsTrait for WeightedMean {
    fn get_names(&self) -> Vec<&str> {
        vec!["weighted_mean"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["magnitude averaged weighted by inverse square error"]
    }
}

impl<T> FeatureEvaluator<T> for WeightedMean
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.get_m_weighted_mean()])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(WeightedMean);

    feature_test!(
        weighted_mean,
        [WeightedMean::new()],
        [1.1777777777777778],
        [1.0; 5], // isn't used
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
        [10.0, 5.0, 3.0, 2.5, 2.0],
    );
}
