use crate::evaluator::*;

macro_const! {
    const DOC: &str = r"
Mean magnitude

$$
\langle m \rangle \equiv \frac1{N} \sum_i m_i.
$$
This is non-weighted mean, see [WeightedMean](crate::WeightedMean) for weighted mean.

- Depends on: **magnitude**
- Minimum number of observations: **1**
- Number of features: **1**
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Mean {}

lazy_info!(
    MEAN_INFO,
    Mean,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl Mean {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

impl FeatureNamesDescriptionsTrait for Mean {
    fn get_names(&self) -> Vec<&str> {
        vec!["mean"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["mean magnitude"]
    }
}

impl<T> FeatureEvaluator<T> for Mean
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        Ok(vec![ts.m.get_mean()])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(Mean);

    feature_test!(
        mean,
        [Mean::new()],
        [14.0],
        [1.0_f32, 1.0, 1.0, 1.0, 5.0, 6.0, 6.0, 6.0, 99.0],
    );
}
