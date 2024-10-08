use crate::evaluator::*;
use crate::sorted_array::SortedArray;

macro_const! {
    const DOC: &'static str = r"
Median of the absolute value of the difference between magnitude and its median

$$
\mathrm{median~absolute~deviation} \equiv \mathrm{Median}\left(|m_i - \mathrm{Median}(m)|\right).
$$

- Depends on: **magnitude**
- Minimum number of observations: **1**
- Number of features: **1**

D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MedianAbsoluteDeviation {}

lazy_info!(
    MEDIAN_ABSOLUTE_DEVIATION_INFO,
    MedianAbsoluteDeviation,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl MedianAbsoluteDeviation {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

impl FeatureNamesDescriptionsTrait for MedianAbsoluteDeviation {
    fn get_names(&self) -> Vec<&str> {
        vec!["median_absolute_deviation"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["median of absolute magnitude deviation from its median"]
    }
}

impl<T> FeatureEvaluator<T> for MedianAbsoluteDeviation
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_median = ts.m.get_median();
        let sorted_deviation: SortedArray<_> =
            ts.m.sample
                .mapv(|m| T::abs(m - m_median))
                .as_slice_memory_order()
                .expect("TimeSeries::m::sample::mapv(...) is supposed to be contiguous")
                .into();
        Ok(vec![sorted_deviation.median()])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(MedianAbsoluteDeviation);

    feature_test!(
        median_absolute_deviation,
        [MedianAbsoluteDeviation::new()],
        [4.0],
        [1.0_f32, 1.0, 1.0, 1.0, 5.0, 6.0, 6.0, 6.0, 100.0],
    );
}
