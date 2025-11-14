use crate::evaluator::*;
use crate::sorted_array::SortedArray;
use itertools::Itertools;

macro_const! {
    const DOC: &'static str = r"
S_B variability detection statistic

$$
S_B \equiv \sqrt{\frac{1}{2} \mathrm{median}((m_{i+1} - m_i)^2)},
$$

where $m_i$ are magnitudes ordered in time.

This statistic uses the median of squared successive differences and is useful for
variability detection as it combines sensitivity to scatter and correlation between
successive measurements while being robust to outliers.

- Depends on: **time**, **magnitude**
- Minimum number of observations: **2**
- Number of features: **1**

References:
- [Figuera Jaimes et al. 2013, A&A 556, A20](https://ui.adsabs.harvard.edu/abs/2013A%26A...556A..20F)
- [Sokolovsky et al. 2016, MNRAS 464, 274, Eq. 21](https://arxiv.org/abs/1609.01716)
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]
pub struct SBVariability {}

lazy_info!(
    SB_VARIABILITY_INFO,
    SBVariability,
    size: 1,
    min_ts_length: 2,
    t_required: true,
    m_required: true,
    w_required: false,
    sorting_required: true,
);

impl SBVariability {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

impl FeatureNamesDescriptionsTrait for SBVariability {
    fn get_names(&self) -> Vec<&str> {
        vec!["sb_variability"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["S_B variability detection statistic (root median squared successive difference)"]
    }
}

impl<T> FeatureEvaluator<T> for SBVariability
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;

        // Calculate squared successive differences
        let sq_diffs: Vec<T> =
            ts.m.as_slice()
                .iter()
                .tuple_windows()
                .map(|(&a, &b)| {
                    let diff = b - a;
                    diff * diff
                })
                .collect();

        // S_B = sqrt(median(squared differences) / 2)
        let sorted_sq_diffs: SortedArray<T> = sq_diffs.into();
        let median_sq_diff = sorted_sq_diffs.median();
        let sb = (median_sq_diff / T::two()).sqrt();

        Ok(vec![sb])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(SBVariability);

    feature_test!(
        sb_variability_constant,
        [SBVariability::new()],
        [0.0],
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
        [5.0, 5.0, 5.0, 5.0, 5.0],
    );

    feature_test!(
        sb_variability_linear,
        [SBVariability::new()],
        [0.7071067811865476], // sqrt(1/2) for constant differences of 1
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
        [0.0, 1.0, 2.0, 3.0, 4.0],
    );

    feature_test!(
        sb_variability_alternating,
        [SBVariability::new()],
        [1.4142135623730951], // sqrt(4/2) = sqrt(2)
        [0.0_f32, 1.0, 2.0, 3.0, 4.0],
        [0.0, 2.0, 0.0, 2.0, 0.0],
    );

    #[test]
    fn sb_variability_specific_calculation() {
        // Test with known values
        let t = [0.0_f64, 1.0, 2.0, 3.0];
        let m = [1.0, 2.0, 4.0, 3.0];
        // Differences: (2-1)^2 = 1, (4-2)^2 = 4, (3-4)^2 = 1
        // Squared differences: [1, 4, 1]
        // Median = 1
        // S_B = sqrt(1 / 2) = sqrt(0.5) ≈ 0.7071067811865476
        let expected = 0.7071067811865476;

        let fe = FeatureExtractor::new(vec![SBVariability::new()]);
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let actual = fe.eval(&mut ts).unwrap();

        assert!((actual[0] - expected).abs() < 1e-10);
    }
}
