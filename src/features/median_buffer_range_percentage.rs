use crate::evaluator::*;

use conv::prelude::*;
use ordered_float::NotNan;

macro_const! {
    const DOC: &str = r"
Fraction of observations inside $\mathrm{Median}(m) \pm q \times (\max(m) - \min(m)) / 2$ interval

- Depends on: **magnitude**
- Minimum number of observations: **1**
- Number of features: **1**

D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
";
}

#[doc = DOC!()]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(
    into = "MedianBufferRangePercentageParameters",
    from = "MedianBufferRangePercentageParameters",
    bound(deserialize = "T: Float")
)]
pub struct MedianBufferRangePercentage<T>
where
    T: Float,
{
    quantile: NotNan<f32>,
    name: String,
    description: String,
    #[serde(skip)]
    _phantom: std::marker::PhantomData<T>,
}

lazy_info!(
    MEDIAN_BUFFER_RANGE_PERCENTAGE_INFO,
    MedianBufferRangePercentage<T>,
    T,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl<T> MedianBufferRangePercentage<T>
where
    T: Float,
{
    pub fn new(quantile: f32) -> Self {
        assert!(quantile > 0.0, "Quantile should be positive");
        assert!(quantile.is_finite(), "quantile must be finite");
        let quantile = NotNan::new(quantile).expect("quantile must not be NaN");
        Self {
            quantile,
            name: format!(
                "median_buffer_range_percentage_{:.0}",
                100.0 * quantile.into_inner()
            ),
            description: format!(
                "fraction of observations which magnitudes differ from median by no more than \
                {:.3e} of amplitude",
                quantile.into_inner()
            ),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    #[inline]
    pub fn default_quantile() -> f32 {
        0.1
    }
}

impl<T> MedianBufferRangePercentage<T>
where
    T: Float,
{
    pub const fn doc() -> &'static str {
        DOC
    }
}

impl<T> Default for MedianBufferRangePercentage<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(Self::default_quantile())
    }
}

impl<T> FeatureNamesDescriptionsTrait for MedianBufferRangePercentage<T>
where
    T: Float,
{
    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![self.description.as_str()]
    }
}

impl<T> FeatureEvaluator<T> for MedianBufferRangePercentage<T>
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_median = ts.m.get_median();
        let amplitude = T::half() * (ts.m.get_max() - ts.m.get_min());
        // This conversion should never fail because f32 is always convertible to f32 or f64
        let quantile = self.quantile.into_inner().value_as::<T>().unwrap();
        let threshold = quantile * amplitude;
        let count_under = ts.m.sample.fold(0, |count, &m| {
            let under = T::abs(m - m_median) < threshold;
            count + usize::from(under)
        });
        Ok(vec![count_under.approx_as::<T>().unwrap() / ts.lenf()])
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "MedianBufferRangePercentage")]
struct MedianBufferRangePercentageParameters {
    quantile: f32,
}

impl<T> From<MedianBufferRangePercentage<T>> for MedianBufferRangePercentageParameters
where
    T: Float,
{
    fn from(f: MedianBufferRangePercentage<T>) -> Self {
        Self {
            quantile: f.quantile.into_inner(),
        }
    }
}

impl<T> From<MedianBufferRangePercentageParameters> for MedianBufferRangePercentage<T>
where
    T: Float,
{
    fn from(p: MedianBufferRangePercentageParameters) -> Self {
        Self::new(p.quantile)
    }
}

impl<T> JsonSchema for MedianBufferRangePercentage<T>
where
    T: Float,
{
    json_schema!(MedianBufferRangePercentageParameters, false);
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    use serde_test::{Token, assert_tokens};

    check_feature!(MedianBufferRangePercentage<f64>);

    feature_test!(
        median_buffer_range_percentage,
        [
            MedianBufferRangePercentage::default(),
            MedianBufferRangePercentage::new(0.1), // should be the same
            MedianBufferRangePercentage::new(0.2),
        ],
        [0.5555555555555556, 0.5555555555555556, 0.7777777777777778],
        [1.0_f32, 41.0, 49.0, 49.0, 50.0, 51.0, 52.0, 58.0, 100.0],
    );

    feature_test!(
        median_buffer_range_percentage_plateau,
        [MedianBufferRangePercentage::default()],
        [0.0],
        [0.0; 10],
    );

    #[test]
    fn serialization() {
        const QUANTILE: f32 = 0.432;
        let median_buffer_range_percentage = MedianBufferRangePercentage::<f64>::new(QUANTILE);
        assert_tokens(
            &median_buffer_range_percentage,
            &[
                Token::Struct {
                    len: 1,
                    name: "MedianBufferRangePercentage",
                },
                Token::String("quantile"),
                Token::F32(QUANTILE),
                Token::StructEnd,
            ],
        )
    }

    #[test]
    fn no_positive_overflow() {
        // Minimal size to trigger the overflow
        const N: usize = (1 << 24) + 1;
        let t = Array1::linspace(0.0_f32, 1.0, N);
        let mut ts = TimeSeries::new_without_weight(t.view(), t.view());
        // Absurdly large quantile just to make feature values equals to 1.0
        let feature = MedianBufferRangePercentage::new(2.0);
        // Should not panic
        let values = feature.eval(&mut ts).unwrap();
        assert_eq!(values[0], 1.0);
    }
}
