use crate::data::SortedArray;
use crate::evaluator::*;

use conv::ConvUtil;
use ordered_float::NotNan;

macro_const! {
    const DOC: &str = r"
Biweight scale — a robust measure of the magnitude scale

Tukey's biweight (bisquare) scale estimator, a robust alternative to the standard deviation
that down-weights outliers smoothly. With the median $M$ and the median absolute deviation
$\mathrm{MAD} = \mathrm{Median}(|m_i - M|)$, define
$$
u_i = \frac{m_i - M}{c \cdot \mathrm{MAD}},
$$
with tuning constant $c$ (default $c = 9$). Only points with $|u_i| < 1$ contribute, and the
scale is
$$
\zeta_\mathrm{BI} = \sqrt{N} \,
\frac{\sqrt{\sum_{|u_i| < 1} (m_i - M)^2 \, (1 - u_i^2)^4}}
{\left| \sum_{|u_i| < 1} (1 - u_i^2)(1 - 5 u_i^2) \right|}.
$$
Smaller $c$ rejects outliers more aggressively; larger $c$ approaches the standard deviation.

When $\mathrm{MAD} = 0$ (at least half of the values equal the median) the estimator is
undefined and zero is returned.

- Depends on: **magnitude**
- Minimum number of observations: **1**
- Number of features: **1**

Beers, Flynn & Gebhardt 1990 [DOI:10.1086/115487](https://doi.org/10.1086/115487)
";
}

#[doc = DOC!()]
/// ### Example
/// ```
/// use light_curve_feature::*;
///
/// let fe = FeatureExtractor::new(vec![BiweightScale::default(), BiweightScale::new(6.0)]);
/// let time = [0.0; 5]; // Doesn't depend on time
/// let magn = [1.0, 2.0, 3.0, 4.0, 100.0];
/// let mut ts = TimeSeries::new_without_weight(&time, &magn);
/// let values = fe.eval(&mut ts).unwrap();
/// assert_eq!(values.len(), 2);
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(
    from = "BiweightScaleParameters",
    into = "BiweightScaleParameters",
    bound(deserialize = "T: Float")
)]
pub struct BiweightScale<T>
where
    T: Float,
{
    c: NotNan<f32>,
    name: String,
    description: String,
    #[serde(skip)]
    _phantom: std::marker::PhantomData<T>,
}

impl<T> BiweightScale<T>
where
    T: Float,
{
    pub fn new(c: f32) -> Self {
        assert!(c > 0.0, "c should be positive");
        assert!(c.is_finite(), "c must be finite");
        let c = NotNan::new(c).expect("c must not be NaN");
        Self {
            c,
            name: format!("biweight_scale_{:.0}", c.into_inner()),
            description: format!(
                "robust biweight scale estimator of magnitude with tuning constant {:.3e}",
                c.into_inner()
            ),
            _phantom: std::marker::PhantomData,
        }
    }

    #[inline]
    pub fn default_c() -> f32 {
        9.0
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    BIWEIGHT_SCALE_INFO,
    BiweightScale<T>,
    T,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
    variability_required: false,
);

impl<T> Default for BiweightScale<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(Self::default_c())
    }
}

impl<T> FeatureNamesDescriptionsTrait for BiweightScale<T>
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

impl<T> FeatureEvaluator<T> for BiweightScale<T>
where
    T: Float,
{
    fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        let m_median = ts.m.get_median();

        // Median absolute deviation from the median.
        let sorted_deviation: SortedArray<_> =
            ts.m.sample
                .mapv(|m| (m - m_median).abs())
                .as_slice_memory_order()
                .expect("TimeSeries::m::sample::mapv(...) is supposed to be contiguous")
                .into();
        let mad = sorted_deviation.median();

        // MAD == 0 means at least half of the points equal the median; the
        // biweight scale is undefined, return zero.
        if mad <= T::zero() {
            return Ok(vec![T::zero()]);
        }

        // f32 -> T conversion never fails (f32 fits both f32 and f64).
        let c = self.c.into_inner().value_as::<T>().unwrap();
        let denominator = c * mad;

        // numerator = sum d^2 (1 - u^2)^4, denominator = sum (1 - u^2)(1 - 5 u^2)
        // over points with u^2 < 1.
        let (num, den) =
            ts.m.sample
                .iter()
                .fold((T::zero(), T::zero()), |(num, den), &m| {
                    let d = m - m_median;
                    let u2 = (d / denominator).powi(2);
                    if u2 < T::one() {
                        let one_minus_u2 = T::one() - u2;
                        (
                            num + d * d * one_minus_u2.powi(4),
                            den + one_minus_u2 * (T::one() - T::five() * u2),
                        )
                    } else {
                        (num, den)
                    }
                });

        let midvariance = ts.lenf() * num / den.powi(2);
        Ok(vec![midvariance.sqrt()])
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "BiweightScale")]
struct BiweightScaleParameters {
    c: f32,
}

impl<T> From<BiweightScale<T>> for BiweightScaleParameters
where
    T: Float,
{
    fn from(f: BiweightScale<T>) -> Self {
        Self {
            c: f.c.into_inner(),
        }
    }
}

impl<T> From<BiweightScaleParameters> for BiweightScale<T>
where
    T: Float,
{
    fn from(p: BiweightScaleParameters) -> Self {
        Self::new(p.c)
    }
}

impl<T> JsonSchema for BiweightScale<T>
where
    T: Float,
{
    json_schema!(BiweightScaleParameters, false);
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    use serde_test::{Token, assert_tokens};

    check_feature!(BiweightScale<f64>);

    feature_test!(
        biweight_scale,
        [BiweightScale::default(), BiweightScale::new(9.0)], // identical
        // astropy.stats.biweight_scale(c=9, modify_sample_size=False)
        [7.922078257662727, 7.922078257662727],
        [1.0_f64, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 100.0],
    );

    #[test]
    fn constant_is_zero() {
        // MAD == 0: a flat light curve has zero robust scale, no NaN/panic.
        let m = [5.0_f64; 6];
        let mut ts = TimeSeries::new_without_weight(&m, &m);
        let actual = BiweightScale::<f64>::default().eval(&mut ts).unwrap();
        assert_eq!(actual, vec![0.0]);
    }

    #[test]
    fn serialization() {
        const C: f32 = 7.5;
        let biweight_scale = BiweightScale::<f64>::new(C);
        assert_tokens(
            &biweight_scale,
            &[
                Token::Struct {
                    len: 1,
                    name: "BiweightScale",
                },
                Token::String("c"),
                Token::F32(C),
                Token::StructEnd,
            ],
        )
    }
}
