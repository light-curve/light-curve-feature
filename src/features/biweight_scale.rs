use crate::data::SortedArray;
use crate::evaluator::*;

macro_const! {
    const DOC: &str = r"
Biweight scale — a robust measure of the magnitude scale

Tukey's biweight (bisquare) scale estimator, a robust alternative to the standard deviation
that down-weights outliers smoothly. With the median $M$ and the median absolute deviation
$\mathrm{MAD} = \mathrm{Median}(|m_i - M|)$, define
$$
u_i = \frac{m_i - M}{c \cdot \mathrm{MAD}},
$$
with tuning constant $c = 9$. Only points with $|u_i| < 1$ contribute, and the scale is
$$
\zeta_\mathrm{BI} = \sqrt{N} \,
\frac{\sqrt{\sum_{|u_i| < 1} (m_i - M)^2 \, (1 - u_i^2)^4}}
{\left| \sum_{|u_i| < 1} (1 - u_i^2)(1 - 5 u_i^2) \right|}.
$$

When $\mathrm{MAD} = 0$ (at least half of the values equal the median) the estimator is
undefined and zero is returned.

- Depends on: **magnitude**
- Minimum number of observations: **1**
- Number of features: **1**

Beers, Flynn & Gebhardt 1990 [DOI:10.1086/115487](https://doi.org/10.1086/115487)
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub struct BiweightScale {}

lazy_info!(
    BIWEIGHT_SCALE_INFO,
    BiweightScale,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
    variability_required: false,
);

impl BiweightScale {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

impl FeatureNamesDescriptionsTrait for BiweightScale {
    fn get_names(&self) -> Vec<&str> {
        vec!["biweight_scale"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["robust biweight scale estimator of magnitude"]
    }
}

impl<T> FeatureEvaluator<T> for BiweightScale
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

        let c = T::three() * T::three(); // tuning constant, 9
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

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(BiweightScale);

    feature_test!(
        biweight_scale,
        [BiweightScale::new()],
        // astropy.stats.biweight_scale (c=9, modify_sample_size=False)
        [7.922078257662727],
        [1.0_f64, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 100.0],
    );

    #[test]
    fn constant_is_zero() {
        // MAD == 0: a flat light curve has zero robust scale, no NaN/panic.
        let m = [5.0_f64; 6];
        let mut ts = TimeSeries::new_without_weight(&m, &m);
        let actual = BiweightScale::new().eval(&mut ts).unwrap();
        assert_eq!(actual, vec![0.0]);
    }
}
