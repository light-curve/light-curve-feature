use crate::evaluator::*;

use conv::ConvUtil;
use ordered_float::NotNan;

macro_const! {
    const DOC: &str = r"
Fraction of observations beyond $n\,\sigma\_m$ from the mean magnitude $\langle m \rangle$

$$
\mathrm{beyond}~n\,\sigma\_m \equiv \frac{\sum\_i I\_{|m - \langle m \rangle| > n\,\sigma\_m}(m_i)}{N},
$$
where $I$ is the [indicator function](https://en.wikipedia.org/wiki/Indicator_function),
$N$ is the number of observations,
$\langle m \rangle$ is the mean magnitude
and $\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.

- Depends on: **magnitude**
- Minimum number of observations: **2**
- Number of features: **1**

D’Isanto et al. 2016 [DOI:10.1093/mnras/stw157](https://doi.org/10.1093/mnras/stw157)
";
}

#[doc = DOC!()]
/// ### Example
/// ```
/// use light_curve_feature::*;
/// use light_curve_common::all_close;
/// use std::f64::consts::SQRT_2;
///
/// let fe = FeatureExtractor::new(vec![BeyondNStd::default(), BeyondNStd::new(2.0)]);
/// let time = [0.0; 21];  // Doesn't depend on time
/// let mut magn = vec![0.0; 17];
/// magn.extend_from_slice(&[SQRT_2, -SQRT_2, 2.0 * SQRT_2, -2.0 * SQRT_2]);
/// let mut ts = TimeSeries::new_without_weight(&time[..], &magn[..]);
/// assert_eq!(0.0, ts.m.get_mean());
/// assert!((1.0 - ts.m.get_std()).abs() < 1e-15);
/// assert_eq!(vec![4.0 / 21.0, 2.0 / 21.0], fe.eval(&mut ts).unwrap());
/// ```
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(
    from = "BeyondNStdParameters",
    into = "BeyondNStdParameters",
    bound(deserialize = "T: Float")
)]
pub struct BeyondNStd<T>
where
    T: Float,
{
    nstd: NotNan<f32>,
    name: String,
    description: String,
    #[serde(skip)]
    _phantom: std::marker::PhantomData<T>,
}

impl<T> BeyondNStd<T>
where
    T: Float,
{
    pub fn new(nstd: f32) -> Self {
        assert!(nstd > 0.0, "nstd should be positive");
        assert!(nstd.is_finite(), "nstd must be finite");
        let nstd = NotNan::new(nstd).expect("nstd must not be NaN");
        Self {
            nstd,
            name: format!("beyond_{:.0}_std", nstd.into_inner()),
            description: format!(
                "fraction of observations which magnitudes are beyond {:.3e} standard deviations \
                from the mean magnitude",
                nstd.into_inner()
            ),
            _phantom: std::marker::PhantomData,
        }
    }

    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    #[inline]
    pub fn default_nstd() -> f32 {
        1.0
    }
}

impl<T> BeyondNStd<T>
where
    T: Float,
{
    pub const fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    BEYOND_N_STD_INFO,
    BeyondNStd<T>,
    T,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl<T> Default for BeyondNStd<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(Self::default_nstd())
    }
}

impl<T> FeatureNamesDescriptionsTrait for BeyondNStd<T>
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

impl<T> FeatureEvaluator<T> for BeyondNStd<T>
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_mean = ts.m.get_mean();
        // This conversion should never fail because f32 is always convertible to f32 or f64
        let nstd = T::from(self.nstd.into_inner()).unwrap();
        let threshold = ts.m.get_std() * nstd;
        let count_beyond = ts.m.sample.fold(0, |count, &m| {
            let beyond = T::abs(m - m_mean) > threshold;
            count + usize::from(beyond)
        });
        Ok(vec![count_beyond.approx_as::<T>().unwrap() / ts.lenf()])
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "BeyondNStd")]
struct BeyondNStdParameters {
    nstd: f32,
}

impl<T> From<BeyondNStd<T>> for BeyondNStdParameters
where
    T: Float,
{
    fn from(f: BeyondNStd<T>) -> Self {
        Self {
            nstd: f.nstd.into_inner(),
        }
    }
}

impl<T> From<BeyondNStdParameters> for BeyondNStd<T>
where
    T: Float,
{
    fn from(p: BeyondNStdParameters) -> Self {
        Self::new(p.nstd)
    }
}

impl<T> JsonSchema for BeyondNStd<T>
where
    T: Float,
{
    json_schema!(BeyondNStdParameters, false);
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    use serde_test::{Token, assert_tokens};

    check_feature!(BeyondNStd<f64>);

    feature_test!(
        beyond_n_std,
        [
            BeyondNStd::default(),
            BeyondNStd::new(1.0), // should be the same as the previous one
            BeyondNStd::new(2.0),
        ],
        [0.2, 0.2, 0.0],
        [1.0_f32, 2.0, 3.0, 4.0, 100.0],
    );

    #[test]
    fn serialization() {
        const NSTD: f32 = 2.34;
        let beyond_n_std = BeyondNStd::<f64>::new(NSTD);
        assert_tokens(
            &beyond_n_std,
            &[
                Token::Struct {
                    len: 1,
                    name: "BeyondNStd",
                },
                Token::String("nstd"),
                Token::F32(NSTD),
                Token::StructEnd,
            ],
        )
    }
}
