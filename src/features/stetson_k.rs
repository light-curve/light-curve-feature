use crate::evaluator::*;

use ndarray::Zip;

macro_const! {
    const DOC: &str = r"
Stetson $K$ coefficient described light curve shape

$$
\mathrm{Stetson}~K \equiv \frac{\sum_i\left|\frac{m_i - \bar{m}}{\delta_i}\right|}{\sqrt{N\\,\chi^2}},
$$
where N is the number of observations,
$\bar{m}$ is the weighted mean magnitude
and $\chi^2 = \sum_i\left(\frac{m_i - \langle m \rangle}{\delta\_i}\right)^2$.

- Depends on: **magnitude**, **magnitude error**
- Minimum number of observations: **2**
- Number of features: **1**

P. B. Stetson, 1996. [DOI:10.1086/133808](https://doi.org/10.1086/133808)
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]
pub struct StetsonK {}

lazy_info!(
    STETSON_K_INFO,
    StetsonK,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: true,
    sorting_required: false,
);

impl StetsonK {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

impl FeatureNamesDescriptionsTrait for StetsonK {
    fn get_names(&self) -> Vec<&str> {
        vec!["stetson_K"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["normalized weighted deviation of magnitude from its weighted mean"]
    }
}

impl<T> FeatureEvaluator<T> for StetsonK
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let chi2 = get_nonzero_reduced_chi2(ts)? * (ts.lenf() - T::one());
        let mean = ts.get_m_weighted_mean();
        let value = Zip::from(&ts.m.sample)
            .and(&ts.w.sample)
            .fold(T::zero(), |acc, &y, &w| acc + T::abs(y - mean) * T::sqrt(w))
            / T::sqrt(ts.lenf() * chi2);
        Ok(vec![value])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    use std::f64::consts::*;

    check_feature!(StetsonK);

    feature_test!(
        stetson_k_square_wave,
        [StetsonK::new()],
        [1.0],
        [1.0; 1000], // isn't used
        (0..1000)
            .map(|i| { if i < 500 { 1.0 } else { -1.0 } })
            .collect::<Vec<_>>(),
        [1.0; 1000],
    );

    // Slow convergence, use high tol
    feature_test!(
        stetson_k_sinus,
        [StetsonK::new()],
        [8_f64.sqrt() / PI],
        [1.0; 1000], // isn't used
        linspace(0.0, 2.0 * PI, 1000)
            .iter()
            .map(|&x| f64::sin(x))
            .collect::<Vec<_>>(),
        [1.0; 1000],
        1e-3,
    );

    feature_test!(
        stetson_k_sawtooth,
        [StetsonK::new()],
        [12_f64.sqrt() / 4.0],
        [1.0; 1000], // isn't used
        linspace(0.0, 1.0, 1000),
    );

    // It seems that Stetson (1996) formula for this case is wrong by the factor of 2 * sqrt((N-1) / N)
    feature_test!(
        stetson_k_single_peak,
        [StetsonK::new()],
        [2.0 * 99.0_f64.sqrt() / 100.0],
        [1.0; 100], // isn't used
        (0..100)
            .map(|i| { if i == 0 { 1.0 } else { -1.0 } })
            .collect::<Vec<_>>(),
    );

    #[test]
    fn stetson_k_plateau() {
        let eval = StetsonK::new();
        let x = [0.0; 10];
        let mut ts = TimeSeries::new_without_weight(&x, &x);
        assert_eq!(eval.eval(&mut ts), Err(EvaluatorError::FlatTimeSeries));
    }
}
