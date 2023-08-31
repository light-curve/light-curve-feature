use crate::evaluator::*;
use itertools::Itertools;

macro_const! {
    const DOC: &'static str = r"
$\eta^e$ — modification of [Eta](crate::Eta) for unevenly time series

$$
\eta^e \equiv \frac{(t_{N-1} - t_0)^2}{(N - 1)^3} \frac{\sum_{i=0}^{N-2} \left(\frac{m_{i+1} - m_i}{t_{i+1} - t_i}\right)^2}{\sigma_m^2}
$$
where $N$ is the number of observations,
$\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.
Note that this definition is a bit different from both Kim et al. 2014 and
[feets](https://feets.readthedocs.io/en/latest/)

Note that this feature can have very high values and be highly cadence-dependent in the case of large range of time
lags. In this case consider to use this feature with [Bins](crate::Bins).

- Depends on: **time**, **magnitude**
- Minimum number of observations: **2**
- Number of features: **1**

Kim et al. 2014, [DOI:10.1051/0004-6361/201323252](https://doi.org/10.1051/0004-6361/201323252)
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]
pub struct EtaE {}

impl EtaE {
    pub fn new() -> Self {
        Self {}
    }

    pub fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    ETA_E_INFO,
    EtaE,
    size: 1,
    min_ts_length: 2,
    t_required: true,
    m_required: true,
    w_required: false,
    sorting_required: true,
);

impl FeatureNamesDescriptionsTrait for EtaE {
    fn get_names(&self) -> Vec<&str> {
        vec!["eta_e"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["generalised Von Neummann eta for irregular time-series"]
    }
}

impl<T> FeatureEvaluator<T> for EtaE
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_std2 = get_nonzero_m_std2(ts)?;
        let sq_slope_sum =
            ts.t.as_slice()
                .iter()
                .zip(ts.m.as_slice().iter())
                .tuple_windows()
                .map(|((&t1, &m1), (&t2, &m2))| ((m2 - m1) / (t2 - t1)).powi(2))
                .filter(|&x| x.is_finite())
                .sum::<T>();
        let value = (ts.t.sample[ts.lenu() - 1] - ts.t.sample[0]).powi(2) * sq_slope_sum
            / m_std2
            / (ts.lenf() - T::one()).powi(3);
        Ok(vec![value])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::extractor::FeatureExtractor;
    use crate::features::Eta;
    use crate::tests::*;

    check_feature!(EtaE);

    feature_test!(
        eta_e,
        [EtaE::new()],
        [0.6957894],
        [1.0_f32, 2.0, 5.0, 10.0],
        [1.0_f32, 1.0, 6.0, 8.0],
    );

    #[test]
    fn eta_is_eta_e_for_even_grid() {
        let fe = FeatureExtractor::<_, Feature<_>>::new(vec![
            Eta::default().into(),
            EtaE::default().into(),
        ]);
        let x = linspace(0.0_f64, 1.0, 11);
        let y: Vec<_> = x.iter().map(|&t| 3.0 + t.powi(2)).collect();
        let mut ts = TimeSeries::new_without_weight(&x, &y);
        let values = fe.eval(&mut ts).unwrap();
        all_close(&values[0..1], &values[1..2], 1e-10);
    }

    /// See [Issue #2](https://github.com/hombit/light-curve/issues/2)
    #[test]
    fn eta_e_finite() {
        let eval = EtaE::default();
        let (t, m, _) =
            light_curve_feature_test_util::issue_light_curve_mag("light-curve-2/1.csv", None);
        let mut ts = TimeSeries::new_without_weight(t, m);
        let actual: f32 = eval.eval(&mut ts).unwrap()[0];
        assert!(actual.is_finite());
    }
}
