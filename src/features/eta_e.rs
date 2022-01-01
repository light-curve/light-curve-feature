use crate::evaluator::*;
use itertools::Itertools;

macro_const! {
    const DOC: &'static str = r#"
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
"#;
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
        let x = [
            58197.50390625,
            58218.48828125,
            58218.5078125,
            58222.46875,
            58222.4921875,
            58230.48046875,
            58244.48046875,
            58246.43359375,
            58247.4609375,
            58247.48046875,
            58247.48046875,
            58249.44140625,
            58249.4765625,
            58255.4609375,
            58256.41796875,
            58256.45703125,
            58257.44140625,
            58257.4609375,
            58258.44140625,
            58259.4453125,
            58262.3828125,
            58263.421875,
            58266.359375,
            58268.42578125,
            58269.41796875,
            58270.40625,
            58271.4609375,
            58273.421875,
            58274.33984375,
            58275.40234375,
            58276.42578125,
            58277.3984375,
            58279.40625,
            58280.36328125,
            58282.44921875,
            58283.3828125,
            58285.3828125,
            58286.34375,
            58288.44921875,
            58289.4453125,
            58290.3828125,
            58291.3203125,
            58292.3359375,
            58293.30078125,
            58296.32421875,
            58297.33984375,
            58298.33984375,
            58301.36328125,
            58302.359375,
            58303.33984375,
            58304.36328125,
            58305.36328125,
            58307.3828125,
            58308.37890625,
            58310.38671875,
            58311.3828125,
            58313.38671875,
            58314.421875,
            58315.33984375,
            58316.34375,
            58317.40625,
            58318.33984375,
            58320.3515625,
            58321.3515625,
            58322.359375,
            58323.27734375,
            58324.23828125,
            58326.3203125,
            58327.31640625,
            58329.3203125,
            58330.37890625,
            58332.296875,
            58333.32421875,
            58334.34765625,
            58336.23046875,
            58338.2109375,
            58340.3046875,
            58341.328125,
            58342.328125,
            58343.32421875,
            58344.31640625,
            58345.32421875,
            58348.21875,
            58351.234375,
            58354.2578125,
            58355.2734375,
            58356.1953125,
            58358.25390625,
            58360.21875,
            58361.234375,
            58366.18359375,
            58370.15234375,
            58373.171875,
            58374.171875,
            58376.171875,
            58425.0859375,
            58427.0859375,
            58428.1015625,
            58430.0859375,
            58431.12890625,
            58432.0859375,
            58433.08984375,
            58436.0859375,
        ];
        let y = [
            17.357999801635742,
            17.329999923706055,
            17.332000732421875,
            17.312999725341797,
            17.30500030517578,
            17.31599998474121,
            17.27899932861328,
            17.305999755859375,
            17.333999633789062,
            17.332000732421875,
            17.332000732421875,
            17.323999404907227,
            17.256000518798828,
            17.308000564575195,
            17.290000915527344,
            17.298999786376953,
            17.270000457763672,
            17.270000457763672,
            17.297000885009766,
            17.288000106811523,
            17.358999252319336,
            17.273000717163086,
            17.354999542236328,
            17.301000595092773,
            17.2810001373291,
            17.299999237060547,
            17.341999053955078,
            17.30500030517578,
            17.29599952697754,
            17.336000442504883,
            17.31399917602539,
            17.336999893188477,
            17.304000854492188,
            17.309999465942383,
            17.304000854492188,
            17.29199981689453,
            17.31100082397461,
            17.28499984741211,
            17.327999114990234,
            17.347999572753906,
            17.32200050354004,
            17.319000244140625,
            17.2810001373291,
            17.327999114990234,
            17.291000366210938,
            17.3439998626709,
            17.336000442504883,
            17.27899932861328,
            17.38800048828125,
            17.27899932861328,
            17.297000885009766,
            17.29599952697754,
            17.312000274658203,
            17.253999710083008,
            17.312000274658203,
            17.284000396728516,
            17.319000244140625,
            17.32200050354004,
            17.290000915527344,
            17.31599998474121,
            17.28499984741211,
            17.30299949645996,
            17.284000396728516,
            17.336000442504883,
            17.31399917602539,
            17.356000900268555,
            17.308000564575195,
            17.31999969482422,
            17.301000595092773,
            17.325000762939453,
            17.30900001525879,
            17.29800033569336,
            17.29199981689453,
            17.339000701904297,
            17.32699966430664,
            17.31800079345703,
            17.320999145507812,
            17.315000534057617,
            17.304000854492188,
            17.327999114990234,
            17.308000564575195,
            17.34000015258789,
            17.325000762939453,
            17.322999954223633,
            17.30900001525879,
            17.308000564575195,
            17.275999069213867,
            17.33799934387207,
            17.343000411987305,
            17.437999725341797,
            17.280000686645508,
            17.305999755859375,
            17.320999145507812,
            17.325000762939453,
            17.32699966430664,
            17.339000701904297,
            17.298999786376953,
            17.29199981689453,
            17.336000442504883,
            17.32699966430664,
            17.28499984741211,
            17.284000396728516,
            17.257999420166016,
        ];
        let mut ts = TimeSeries::new_without_weight(&x[..], &y[..]);
        let actual: f32 = eval.eval(&mut ts).unwrap()[0];
        assert!(actual.is_finite());
    }
}