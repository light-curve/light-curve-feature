use crate::evaluator::*;
use crate::extractor::FeatureExtractor;
use crate::features::_periodogram_peaks::PeriodogramPeaks;
use crate::periodogram;
use crate::periodogram::{AverageNyquistFreq, NyquistFreq, PeriodogramPower, PeriodogramPowerFft};

use ndarray::Array1;
use std::convert::TryInto;
use std::fmt::Debug;

macro_const! {
    const DOC: &str = r#"
Peaks of Lomb–Scargle periodogram and periodogram as a meta-feature

Periodogram $P(\omega)$ is an estimate of spectral density of unevenly time series. `peaks` argument
corresponds to a number of the most significant spectral density peaks to return. For each peak its
period and "signal to noise" ratio is returned:

$$
\mathrm{signal~to~noise~of~peak} \equiv \frac{P(\omega_\mathrm{peak}) - \langle P(\omega) \rangle}{\sigma\_{P(\omega)}}.
$$

[Periodogram] can accept other features for feature extraction from periodogram as it was time
series without observation errors (unity weights are used if required). You can even pass one
[Periodogram] to another one if you are crazy enough

- Depends on: **time**, **magnitude**
- Minimum number of observations: as required by sub-features, but at least two
- Number of features: **$2 \times \mathrm{peaks}$** plus sub-features
"#;
}

#[doc = DOC!()]
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(
    bound = "T: Float, F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>, <F as TryInto<PeriodogramPeaks>>::Error: Debug,",
    from = "PeriodogramParameters<T, F>",
    into = "PeriodogramParameters<T, F>"
)]
pub struct Periodogram<T, F>
where
    T: Float,
{
    resolution: f32,
    max_freq_factor: f32,
    nyquist: NyquistFreq,
    pub(crate) feature_extractor: FeatureExtractor<T, F>,
    // In can be re-defined in MultiColorPeriodogram
    pub(crate) name_prefix: String,
    // In can be re-defined in MultiColorPeriodogram
    pub(crate) description_suffix: String,
    periodogram_algorithm: PeriodogramPower<T>,
    properties: Box<EvaluatorProperties>,
}

impl<T, F> Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    #[inline]
    pub fn default_peaks() -> usize {
        PeriodogramPeaks::default_peaks()
    }

    #[inline]
    pub fn default_resolution() -> f32 {
        10.0
    }

    #[inline]
    pub fn default_max_freq_factor() -> f32 {
        1.0
    }

    /// Set frequency resolution
    ///
    /// The larger frequency resolution allows to find peak period with better precision
    pub fn set_freq_resolution(&mut self, resolution: f32) -> &mut Self {
        self.resolution = resolution;
        self
    }

    /// Multiply maximum (Nyquist) frequency
    ///
    /// Maximum frequency is Nyquist frequncy multiplied by this factor. The larger factor allows
    /// to find larger frequency and makes [PeriodogramPowerFft] more precise. However large
    /// frequencies can show false peaks
    pub fn set_max_freq_factor(&mut self, max_freq_factor: f32) -> &mut Self {
        self.max_freq_factor = max_freq_factor;
        self
    }

    /// Define Nyquist frequency
    pub fn set_nyquist(&mut self, nyquist: NyquistFreq) -> &mut Self {
        self.nyquist = nyquist;
        self
    }

    /// Extend a feature to extract from periodogram
    pub fn add_feature(&mut self, feature: F) -> &mut Self {
        self.properties.info.size += feature.size_hint();
        self.properties.names.extend(
            feature
                .get_names()
                .iter()
                .map(|name| format!("{}_{}", self.name_prefix, name)),
        );
        self.properties.descriptions.extend(
            feature
                .get_descriptions()
                .into_iter()
                .map(|desc| format!("{} {}", desc, self.description_suffix)),
        );
        self.feature_extractor.add_feature(feature);
        self
    }

    pub fn set_periodogram_algorithm(
        &mut self,
        periodogram_power: PeriodogramPower<T>,
    ) -> &mut Self {
        self.periodogram_algorithm = periodogram_power;
        self
    }

    pub(crate) fn periodogram(&self, t: &[T]) -> periodogram::Periodogram<T> {
        periodogram::Periodogram::from_t(
            self.periodogram_algorithm.clone(),
            t,
            self.resolution,
            self.max_freq_factor,
            self.nyquist.clone(),
        )
    }

    pub fn power(&self, ts: &mut TimeSeries<T>) -> Array1<T> {
        self.periodogram(ts.t.as_slice()).power(ts)
    }

    pub fn freq_power(&self, ts: &mut TimeSeries<T>) -> (Array1<T>, Array1<T>) {
        let p = self.periodogram(ts.t.as_slice());
        let power = p.power(ts);
        let freq = (0..power.len()).map(|i| p.freq_by_index(i)).collect();
        (freq, power)
    }
}

impl<T, F> Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks>,
{
    /// New [Periodogram] that finds given number of peaks
    pub fn new(peaks: usize) -> Self {
        Self::with_name_description(
            peaks,
            "periodogram",
            "of periodogram (interpreting frequency as time, power as magnitude)",
        )
    }

    pub(crate) fn with_name_description(
        peaks: usize,
        name_prefix: impl ToString,
        description_suffix: impl ToString,
    ) -> Self {
        let info = EvaluatorInfo {
            size: 0,
            min_ts_length: 2,
            t_required: true,
            m_required: true,
            w_required: false,
            sorting_required: true,
            variability_required: false,
        };
        let mut slf = Self {
            properties: EvaluatorProperties {
                info,
                names: vec![],
                descriptions: vec![],
            }
            .into(),
            resolution: Self::default_resolution(),
            name_prefix: name_prefix.to_string(),
            description_suffix: description_suffix.to_string(),
            max_freq_factor: Self::default_max_freq_factor(),
            nyquist: AverageNyquistFreq.into(),
            feature_extractor: FeatureExtractor::new(vec![]),
            periodogram_algorithm: PeriodogramPowerFft::new().into(),
        };
        slf.add_feature(PeriodogramPeaks::new(peaks).into());
        slf
    }
}

impl<T, F> Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn transform_ts(&self, ts: &mut TimeSeries<T>) -> Result<TmArrays<T>, EvaluatorError> {
        self.check_ts(ts)?;
        let (freq, power) = self.freq_power(ts);
        Ok(TmArrays { t: freq, m: power })
    }
}

impl<T, F> Periodogram<T, F>
where
    T: Float,
{
    pub const fn doc() -> &'static str {
        DOC
    }
}

impl<T, F> Default for Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks>,
{
    fn default() -> Self {
        Self::new(Self::default_peaks())
    }
}

impl<T, F> EvaluatorInfoTrait for Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}

impl<T, F> FeatureNamesDescriptionsTrait for Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn get_names(&self) -> Vec<&str> {
        self.properties.names.iter().map(String::as_str).collect()
    }

    fn get_descriptions(&self) -> Vec<&str> {
        self.properties
            .descriptions
            .iter()
            .map(String::as_str)
            .collect()
    }
}

impl<T, F> FeatureEvaluator<T> for Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    transformer_eval!();
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Periodogram", bound = "T: Float, F: FeatureEvaluator<T>")]
struct PeriodogramParameters<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    resolution: f32,
    max_freq_factor: f32,
    nyquist: NyquistFreq,
    features: Vec<F>,
    peaks: usize,
    periodogram_algorithm: PeriodogramPower<T>,
}

impl<T, F> From<Periodogram<T, F>> for PeriodogramParameters<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn from(f: Periodogram<T, F>) -> Self {
        let Periodogram {
            resolution,
            max_freq_factor,
            nyquist,
            feature_extractor,
            periodogram_algorithm,
            ..
        } = f;

        let mut features = feature_extractor.into_vec();
        let rest_of_features = features.split_off(1);
        let periodogram_peaks: PeriodogramPeaks = features.pop().unwrap().try_into().unwrap();
        let peaks = periodogram_peaks.get_peaks();

        Self {
            resolution,
            max_freq_factor,
            nyquist,
            features: rest_of_features,
            peaks,
            periodogram_algorithm,
        }
    }
}

impl<T, F> From<PeriodogramParameters<T, F>> for Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks>,
{
    fn from(p: PeriodogramParameters<T, F>) -> Self {
        let PeriodogramParameters {
            resolution,
            max_freq_factor,
            nyquist,
            features,
            peaks,
            periodogram_algorithm,
        } = p;

        let mut periodogram = Periodogram::new(peaks);
        periodogram.set_freq_resolution(resolution);
        periodogram.set_max_freq_factor(max_freq_factor);
        periodogram.set_nyquist(nyquist);
        for feature in features {
            periodogram.add_feature(feature);
        }
        periodogram.set_periodogram_algorithm(periodogram_algorithm);
        periodogram
    }
}

impl<T, F> JsonSchema for Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    json_schema!(PeriodogramParameters<T, F>, false);
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::features::amplitude::Amplitude;
    use crate::periodogram::{PeriodogramPowerDirect, QuantileNyquistFreq};
    use crate::tests::*;

    check_feature!(Periodogram<f64, Feature<f64>>);

    serde_json_test!(
        periodogram_ser_json_de_non_default,
        Periodogram<f64, Feature<f64>>,
        {
            let mut periodogram = Periodogram::default();
            periodogram.add_feature(Amplitude::default().into());
            periodogram.set_periodogram_algorithm(PeriodogramPowerDirect.into());
            periodogram
        },
    );

    eval_info_test!(periodogram_info_1, {
        let mut periodogram = Periodogram::default();
        periodogram.set_periodogram_algorithm(PeriodogramPowerDirect.into());
        periodogram
    });

    eval_info_test!(periodogram_info_2, {
        let mut periodogram = Periodogram::new(5);
        periodogram.set_periodogram_algorithm(PeriodogramPowerDirect.into());
        periodogram
    });

    eval_info_test!(periodogram_info_3, {
        let mut periodogram = Periodogram::default();
        periodogram.add_feature(Amplitude::default().into());
        periodogram.set_periodogram_algorithm(PeriodogramPowerDirect.into());
        periodogram
    });

    #[test]
    fn periodogram_plateau() {
        let periodogram: Periodogram<_, Feature<_>> = Periodogram::default();
        let x = linspace(0.0_f32, 1.0, 100);
        let y = [0.0_f32; 100];
        let mut ts = TimeSeries::new_without_weight(&x, &y);
        let desired = vec![0.0, 0.0];
        let actual = periodogram.eval(&mut ts).unwrap();
        assert_eq!(desired, actual);
    }

    #[test]
    fn periodogram_evenly_sinus() {
        let periodogram: Periodogram<_, Feature<_>> = Periodogram::default();
        let mut rng = StdRng::seed_from_u64(0);
        let period = 0.17;
        let x = linspace(0.0_f32, 1.0, 101);
        let y: Vec<_> = x
            .iter()
            .map(|&x| {
                3.0 * f32::sin(2.0 * std::f32::consts::PI / period * x + 0.5)
                    + 4.0
                    + 0.01 * rng.gen::<f32>() // noise stabilizes solution
            })
            .collect();
        let mut ts = TimeSeries::new_without_weight(&x[..], &y[..]);
        let desired = [period];
        let actual = [periodogram.eval(&mut ts).unwrap()[0]]; // Test period only
        all_close(&desired[..], &actual[..], 5e-3);
    }

    #[test]
    fn periodogram_unevenly_sinus() {
        let periodogram: Periodogram<_, Feature<_>> = Periodogram::default();
        let period = 0.17;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..100).map(|_| rng.gen()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| 3.0 * f32::sin(2.0 * std::f32::consts::PI / period * x + 0.5) + 4.0)
            .collect();
        let mut ts = TimeSeries::new_without_weight(&x[..], &y[..]);
        let desired = [period];
        let actual = [periodogram.eval(&mut ts).unwrap()[0]]; // Test period only
        all_close(&desired[..], &actual[..], 5e-3);
    }

    #[test]
    fn periodogram_one_peak_vs_two_peaks() {
        let fe = FeatureExtractor::<_, Periodogram<_, Feature<_>>>::new(vec![
            Periodogram::new(1),
            Periodogram::new(2),
        ]);
        let period = 0.17;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..100).map(|_| rng.gen()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| 3.0 * f32::sin(2.0 * std::f32::consts::PI / period * x + 0.5) + 4.0)
            .collect();
        let mut ts = TimeSeries::new_without_weight(&x[..], &y[..]);
        let features = fe.eval(&mut ts).unwrap();
        all_close(
            &[features[0], features[1]],
            &[features[2], features[3]],
            1e-6,
        );
    }

    #[test]
    fn periodogram_unevenly_sinus_cosine() {
        let periodogram: Periodogram<_, Feature<_>> = Periodogram::new(2);
        let period1 = 0.0753;
        let period2 = 0.45;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..1000).map(|_| rng.gen()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| {
                3.0 * f32::sin(2.0 * std::f32::consts::PI / period1 * x + 0.5)
                    + -5.0 * f32::cos(2.0 * std::f32::consts::PI / period2 * x + 0.5)
                    + 4.0
            })
            .collect();
        let mut ts = TimeSeries::new_without_weight(&x[..], &y[..]);
        let desired = [period2, period1];
        let features = periodogram.eval(&mut ts).unwrap();
        let actual = [features[0], features[2]]; // Test period only
        all_close(&desired[..], &actual[..], 1e-2);
        assert!(features[1] > features[3]);
    }

    #[test]
    fn periodogram_unevenly_sinus_cosine_noised() {
        let periodogram: Periodogram<_, Feature<_>> = Periodogram::new(2);
        let period1 = 0.0753;
        let period2 = 0.46;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..1000).map(|_| rng.gen()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| {
                3.0 * f32::sin(2.0 * std::f32::consts::PI / period1 * x + 0.5)
                    + -5.0 * f32::cos(2.0 * std::f32::consts::PI / period2 * x + 0.5)
                    + 10.0 * rng.gen::<f32>()
                    + 4.0
            })
            .collect();
        let mut ts = TimeSeries::new_without_weight(&x[..], &y[..]);
        let desired = [period2, period1];
        let features = periodogram.eval(&mut ts).unwrap();
        let actual = [features[0], features[2]]; // Test period only
        all_close(&desired[..], &actual[..], 1e-2);
        assert!(features[1] > features[3]);
    }

    #[test]
    fn periodogram_different_time_scales() {
        let mut periodogram: Periodogram<_, Feature<_>> = Periodogram::new(2);
        periodogram
            .set_nyquist(QuantileNyquistFreq { quantile: 0.05 }.into())
            .set_freq_resolution(10.0)
            .set_max_freq_factor(1.0)
            .set_periodogram_algorithm(PeriodogramPowerFft::new().into());
        let period1 = 0.01;
        let period2 = 1.0;
        let n = 100;
        let mut x = linspace(0.0, 0.1, n);
        x.append(&mut linspace(1.0, 10.0, n));
        let y: Vec<_> = x
            .iter()
            .map(|&x| {
                3.0 * f32::sin(2.0 * std::f32::consts::PI / period1 * x + 0.5)
                    + -5.0 * f32::cos(2.0 * std::f32::consts::PI / period2 * x + 0.5)
                    + 4.0
            })
            .collect();
        let mut ts = TimeSeries::new_without_weight(&x, &y);
        let features = periodogram.eval(&mut ts).unwrap();
        assert!(f32::abs(features[0] - period2) / period2 < 1.0 / n as f32);
        assert!(f32::abs(features[2] - period1) / period1 < 1.0 / n as f32);
        assert!(features[1] > features[3]);
    }
}
