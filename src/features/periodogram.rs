use crate::evaluator::*;
use crate::extractor::FeatureExtractor;
use crate::peak_indices::peak_indices_reverse_sorted;
use crate::periodogram;
use crate::periodogram::{
    AverageNyquistFreq, FreqGrid, FreqGridStrategy, NyquistFreq, PeriodogramPower,
    PeriodogramPowerError, PeriodogramPowerFft,
};

use std::convert::TryInto;
use std::fmt::Debug;
use std::hash::Hash;
use std::iter;

fn number_ending(i: usize) -> &'static str {
    #[allow(clippy::match_same_arms)]
    match (i % 10, i % 100) {
        (1, 11) => "th",
        (1, _) => "st",
        (2, 12) => "th",
        (2, _) => "nd",
        (3, 13) => "th",
        (3, _) => "rd",
        (_, _) => "th",
    }
}

macro_const! {
    const PERIODOGRAM_PEAK_DOC: &'static str = r#"
Peak evaluator for [Periodogram]
"#;
}

#[doc(hidden)]
#[doc = PERIODOGRAM_PEAK_DOC!()]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(
    from = "PeriodogramPeaksParameters",
    into = "PeriodogramPeaksParameters"
)]
pub struct PeriodogramPeaks {
    peaks: usize,
    properties: Box<EvaluatorProperties>,
}

impl PeriodogramPeaks {
    pub fn new(peaks: usize) -> Self {
        assert!(peaks > 0, "Number of peaks should be at least one");
        let info = EvaluatorInfo {
            size: 2 * peaks,
            min_ts_length: 1,
            t_required: true,
            m_required: true,
            w_required: false,
            sorting_required: true,
        };
        let names = (0..peaks)
            .flat_map(|i| vec![format!("period_{}", i), format!("period_s_to_n_{}", i)])
            .collect();
        let descriptions = (0..peaks)
            .flat_map(|i| {
                vec![
                    format!(
                        "period of the {}{} highest peak of periodogram",
                        i + 1,
                        number_ending(i + 1),
                    ),
                    format!(
                        "Spectral density to spectral density standard deviation ratio of \
                            the {}{} highest peak of periodogram",
                        i + 1,
                        number_ending(i + 1)
                    ),
                ]
            })
            .collect();
        Self {
            properties: EvaluatorProperties {
                info,
                names,
                descriptions,
            }
            .into(),
            peaks,
        }
    }

    #[inline]
    pub fn default_peaks() -> usize {
        1
    }

    pub const fn doc() -> &'static str {
        PERIODOGRAM_PEAK_DOC
    }
}

impl Default for PeriodogramPeaks {
    fn default() -> Self {
        Self::new(Self::default_peaks())
    }
}

impl EvaluatorInfoTrait for PeriodogramPeaks {
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}
impl FeatureNamesDescriptionsTrait for PeriodogramPeaks {
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

impl<T> FeatureEvaluator<T> for PeriodogramPeaks
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let peak_indices = peak_indices_reverse_sorted(&ts.m.sample);
        Ok(peak_indices
            .iter()
            .flat_map(|&i| {
                iter::once(T::two() * T::PI() / ts.t.sample[i])
                    .chain(iter::once(ts.m.signal_to_noise(ts.m.sample[i])))
            })
            .chain(iter::repeat(T::zero()))
            .take(2 * self.peaks)
            .collect())
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "PeriodogramPeaks")]
struct PeriodogramPeaksParameters {
    peaks: usize,
}

impl From<PeriodogramPeaks> for PeriodogramPeaksParameters {
    fn from(f: PeriodogramPeaks) -> Self {
        Self { peaks: f.peaks }
    }
}

impl From<PeriodogramPeaksParameters> for PeriodogramPeaks {
    fn from(p: PeriodogramPeaksParameters) -> Self {
        Self::new(p.peaks)
    }
}

impl JsonSchema for PeriodogramPeaks {
    json_schema!(PeriodogramPeaksParameters, false);
}

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
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    bound = "T: Float, F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>, <F as std::convert::TryInto<PeriodogramPeaks>>::Error: Debug,",
    from = "PeriodogramParameters<T, F>",
    into = "PeriodogramParameters<T, F>"
)]
pub struct Periodogram<T, F>
where
    T: Float,
{
    freq_grid_strategy: FreqGridStrategy<T>,
    feature_extractor: FeatureExtractor<T, F>,
    periodogram_algorithm: PeriodogramPower<T>,
    properties: Box<EvaluatorProperties>,
}

impl<T, F> std::hash::Hash for Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.freq_grid_strategy.hash(state);
        self.feature_extractor.hash(state);
        self.periodogram_algorithm.hash(state);
        self.properties.hash(state);
    }
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
    /// The larger frequency resolution allows to find peak period with better precision.
    ///
    /// Returns [None] if the underlying freq_grid_strategy is [FreqGridStrategy::Fixed],
    /// changes the resolution and returns [Some] if it is [FreqGridStrategy::Dynamic].
    pub fn set_freq_resolution(&mut self, resolution: f32) -> Option<&mut Self> {
        match &mut self.freq_grid_strategy {
            FreqGridStrategy::Fixed(_) => None,
            FreqGridStrategy::Dynamic(params) => {
                params.resolution = resolution;
                Some(self)
            }
        }
    }

    /// Set maximum (Nyquist) frequency multiplier
    ///
    /// Maximum frequency is Nyquist frequency multiplied by this factor. The larger factor allows
    /// to find larger frequency and makes [PeriodogramPowerFft] more precise. However, large
    /// frequencies can show false peaks.
    ///
    /// Returns [None] if the underlying freq_grid_strategy is [FreqGridStrategy::Fixed],
    /// changes the multiplier and returns [Some] if it is [FreqGridStrategy::Dynamic].
    pub fn set_max_freq_factor(&mut self, max_freq_factor: f32) -> Option<&mut Self> {
        match &mut self.freq_grid_strategy {
            FreqGridStrategy::Fixed(_) => None,
            FreqGridStrategy::Dynamic(params) => {
                params.max_freq_factor = max_freq_factor;
                Some(self)
            }
        }
    }

    /// Set Nyquist frequency strategy
    ///
    /// Returns [None] if the underlying freq_grid_strategy is [FreqGridStrategy::Fixed],
    /// changes the resolution and returns [Some] if it is [FreqGridStrategy::Dynamic].
    pub fn set_nyquist(&mut self, nyquist: impl Into<NyquistFreq>) -> Option<&mut Self> {
        match &mut self.freq_grid_strategy {
            FreqGridStrategy::Fixed(_) => None,
            FreqGridStrategy::Dynamic(params) => {
                params.nyquist = nyquist.into();
                Some(self)
            }
        }
    }

    /// Set fixed frequency grid
    ///
    /// Changes the underlying frequency grid to the given one.
    pub fn set_freq_grid(&mut self, freq_grid: impl Into<FreqGrid<T>>) -> &mut Self {
        self.freq_grid_strategy = FreqGridStrategy::Fixed(freq_grid.into());
        self
    }

    /// Set a new [FreqGridStrategy]
    pub fn set_freq_grid_strategy(
        &mut self,
        freq_grid_strategy: impl Into<FreqGridStrategy<T>>,
    ) -> &mut Self {
        self.freq_grid_strategy = freq_grid_strategy.into();
        self
    }

    /// Extend a feature to extract from periodogram
    pub fn add_feature(&mut self, feature: F) -> &mut Self {
        self.properties.info.size += feature.size_hint();
        self.properties.names.extend(
            feature
                .get_names()
                .iter()
                .map(|name| "periodogram_".to_owned() + name),
        );
        self.properties.descriptions.extend(
            feature
                .get_descriptions()
                .into_iter()
                .map(|desc| format!("{desc} of periodogram")),
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

    fn periodogram(
        &self,
        ts: &mut TimeSeries<T>,
    ) -> Result<periodogram::Periodogram<'_, T>, PeriodogramPowerError> {
        periodogram::Periodogram::from_t(
            self.periodogram_algorithm.clone(),
            ts.t.as_slice(),
            &self.freq_grid_strategy,
        )
    }

    pub fn power(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, PeriodogramPowerError> {
        Ok(self.periodogram(ts)?.power(ts))
    }

    pub fn freq_power(
        &self,
        ts: &mut TimeSeries<T>,
    ) -> Result<(Vec<T>, Vec<T>), PeriodogramPowerError> {
        let p = self.periodogram(ts)?;
        let power = p.power(ts);
        let freq = (0..power.len()).map(|i| p.freq(i)).collect();
        Ok((freq, power))
    }
}

impl<T, F> Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks>,
{
    /// New [Periodogram] that finds given number of peaks
    pub fn new(peaks: usize) -> Self {
        let freq_grid_strategy = FreqGridStrategy::dynamic(
            Self::default_resolution(),
            Self::default_max_freq_factor(),
            AverageNyquistFreq,
        );
        Self::with_freq_frid_strategy(peaks, freq_grid_strategy)
    }

    /// New [Periodogram] with given number of peaks and [FreqGridStrategy] or [FreqGrid]
    pub fn with_freq_frid_strategy(
        peaks: usize,
        freq_grid_strategy: impl Into<FreqGridStrategy<T>>,
    ) -> Self {
        let peaks = PeriodogramPeaks::new(peaks);
        let peak_names = peaks.properties.names.clone();
        let peak_descriptions = peaks.properties.descriptions.clone();
        let peaks_size_hint = peaks.size_hint();
        let peaks_min_ts_length = peaks.min_ts_length();
        let info = EvaluatorInfo {
            size: peaks_size_hint,
            min_ts_length: usize::max(peaks_min_ts_length, 2),
            t_required: true,
            m_required: true,
            w_required: false,
            sorting_required: true,
        };
        Self {
            properties: EvaluatorProperties {
                info,
                names: peak_names,
                descriptions: peak_descriptions,
            }
            .into(),
            freq_grid_strategy: freq_grid_strategy.into(),
            feature_extractor: FeatureExtractor::new(vec![peaks.into()]),
            periodogram_algorithm: PeriodogramPowerFft::new().into(),
        }
    }
}

impl<T, F> Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn transform_ts(&self, ts: &mut TimeSeries<T>) -> Result<TmArrays<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let (freq, power) = self.freq_power(ts)?;
        Ok(TmArrays {
            t: freq.into(),
            m: power.into(),
        })
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
    <F as std::convert::TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}

impl<T, F> FeatureNamesDescriptionsTrait for Periodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as std::convert::TryInto<PeriodogramPeaks>>::Error: Debug,
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
    <F as std::convert::TryInto<PeriodogramPeaks>>::Error: Debug,
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
    freq_grid_strategy: FreqGridStrategy<T>,
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
            freq_grid_strategy,
            feature_extractor,
            periodogram_algorithm,
            properties: _,
        } = f;

        let mut features = feature_extractor.into_vec();
        let rest_of_features = features.split_off(1);
        let periodogram_peaks: PeriodogramPeaks = features.pop().unwrap().try_into().unwrap();
        let peaks = periodogram_peaks.peaks;

        Self {
            freq_grid_strategy,
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
            freq_grid_strategy,
            features,
            peaks,
            periodogram_algorithm,
        } = p;

        let mut periodogram = Periodogram::with_freq_frid_strategy(peaks, freq_grid_strategy);
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
    use rand_distr::StandardNormal;

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

    serde_json_test!(
        periodogram_ser_json_de_freq_grid,
        Periodogram<f64, Feature<f64>>,
        {
            let freq_grid = FreqGrid::linear(0.5, 50.0, 200);
            let mut periodogram = Periodogram::with_freq_frid_strategy(4, freq_grid);
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
                    + 0.01 * rng.random::<f32>() // noise stabilizes solution
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
        let mut x: Vec<f32> = (0..100).map(|_| rng.random()).collect();
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
        let mut x: Vec<f32> = (0..100).map(|_| rng.random()).collect();
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
        let mut x: Vec<f32> = (0..1000).map(|_| rng.random()).collect();
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
        let mut x: Vec<f32> = (0..1000).map(|_| rng.random()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<_> = x
            .iter()
            .map(|&x| {
                3.0 * f32::sin(2.0 * std::f32::consts::PI / period1 * x + 0.5)
                    + -5.0 * f32::cos(2.0 * std::f32::consts::PI / period2 * x + 0.5)
                    + 10.0 * rng.random::<f32>()
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
        let freq_grid_strategy =
            FreqGridStrategy::dynamic(10.0, 1.0, QuantileNyquistFreq { quantile: 0.1 });
        let mut periodogram: Periodogram<_, Feature<_>> =
            Periodogram::with_freq_frid_strategy(2, freq_grid_strategy);
        periodogram.set_periodogram_algorithm(PeriodogramPowerFft::new().into());
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

    #[test]
    fn periodogram_arbitrary_vs_linear() {
        // Create a time series with two frequencies
        let period1 = 0.17;
        let period2 = 0.46;
        let n = 1000;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f32> = (0..n).map(|_| rng.random()).collect();
        x[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let amplitude1: f32 = 3.0;
        let amplitude2: f32 = 5.0;
        let noise_std = 0.1 * (amplitude1.abs() + amplitude2.abs());
        let y: Vec<_> = x
            .iter()
            .map(|&x| {
                3.0 * f32::sin(2.0 * std::f32::consts::PI / period1 * x + 0.5)
                    + -amplitude2 * f32::cos(2.0 * std::f32::consts::PI / period2 * x + 0.5)
                    + 4.0
                    + noise_std * rng.sample::<f32, _>(StandardNormal)
            })
            .collect();
        let mut ts = TimeSeries::new_without_weight(&x[..], &y[..]);

        // Create frequency grids
        let start = 0.1;
        let step = 0.01;
        let size = 100;
        let linear_grid = FreqGrid::linear(start, step, size);
        let freqs: Vec<_> = (0..size).map(|i| start + step * i as f32).collect();
        let arbitrary_grid = FreqGrid::try_from_sorted_array(freqs).unwrap();

        // Create periodograms with different grids
        let mut periodogram_linear: Periodogram<f32, Feature<f32>> = Periodogram::default();
        periodogram_linear.set_freq_grid(linear_grid);
        periodogram_linear.set_periodogram_algorithm(PeriodogramPowerDirect.into());

        let mut periodogram_arbitrary: Periodogram<f32, Feature<f32>> = Periodogram::default();
        periodogram_arbitrary.set_freq_grid(arbitrary_grid);
        periodogram_arbitrary.set_periodogram_algorithm(PeriodogramPowerDirect.into());

        // Compare results
        let features_linear = periodogram_linear.eval(&mut ts).unwrap();
        let features_arbitrary = periodogram_arbitrary.eval(&mut ts).unwrap();

        // The results should be very close since we used the same frequency points
        all_close(&features_linear[..], &features_arbitrary[..], 1e-10);
    }
}
