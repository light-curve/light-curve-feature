use crate::evaluator::*;
use crate::extractor::FeatureExtractor;
use crate::features::_periodogram_peaks::PeriodogramPeaks;
use crate::periodogram;
use crate::periodogram::{
    AverageNyquistFreq, DefaultPeriodogramPowerFft, FreqGrid, FreqGridStrategy, NyquistFreq,
    PeriodogramNormalization, PeriodogramPower, PeriodogramPowerError,
};

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
[Periodogram] to another one if you are crazy enough.

Additionally, [Periodogram] supports phase features: features extracted from the light curve
phase-folded at the best period. The phase runs from 0 to 1, with phase 0 at the magnitude minimum.
Phase feature names are prefixed with `period_folded_`.

- Depends on: **time**, **magnitude**
- Minimum number of observations: as required by sub-features, but at least two
- Number of features: **$2 \times \mathrm{peaks}$** plus spectrum sub-features plus phase sub-features
"#;
}

/// Phase-fold a time series at a given period, with phase 0 at the magnitude minimum.
///
/// Returns a new `TmwArrays` where `t` is the phase in `[0, 1)`, `m` is the magnitude,
/// and `w` are the original weights, all sorted by phase.
pub(crate) fn phase_fold_ts<T: Float>(ts: &mut TimeSeries<T>, period: T) -> TmwArrays<T> {
    let t = ts.t.as_slice();
    let m = ts.m.as_slice();
    let w = ts.w.as_slice();

    // phase in [0, 1)
    let mut phases: Vec<T> = t
        .iter()
        .map(|&ti| {
            let p = (ti / period) % T::one();
            if p < T::zero() { p + T::one() } else { p }
        })
        .collect();

    // index of minimum m (zero phase is defined here)
    let min_idx = m
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);
    let phase_offset = phases[min_idx];

    // shift so minimum-m observation is at phase 0
    for p in &mut phases {
        *p = (*p - phase_offset + T::one()) % T::one();
    }

    // sort by phase
    let mut triples: Vec<(T, T, T)> = phases
        .into_iter()
        .zip(m.iter().copied())
        .zip(w.iter().copied())
        .map(|((ph, mi), wi)| (ph, mi, wi))
        .collect();
    triples.sort_unstable_by(|(a, _, _), (b, _, _)| {
        a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
    });

    let (sorted_phases, sorted_m, sorted_w) = triples.into_iter().fold(
        (Vec::new(), Vec::new(), Vec::new()),
        |(mut ps, mut ms, mut ws), (p, m, w)| {
            ps.push(p);
            ms.push(m);
            ws.push(w);
            (ps, ms, ws)
        },
    );
    TmwArrays {
        t: ndarray::Array1::from_vec(sorted_phases),
        m: ndarray::Array1::from_vec(sorted_m),
        w: ndarray::Array1::from_vec(sorted_w),
    }
}

#[doc = DOC!()]
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(
    bound = "T: Float, F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>, <F as TryInto<PeriodogramPeaks>>::Error: Debug,",
    from = "PeriodogramParameters<T, F>",
    into = "PeriodogramParameters<T, F>"
)]
pub struct Periodogram<T, F>
where
    T: Float,
{
    freq_grid_strategy: FreqGridStrategy<T>,
    pub(crate) spectrum_extractor: FeatureExtractor<T, F>,
    pub(crate) phase_extractor: FeatureExtractor<T, F>,
    // Can be re-defined in MultiColorPeriodogram
    pub(crate) name_prefix: String,
    // Can be re-defined in MultiColorPeriodogram
    pub(crate) description_suffix: String,
    periodogram_algorithm: PeriodogramPower<T>,
    normalization: PeriodogramNormalization,
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

    /// Default normalization strategy (Psd)
    #[inline]
    pub fn default_normalization() -> PeriodogramNormalization {
        PeriodogramNormalization::default()
    }

    /// Set the power normalization strategy
    pub fn set_normalization(&mut self, normalization: PeriodogramNormalization) -> &mut Self {
        self.normalization = normalization;
        self
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

    /// Add a feature to extract from the periodogram spectrum (frequency as time, power as magnitude)
    pub fn add_spectrum_feature(&mut self, feature: F) -> &mut Self {
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
        self.spectrum_extractor.add_feature(feature);
        self
    }

    /// Add a feature to extract from the phase-folded light curve at the best period.
    ///
    /// Feature names are prefixed with `period_folded_`. Phase runs from 0 to 1 with phase 0
    /// at the magnitude minimum. The phase-folded series is sorted by phase.
    pub fn add_phase_feature(&mut self, feature: F) -> &mut Self {
        self.properties.info.size += feature.size_hint();
        self.properties.info.min_ts_length = self
            .properties
            .info
            .min_ts_length
            .max(feature.min_ts_length());
        self.properties.names.extend(
            feature
                .get_names()
                .iter()
                .map(|name| format!("period_folded_{}", name)),
        );
        self.properties.descriptions.extend(
            feature
                .get_descriptions()
                .into_iter()
                .map(|desc| format!("{} (light curve folded at the best period)", desc)),
        );
        self.phase_extractor.add_feature(feature);
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
        self.periodogram_from_t(ts.t.as_slice())
    }

    pub(crate) fn periodogram_from_t(
        &self,
        t: &[T],
    ) -> Result<periodogram::Periodogram<'_, T>, PeriodogramPowerError> {
        periodogram::Periodogram::from_t(
            self.periodogram_algorithm.clone(),
            t,
            &self.freq_grid_strategy,
            self.normalization,
        )
    }

    pub(crate) fn spectrum_extractor_ref(&self) -> &FeatureExtractor<T, F> {
        &self.spectrum_extractor
    }

    #[allow(dead_code)]
    pub(crate) fn phase_extractor_ref(&self) -> &FeatureExtractor<T, F> {
        &self.phase_extractor
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
        Self::with_freq_grid_strategy_name_description(
            peaks,
            freq_grid_strategy,
            "periodogram",
            "of periodogram (interpreting frequency as time, power as magnitude)",
        )
    }

    pub(crate) fn with_name_description(
        peaks: usize,
        name_prefix: impl ToString,
        description_suffix: impl ToString,
    ) -> Self {
        let freq_grid_strategy = FreqGridStrategy::dynamic(
            Self::default_resolution(),
            Self::default_max_freq_factor(),
            AverageNyquistFreq,
        );
        Self::with_freq_grid_strategy_name_description(
            peaks,
            freq_grid_strategy,
            name_prefix,
            description_suffix,
        )
    }

    fn with_freq_grid_strategy_name_description(
        peaks: usize,
        freq_grid_strategy: impl Into<FreqGridStrategy<T>>,
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
            freq_grid_strategy: freq_grid_strategy.into(),
            name_prefix: name_prefix.to_string(),
            description_suffix: description_suffix.to_string(),
            spectrum_extractor: FeatureExtractor::new(vec![]),
            phase_extractor: FeatureExtractor::new(vec![]),
            periodogram_algorithm: DefaultPeriodogramPowerFft::new().into(),
            normalization: PeriodogramNormalization::default(),
        };
        slf.add_spectrum_feature(PeriodogramPeaks::new(peaks).into());
        slf
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
    fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        let (freq, power) = self.freq_power(ts)?;
        let mut spectrum_ts = TmArrays {
            t: freq.into(),
            m: power.into(),
        }
        .ts();
        let mut result = self.spectrum_extractor.eval(&mut spectrum_ts)?;
        if !self.phase_extractor.get_features().is_empty() {
            let best_period = result[0];
            if !best_period.is_finite() || best_period <= T::zero() {
                return Err(EvaluatorError::ZeroDivision(
                    "best period from periodogram is not positive, cannot phase-fold",
                ));
            }
            let phase_arrays = phase_fold_ts(ts, best_period);
            let mut phase_ts = phase_arrays.ts();
            result.extend(self.phase_extractor.eval(&mut phase_ts)?);
        }
        Ok(result)
    }

    fn eval_or_fill(&self, ts: &mut TimeSeries<T>, fill_value: T) -> Vec<T> {
        let (freq, power) = match self.freq_power(ts) {
            Ok(x) => x,
            Err(_) => return vec![fill_value; self.size_hint()],
        };
        let mut spectrum_ts = TmArrays {
            t: freq.into(),
            m: power.into(),
        }
        .ts();
        let mut result = self
            .spectrum_extractor
            .eval_or_fill(&mut spectrum_ts, fill_value);
        if !self.phase_extractor.get_features().is_empty() {
            let best_period = result[0];
            if best_period.is_finite() && best_period > T::zero() {
                let phase_arrays = phase_fold_ts(ts, best_period);
                let mut phase_ts = phase_arrays.ts();
                result.extend(self.phase_extractor.eval_or_fill(&mut phase_ts, fill_value));
            } else {
                result.extend(vec![fill_value; self.phase_extractor.size_hint()]);
            }
        }
        result
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Periodogram", bound = "T: Float, F: FeatureEvaluator<T>")]
struct PeriodogramParameters<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    freq_grid_strategy: FreqGridStrategy<T>,
    spectrum_features: Vec<F>,
    #[serde(default)]
    phase_features: Vec<F>,
    peaks: usize,
    periodogram_algorithm: PeriodogramPower<T>,
    #[serde(default)]
    normalization: PeriodogramNormalization,
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
            spectrum_extractor,
            phase_extractor,
            periodogram_algorithm,
            normalization,
            properties: _,
            ..
        } = f;

        let mut features = spectrum_extractor.into_vec();
        let rest_of_features = features.split_off(1);
        let periodogram_peaks: PeriodogramPeaks = features.pop().unwrap().try_into().unwrap();
        let peaks = periodogram_peaks.get_peaks();

        Self {
            freq_grid_strategy,
            spectrum_features: rest_of_features,
            phase_features: phase_extractor.into_vec(),
            peaks,
            periodogram_algorithm,
            normalization,
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
            spectrum_features,
            phase_features,
            peaks,
            periodogram_algorithm,
            normalization,
        } = p;

        let mut periodogram = Periodogram::with_freq_frid_strategy(peaks, freq_grid_strategy);
        for feature in spectrum_features {
            periodogram.add_spectrum_feature(feature);
        }
        for feature in phase_features {
            periodogram.add_phase_feature(feature);
        }
        periodogram.set_periodogram_algorithm(periodogram_algorithm);
        periodogram.set_normalization(normalization);
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
    use crate::features::lafler_kinman::LaflerKinman;
    use crate::periodogram::{PeriodogramPowerDirect, QuantileNyquistFreq};
    use crate::tests::*;
    use rand_distr::StandardNormal;

    check_feature!(Periodogram<f64, Feature<f64>>);

    serde_json_test!(
        periodogram_ser_json_de_non_default,
        Periodogram<f64, Feature<f64>>,
        {
            let mut periodogram = Periodogram::default();
            periodogram.add_spectrum_feature(Amplitude::default().into());
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

    serde_json_test!(
        periodogram_ser_json_de_with_phase_features,
        Periodogram<f64, Feature<f64>>,
        {
            let mut periodogram = Periodogram::default();
            periodogram.add_phase_feature(LaflerKinman::new().into());
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
        periodogram.add_spectrum_feature(Amplitude::default().into());
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
        periodogram.set_periodogram_algorithm(DefaultPeriodogramPowerFft::new().into());
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

    #[test]
    fn periodogram_phase_feature_names() {
        let mut periodogram: Periodogram<f64, Feature<f64>> = Periodogram::new(1);
        periodogram.add_phase_feature(LaflerKinman::new().into());
        let names = periodogram.get_names();
        // spectrum: period_0, period_s_to_n_0; phase: period_folded_lafler_kinman
        assert_eq!(names[0], "periodogram_period_0");
        assert_eq!(names[1], "periodogram_period_s_to_n_0");
        assert_eq!(names[2], "period_folded_lafler_kinman");
        assert_eq!(periodogram.size_hint(), 3);
    }

    #[test]
    fn periodogram_phase_feature_recovery() {
        use crate::features::lafler_kinman::LaflerKinman;
        let period = 0.17_f64;
        let mut rng = StdRng::seed_from_u64(0);
        let mut x: Vec<f64> = (0..200).map(|_| rng.random()).collect();
        x.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let y: Vec<f64> = x
            .iter()
            .map(|&t| 3.0 * f64::sin(2.0 * std::f64::consts::PI / period * t + 0.5) + 4.0)
            .collect();

        let mut periodogram: Periodogram<f64, Feature<f64>> = Periodogram::new(1);
        periodogram.add_phase_feature(LaflerKinman::new().into());
        periodogram.set_periodogram_algorithm(PeriodogramPowerDirect.into());

        let mut ts = TimeSeries::new_without_weight(&x, &y);
        let result = periodogram.eval(&mut ts).unwrap();

        // result = [period_0, snr_0, period_folded_lafler_kinman]
        assert_eq!(result.len(), 3);
        // recovered period close to true
        assert!(
            (result[0] - period).abs() / period < 0.05,
            "period {}",
            result[0]
        );
        // smooth phase curve: theta < 0.5
        assert!(result[2] < 0.5, "lafler_kinman = {}", result[2]);
    }
}
