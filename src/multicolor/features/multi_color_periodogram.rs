use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{
    EvaluatorInfo, EvaluatorInfoTrait, EvaluatorProperties, FeatureEvaluator,
    FeatureNamesDescriptionsTrait, OwnedArrays, TmArrays,
};
use crate::extractor::FeatureExtractor;
use crate::features::phase_fold_or_compute;
use crate::features::{Periodogram, PeriodogramPeaks};
use crate::features::{eval_phase_ts, eval_phase_ts_or_fill};
use crate::float_trait::Float;
use crate::multicolor::multicolor_evaluator::*;
use crate::multicolor::{PassbandSet, PassbandTrait};
use crate::periodogram::{self, FreqGridStrategy, NyquistFreq, PeriodogramPower};

use ndarray::Array1;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Normalisation of the combined periodogram across passbands
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
pub enum MultiColorPeriodogramNormalisation {
    /// Weight individual periodograms by the number of observations in each passband.
    ///
    /// Useful when observations carry no explicit uncertainty weights. With this
    /// normalisation the combined power at each frequency is a weighted average of
    /// the per-band Lomb-Scargle powers, where bands with more observations
    /// contribute proportionally more.
    Count,
    /// Weight individual periodograms by $\chi^2 = \sum \left(\frac{m_i - \bar{m}}{\delta_i}\right)^2$
    ///
    /// Bands whose light curves show a larger overall variability (relative to
    /// measurement uncertainties) receive a higher weight. Be aware that if no
    /// weights are given to observations (i.e. via
    /// [TimeSeries::new_without_weight]) unity weights are assumed, and this is
    /// **not** equivalent to [Count] -- it weights by magnitude variance instead.
    Chi2,
}

/// Multi-passband periodogram
///
/// Combines per-band Lomb-Scargle periodograms into a single power spectrum
/// using a common frequency grid derived from the union of all observation
/// times in the specified passbands. Individual band powers are weighted and
/// summed according to the chosen [`MultiColorPeriodogramNormalisation`] strategy.
///
/// The set of passbands to include must be specified at construction time via
/// [`MultiColorPeriodogram::new`]. Input data is subsampled to the specified bands.
///
/// The frequency grid, peak-extraction features, and periodogram algorithm are
/// inherited from the underlying [`Periodogram`] and can be configured through
/// the same builder methods.
///
/// Phase features (added via [`MultiColorPeriodogram::add_phase_feature`]) are
/// applied per-band at the best period from the combined periodogram. A fixed
/// set of passbands must be registered with
/// [`MultiColorPeriodogram::set_phase_bands`] before adding phase features.
/// Phase feature names are prefixed with `period_folded_{band}_`.
///
/// # Example
///
/// ```rust,ignore
/// use std::collections::BTreeSet;
/// let passbands = ["g", "r"].iter().map(|&s| StringPassband::from(s)).collect::<BTreeSet<_>>();
/// let mut eval = MultiColorPeriodogram::<StringPassband, f64, Feature<f64>>::new(
///     1,
///     MultiColorPeriodogramNormalisation::Count,
///     passbands,
/// );
/// eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());
/// let result = eval.eval_multicolor(&mut mcts)?;
/// ```
#[derive(Clone, Debug)]
pub struct MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T>,
{
    // We use it to not reimplement some internals
    monochrome: Periodogram<T, F>,
    normalization: MultiColorPeriodogramNormalisation,
    passband_set: PassbandSet<P>,
    /// Passbands on which phase features are evaluated. Empty = no phase features.
    phase_bands: Vec<P>,
    phase_extractor: FeatureExtractor<T, F>,
    /// Combined properties (spectrum + phase). Updated whenever features/bands change.
    properties: Box<EvaluatorProperties>,
}

// ── Serialization ─────────────────────────────────────────────────────────────

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(
    rename = "MultiColorPeriodogram",
    bound(
        serialize = "P: PassbandTrait, T: Float, F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>, <F as TryInto<PeriodogramPeaks>>::Error: Debug",
        deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float, F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>, <F as TryInto<PeriodogramPeaks>>::Error: Debug",
    )
)]
struct MultiColorPeriodogramParameters<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T>,
{
    monochrome: Periodogram<T, F>,
    normalization: MultiColorPeriodogramNormalisation,
    passband_set: PassbandSet<P>,
    #[serde(default)]
    phase_bands: Vec<P>,
    #[serde(default)]
    phase_features: Vec<F>,
}

impl<P, T, F> From<MultiColorPeriodogram<P, T, F>> for MultiColorPeriodogramParameters<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn from(v: MultiColorPeriodogram<P, T, F>) -> Self {
        Self {
            monochrome: v.monochrome,
            normalization: v.normalization,
            passband_set: v.passband_set,
            phase_bands: v.phase_bands,
            phase_features: v.phase_extractor.into_vec(),
        }
    }
}

impl<P, T, F> From<MultiColorPeriodogramParameters<P, T, F>> for MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn from(p: MultiColorPeriodogramParameters<P, T, F>) -> Self {
        let mut mcp = Self::new_with_monochrome(p.monochrome, p.normalization, p.passband_set);
        if !p.phase_bands.is_empty() {
            mcp.set_phase_bands(p.phase_bands);
        }
        for feature in p.phase_features {
            mcp.add_phase_feature(feature);
        }
        mcp
    }
}

impl<P, T, F> Serialize for MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait + Serialize,
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks> + Serialize,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        MultiColorPeriodogramParameters::from(self.clone()).serialize(serializer)
    }
}

impl<'de, P, T, F> Deserialize<'de> for MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait + Deserialize<'de>,
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        MultiColorPeriodogramParameters::<P, T, F>::deserialize(deserializer).map(Self::from)
    }
}

impl<P, T, F> JsonSchema for MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait + JsonSchema,
    T: Float,
    F: FeatureEvaluator<T> + JsonSchema,
{
    fn is_referenceable() -> bool {
        false
    }

    fn schema_name() -> String {
        MultiColorPeriodogramParameters::<P, T, F>::schema_name()
    }

    fn json_schema(g: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        MultiColorPeriodogramParameters::<P, T, F>::json_schema(g)
    }
}

// ── Constructors & builders ───────────────────────────────────────────────────

impl<P, T, F> MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    /// Create a new multi-colour periodogram for the given set of passbands.
    ///
    /// `peaks` sets the number of highest-power peaks whose period and
    /// signal-to-noise ratio are returned as features.
    ///
    /// Input [MultiColorTimeSeries](crate::data::MultiColorTimeSeries) is subsampled
    /// to the specified passbands when this feature is evaluated.
    pub fn new(
        peaks: usize,
        normalization: MultiColorPeriodogramNormalisation,
        passbands: impl IntoIterator<Item = P>,
    ) -> Self {
        let monochrome = Periodogram::with_name_description(
            peaks,
            "multicolor_periodogram",
            "of multi-color periodogram (interpreting frequency as time, power as magnitude)",
        );
        let passband_set = PassbandSet(passbands.into_iter().collect());
        Self::new_with_monochrome(monochrome, normalization, passband_set)
    }

    fn new_with_monochrome(
        monochrome: Periodogram<T, F>,
        normalization: MultiColorPeriodogramNormalisation,
        passband_set: PassbandSet<P>,
    ) -> Self {
        let properties = EvaluatorProperties {
            info: monochrome.get_info().clone(),
            names: monochrome
                .get_names()
                .iter()
                .map(|s| s.to_string())
                .collect(),
            descriptions: monochrome
                .get_descriptions()
                .iter()
                .map(|s| s.to_string())
                .collect(),
        }
        .into();
        Self {
            monochrome,
            normalization,
            passband_set,
            phase_bands: vec![],
            phase_extractor: FeatureExtractor::new(vec![]),
            properties,
        }
    }

    #[inline]
    pub fn default_peaks() -> usize {
        PeriodogramPeaks::default_peaks()
    }

    #[inline]
    pub fn default_resolution() -> f32 {
        Periodogram::<T, F>::default_resolution()
    }

    #[inline]
    pub fn default_max_freq_factor() -> f32 {
        Periodogram::<T, F>::default_max_freq_factor()
    }

    /// Set the normalisation strategy used when combining per-band powers.
    pub fn set_normalization(
        &mut self,
        normalization: MultiColorPeriodogramNormalisation,
    ) -> &mut Self {
        self.normalization = normalization;
        self
    }

    /// Set frequency resolution.
    pub fn set_freq_resolution(&mut self, resolution: f32) -> Option<&mut Self> {
        self.monochrome.set_freq_resolution(resolution)?;
        Some(self)
    }

    /// Set the maximum-frequency multiplier.
    pub fn set_max_freq_factor(&mut self, max_freq_factor: f32) -> Option<&mut Self> {
        self.monochrome.set_max_freq_factor(max_freq_factor)?;
        Some(self)
    }

    /// Set the Nyquist frequency strategy.
    pub fn set_nyquist(&mut self, nyquist: impl Into<NyquistFreq>) -> Option<&mut Self> {
        self.monochrome.set_nyquist(nyquist)?;
        Some(self)
    }

    /// Set a fixed frequency grid.
    pub fn set_freq_grid(
        &mut self,
        freq_grid: impl Into<crate::periodogram::FreqGrid<T>>,
    ) -> &mut Self {
        self.monochrome.set_freq_grid(freq_grid);
        self
    }

    /// Set a new frequency-grid strategy.
    pub fn set_freq_grid_strategy(
        &mut self,
        freq_grid_strategy: impl Into<FreqGridStrategy<T>>,
    ) -> &mut Self {
        self.monochrome.set_freq_grid_strategy(freq_grid_strategy);
        self
    }

    /// Set the periodogram power algorithm (FFT or direct).
    pub fn set_periodogram_algorithm(
        &mut self,
        periodogram_power: PeriodogramPower<T>,
    ) -> &mut Self {
        self.monochrome.set_periodogram_algorithm(periodogram_power);
        self
    }

    /// Add a feature to extract from the combined periodogram spectrum.
    pub fn add_spectrum_feature(&mut self, feature: F) -> &mut Self {
        self.monochrome.add_spectrum_feature(feature);
        self.rebuild_properties();
        self
    }

    /// Register the passbands on which phase features will be evaluated.
    ///
    /// Must be called before [Self::add_phase_feature].
    ///
    /// # Panics
    /// Panics if phase features have already been added.
    pub fn set_phase_bands(&mut self, bands: impl Into<Vec<P>>) -> &mut Self {
        assert!(
            self.phase_extractor.get_features().is_empty(),
            "set_phase_bands must be called before add_phase_feature"
        );
        self.phase_bands = bands.into();
        self
    }

    /// Add a feature to extract from each band's phase-folded light curve.
    ///
    /// The light curve of each registered passband is folded at the best period
    /// from the combined periodogram, with phase 0 at the magnitude minimum.
    /// Feature names are prefixed with `period_folded_{band}_`.
    ///
    /// # Panics
    /// Panics if [Self::set_phase_bands] has not been called first.
    pub fn add_phase_feature(&mut self, feature: F) -> &mut Self {
        assert!(
            !self.phase_bands.is_empty(),
            "call set_phase_bands before add_phase_feature"
        );
        self.phase_extractor.add_feature(feature);
        self.rebuild_properties();
        self
    }

    fn rebuild_properties(&mut self) {
        let monochrome_info = self.monochrome.get_info();
        let n_bands = self.phase_bands.len();
        // FeatureExtractor::add_feature does not update its cached info.size, so
        // we must compute the phase feature size by summing over features directly.
        let phase_feature_size: usize = self
            .phase_extractor
            .get_features()
            .iter()
            .map(|f| f.size_hint())
            .sum();
        let phase_size = n_bands * phase_feature_size;
        let phase_min_ts = if n_bands > 0 {
            self.phase_extractor
                .get_features()
                .iter()
                .map(|f| f.min_ts_length())
                .max()
                .unwrap_or(0)
        } else {
            0
        };

        let mut names: Vec<String> = self
            .monochrome
            .get_names()
            .iter()
            .map(|s| s.to_string())
            .collect();
        let mut descriptions: Vec<String> = self
            .monochrome
            .get_descriptions()
            .iter()
            .map(|s| s.to_string())
            .collect();
        for band in &self.phase_bands {
            for name in self.phase_extractor.get_names() {
                names.push(format!("period_folded_{}_{}", band.name(), name));
            }
            for desc in self.phase_extractor.get_descriptions() {
                descriptions.push(format!(
                    "{} (light curve of {} band folded at best period)",
                    desc,
                    band.name()
                ));
            }
        }

        *self.properties = EvaluatorProperties {
            info: EvaluatorInfo {
                size: monochrome_info.size + phase_size,
                min_ts_length: monochrome_info.min_ts_length.max(phase_min_ts),
                ..*monochrome_info
            },
            names,
            descriptions,
        };
    }
}

// ── EvaluatorInfoTrait / FeatureNamesDescriptionsTrait ────────────────────────

impl<P, T, F> EvaluatorInfoTrait for MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}

impl<P, T, F> FeatureNamesDescriptionsTrait for MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait,
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

// ── Power helpers ─────────────────────────────────────────────────────────────

impl<P, T, F> MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    /// Build a [Periodogram] from the observation times of the specified passbands only.
    fn periodogram_from_mcts<'slf>(
        &'slf self,
        mcts: &mut MultiColorTimeSeries<'_, P, T>,
    ) -> Result<periodogram::Periodogram<'slf, T>, MultiColorEvaluatorError> {
        let PassbandSet(passband_set) = &self.passband_set;
        let t_filtered: Vec<T> = mcts.with_mapping_mut(|mapping| {
            mapping
                .iter_matched_passbands_mut(passband_set.iter())
                .filter_map(|(_, ts)| ts)
                .flat_map(|ts| ts.t.as_slice().to_vec())
                .collect()
        });
        self.monochrome
            .periodogram_from_t(&t_filtered)
            .map_err(|e| MultiColorEvaluatorError::UnderlyingEvaluatorError(e.into()))
    }

    fn power_from_periodogram<'slf, 'a, 'mcts>(
        &'slf self,
        p: &periodogram::Periodogram<T>,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<Array1<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        let PassbandSet(passband_set) = &self.passband_set;
        mcts.with_mapping_mut(|mapping| -> Result<Array1<T>, MultiColorEvaluatorError> {
            let ts_weights = {
                let mut a: Array1<_> = match self.normalization {
                    MultiColorPeriodogramNormalisation::Count => mapping
                        .iter_matched_passbands_mut(passband_set.iter())
                        .map(|(_, ts)| {
                            ts.expect("passband must be present after check_mcts")
                                .lenf()
                        })
                        .collect(),
                    MultiColorPeriodogramNormalisation::Chi2 => mapping
                        .iter_matched_passbands_mut(passband_set.iter())
                        .map(|(_, ts)| {
                            ts.expect("passband must be present after check_mcts")
                                .get_m_chi2()
                        })
                        .collect(),
                };
                let norm = a.sum();
                if norm.is_zero() {
                    return Err(match self.normalization {
                        MultiColorPeriodogramNormalisation::Count => {
                            MultiColorEvaluatorError::all_time_series_short(
                                mapping,
                                self.min_ts_length(),
                            )
                        }
                        MultiColorPeriodogramNormalisation::Chi2 => {
                            MultiColorEvaluatorError::AllTimeSeriesAreFlat
                        }
                    });
                }
                a /= norm;
                a
            };
            let combined = mapping
                .iter_matched_passbands_mut(passband_set.iter())
                .filter_map(|(_, ts)| ts)
                .zip(ts_weights.iter())
                .filter(|(ts, _ts_weight)| self.monochrome.check_ts_length(ts).is_ok())
                .map(|(ts, &ts_weight)| {
                    let mut power = Array1::from_vec(p.power(ts));
                    power *= ts_weight;
                    power
                })
                .reduce(|mut acc, power| {
                    acc += &power;
                    acc
                });
            combined.ok_or_else(|| {
                MultiColorEvaluatorError::all_time_series_short(mapping, self.min_ts_length())
            })
        })
    }

    /// Compute the combined multi-band Lomb-Scargle power spectrum.
    ///
    /// The frequency grid is derived from all observation times across the
    /// specified passbands. Returns a 1-D array of power values, one per
    /// frequency grid point.
    pub fn power<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<Array1<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        let p = self.periodogram_from_mcts(mcts)?;
        self.power_from_periodogram(&p, mcts)
    }

    /// Compute the combined multi-band power spectrum together with the
    /// frequency grid.
    ///
    /// Returns `(frequencies, powers)` as a pair of 1-D arrays. The
    /// frequencies are in the same units as the reciprocal of the time axis
    /// (rad per time unit when using the default angular-frequency convention).
    pub fn freq_power<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<(Array1<T>, Array1<T>), MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        let p = self.periodogram_from_mcts(mcts)?;
        let power = self.power_from_periodogram(&p, mcts)?;
        let freq = (0..power.len()).map(|i| p.freq(i)).collect();
        Ok((freq, power))
    }

    fn transform_mcts_to_ts<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<TmArrays<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        let (freq, power) = self.freq_power(mcts)?;
        Ok(TmArrays { t: freq, m: power })
    }
}

// ── PassbandSetTrait / MultiColorEvaluator ────────────────────────────────────

impl<P, T, F> MultiColorPassbandSetTrait<P> for MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn get_passband_set(&self) -> &PassbandSet<P> {
        &self.passband_set
    }
}

impl<P, T, F> MultiColorEvaluator<P, T> for MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn eval_multicolor_no_mcts_check<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        let arrays = self.transform_mcts_to_ts(mcts)?;
        let mut ts = arrays.ts();
        let mut result = self
            .monochrome
            .spectrum_extractor_ref()
            .eval(&mut ts)
            .map_err(MultiColorEvaluatorError::from)?;

        if !self.phase_bands.is_empty() && !self.phase_extractor.get_features().is_empty() {
            let best_period = result[0];
            if !best_period.is_finite() || best_period <= T::zero() {
                return Err(MultiColorEvaluatorError::UnderlyingEvaluatorError(
                    crate::EvaluatorError::ZeroDivision(
                        "best period from periodogram is not positive, cannot phase-fold",
                    ),
                ));
            }
            let actual_passbands: std::collections::BTreeSet<String> =
                mcts.passbands().map(|p| p.name().into()).collect();
            let desired_passbands: std::collections::BTreeSet<String> =
                self.phase_bands.iter().map(|p| p.name().into()).collect();
            mcts.with_mapping_mut(|mapping| -> Result<(), MultiColorEvaluatorError> {
                for (band, maybe_ts) in mapping.iter_matched_passbands_mut(self.phase_bands.iter())
                {
                    let band_ts =
                        maybe_ts.ok_or_else(|| MultiColorEvaluatorError::WrongPassbandsError {
                            actual: actual_passbands.clone(),
                            desired: desired_passbands.clone(),
                        })?;
                    let phase_arrays = phase_fold_or_compute(
                        band_ts,
                        best_period,
                        self.phase_extractor.is_t_required(),
                        self.phase_extractor.is_sorting_required(),
                    );
                    match phase_arrays {
                        Some(arrays) => {
                            let mut phase_ts = arrays.ts();
                            result.extend(
                                eval_phase_ts(&self.phase_extractor, &mut phase_ts).map_err(
                                    |e| MultiColorEvaluatorError::MonochromeEvaluatorError {
                                        error: e,
                                        passband: band.name().to_string(),
                                    },
                                )?,
                            );
                        }
                        None => {
                            result.extend(self.phase_extractor.eval(band_ts).map_err(|e| {
                                MultiColorEvaluatorError::MonochromeEvaluatorError {
                                    error: e,
                                    passband: band.name().to_string(),
                                }
                            })?);
                        }
                    }
                }
                Ok(())
            })?;
        }
        Ok(result)
    }

    fn eval_or_fill_multicolor<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
        fill_value: T,
    ) -> Result<Vec<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        let arrays = match self.transform_mcts_to_ts(mcts) {
            Ok(arrays) => arrays,
            Err(_) => return Ok(vec![fill_value; self.size_hint()]),
        };
        let mut ts = arrays.ts();
        let mut result = self
            .monochrome
            .spectrum_extractor_ref()
            .eval_or_fill(&mut ts, fill_value);

        if !self.phase_bands.is_empty() && !self.phase_extractor.get_features().is_empty() {
            let best_period = result[0];
            let phase_feature_size: usize = self
                .phase_extractor
                .get_features()
                .iter()
                .map(|f| f.size_hint())
                .sum();
            if best_period.is_finite() && best_period > T::zero() {
                mcts.with_mapping_mut(|mapping| {
                    for (_band, maybe_ts) in
                        mapping.iter_matched_passbands_mut(self.phase_bands.iter())
                    {
                        if let Some(band_ts) = maybe_ts {
                            let phase_arrays = phase_fold_or_compute(
                                band_ts,
                                best_period,
                                self.phase_extractor.is_t_required(),
                                self.phase_extractor.is_sorting_required(),
                            );
                            match phase_arrays {
                                Some(arrays) => {
                                    let mut phase_ts = arrays.ts();
                                    result.extend(eval_phase_ts_or_fill(
                                        &self.phase_extractor,
                                        &mut phase_ts,
                                        fill_value,
                                    ));
                                }
                                None => result
                                    .extend(self.phase_extractor.eval_or_fill(band_ts, fill_value)),
                            }
                        } else {
                            result.extend(vec![fill_value; phase_feature_size]);
                        }
                    }
                });
            } else {
                let phase_total = self.phase_bands.len() * phase_feature_size;
                result.extend(vec![fill_value; phase_total]);
            }
        }
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Feature, MultiColorTimeSeries, StringPassband};
    use std::collections::BTreeSet;

    type McPeriodogram = MultiColorPeriodogram<StringPassband, f64, Feature<f64>>;

    fn passbands_from_array(bands: &ndarray::Array1<String>) -> BTreeSet<StringPassband> {
        bands
            .iter()
            .map(|b| StringPassband::from(b.as_str()))
            .collect()
    }

    fn check_finite_with_norm(norm: MultiColorPeriodogramNormalisation) {
        for (name, mclc) in light_curve_feature_test_util::RRLYR_LIGHT_CURVES_MAG_F64.iter() {
            let (t, m, w, bands) = mclc.clone().into_quadruple();
            let passbands: Vec<StringPassband> = bands
                .iter()
                .map(|b| StringPassband::from(b.as_str()))
                .collect();
            let unique_passbands = passbands_from_array(&bands);
            let eval = McPeriodogram::new(1, norm.clone(), unique_passbands);
            let mut mcts = MultiColorTimeSeries::from_flat(
                t.into_raw_vec_and_offset().0,
                m.into_raw_vec_and_offset().0,
                w.into_raw_vec_and_offset().0,
                passbands,
            );
            let result = eval.eval_multicolor(&mut mcts);
            assert!(result.is_ok(), "{name}: {result:?}");
            for (value, feature_name) in result.unwrap().into_iter().zip(eval.get_names()) {
                assert!(value.is_finite(), "{name}: {feature_name} is not finite");
            }
        }
    }

    #[test]
    fn check_values_finite_count_norm() {
        check_finite_with_norm(MultiColorPeriodogramNormalisation::Count);
    }

    #[test]
    fn check_values_finite_chi2_norm() {
        check_finite_with_norm(MultiColorPeriodogramNormalisation::Chi2);
    }

    #[test]
    fn check_period_recovery() {
        use crate::{LinearFreqGrid, PeriodogramPowerDirect};

        let n_tested = 10;

        let baseline = light_curve_feature_test_util::RR_LYRAE_F64
            .iter()
            .take(n_tested)
            .map(|rrlyr| {
                let t = rrlyr.light_curve.clone().into_quadruple().0;
                let tmax = t.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let tmin = t.iter().cloned().fold(f64::INFINITY, f64::min);
                tmax - tmin
            })
            .fold(0.0_f64, f64::max);

        let resolution = 10.0_f64;
        let min_freq = std::f64::consts::TAU / 1.0;
        let max_freq = std::f64::consts::TAU / 0.2;
        let step = std::f64::consts::TAU / (resolution * baseline);
        let size = ((max_freq - min_freq) / step).ceil() as usize + 1;
        let grid = LinearFreqGrid::new(min_freq, step, size);

        let tolerance = 0.01;

        let mut n_recovered = 0usize;
        for rrlyr in light_curve_feature_test_util::RR_LYRAE_F64
            .iter()
            .take(n_tested)
        {
            let (t, m, w, bands) = rrlyr.light_curve.clone().into_quadruple();
            let passbands: Vec<StringPassband> = bands
                .iter()
                .map(|b| StringPassband::from(b.as_str()))
                .collect();
            let unique_passbands = passbands_from_array(&bands);
            // Use 2 peaks: for short-period RRc stars the 2-day alias may rank higher than
            // the true period, but the true period appears as the 2nd-highest peak.
            let mut eval = McPeriodogram::new(
                2,
                MultiColorPeriodogramNormalisation::Count,
                unique_passbands,
            );
            eval.set_freq_grid(grid.clone());
            eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());
            let mut mcts = MultiColorTimeSeries::from_flat(
                t.into_raw_vec_and_offset().0,
                m.into_raw_vec_and_offset().0,
                w.into_raw_vec_and_offset().0,
                passbands,
            );
            let result = eval.eval_multicolor(&mut mcts).unwrap();
            let known = rrlyr.period;
            let recovered_periods = [result[0], result[2]];
            if recovered_periods
                .iter()
                .any(|&r| (r - known).abs() / known < tolerance)
            {
                n_recovered += 1;
            }
        }
        assert!(
            n_recovered == n_tested,
            "Period not in top-2 peaks: {n_recovered}/{n_tested}"
        );
    }

    fn make_gr_passbands() -> BTreeSet<StringPassband> {
        ["g", "r"]
            .iter()
            .map(|&s| StringPassband::from(s))
            .collect()
    }

    #[test]
    fn serde_json_roundtrip() {
        let eval = McPeriodogram::new(
            1,
            MultiColorPeriodogramNormalisation::Count,
            make_gr_passbands(),
        );
        let json = serde_json::to_string(&eval).unwrap();
        let eval2: McPeriodogram = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string(&eval2).unwrap());
    }

    #[test]
    fn serde_json_chi2_norm() {
        let eval = McPeriodogram::new(
            2,
            MultiColorPeriodogramNormalisation::Chi2,
            make_gr_passbands(),
        );
        let json = serde_json::to_string(&eval).unwrap();
        let eval2: McPeriodogram = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string(&eval2).unwrap());
    }

    #[test]
    fn serde_json_direct_algorithm() {
        use crate::PeriodogramPowerDirect;
        let mut eval = McPeriodogram::new(
            1,
            MultiColorPeriodogramNormalisation::Count,
            make_gr_passbands(),
        );
        eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());
        let json = serde_json::to_string(&eval).unwrap();
        let eval2: McPeriodogram = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string(&eval2).unwrap());
    }

    #[test]
    fn serde_json_with_phase_features() {
        use crate::features::LaflerKinmanStringLength;
        let mut eval = McPeriodogram::new(
            1,
            MultiColorPeriodogramNormalisation::Count,
            make_gr_passbands(),
        );
        eval.set_phase_bands(vec![StringPassband::from("g"), StringPassband::from("r")]);
        eval.add_phase_feature(LaflerKinmanStringLength::new().into());
        let json = serde_json::to_string(&eval).unwrap();
        let eval2: McPeriodogram = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string(&eval2).unwrap());
        // Names and size survive the round-trip
        assert_eq!(eval.get_names(), eval2.get_names());
        assert_eq!(eval.size_hint(), eval2.size_hint());
    }

    #[test]
    fn phase_feature_names_and_size() {
        use crate::features::LaflerKinmanStringLength;

        let mut eval = McPeriodogram::new(
            1,
            MultiColorPeriodogramNormalisation::Count,
            make_gr_passbands(),
        );
        eval.set_phase_bands(vec![StringPassband::from("g"), StringPassband::from("r")]);
        eval.add_phase_feature(LaflerKinmanStringLength::new().into());

        let names = eval.get_names();
        // spectrum: multicolor_periodogram_period_0, multicolor_periodogram_period_s_to_n_0
        assert_eq!(names[0], "multicolor_periodogram_period_0");
        // phase per band
        assert!(
            names.contains(&"period_folded_g_lafler_kinman_string_length"),
            "names = {names:?}"
        );
        assert!(
            names.contains(&"period_folded_r_lafler_kinman_string_length"),
            "names = {names:?}"
        );
        // total size = 2 spectrum + 2 phase (1 per band)
        assert_eq!(eval.size_hint(), 4);
    }

    #[test]
    fn chi2_norm_unity_weights_differ_from_count() {
        use crate::data::TimeSeries;
        use crate::{LinearFreqGrid, PeriodogramPowerDirect};

        let n = 20usize;
        let period = 0.3_f64;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let m_g: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti / period).sin())
            .collect();
        let m_r: Vec<f64> = vec![1.0; n];

        let make_mcts = || {
            let mut map = std::collections::BTreeMap::new();
            map.insert(
                StringPassband::from("g"),
                TimeSeries::new_without_weight(t.as_slice(), m_g.as_slice()),
            );
            map.insert(
                StringPassband::from("r"),
                TimeSeries::new_without_weight(t.as_slice(), m_r.as_slice()),
            );
            MultiColorTimeSeries::from_map(map)
        };

        let make_eval = |norm| {
            let mut eval = McPeriodogram::new(1, norm, make_gr_passbands());
            eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());
            eval.set_freq_grid(LinearFreqGrid::new(1.0 / 2.0, 1.0 / 0.1, 200));
            eval
        };

        let eval_count = make_eval(MultiColorPeriodogramNormalisation::Count);
        let eval_chi2 = make_eval(MultiColorPeriodogramNormalisation::Chi2);

        let mut mcts_count = make_mcts();
        let mut mcts_chi2 = make_mcts();

        let power_count = eval_count.power(&mut mcts_count).unwrap();
        let power_chi2 = eval_chi2.power(&mut mcts_chi2).unwrap();

        assert!(
            power_count.iter().all(|v| v.is_finite()),
            "Count: non-finite power"
        );
        assert!(
            power_chi2.iter().all(|v| v.is_finite()),
            "Chi2: non-finite power"
        );

        for (&pc, &pchi2) in power_count.iter().zip(power_chi2.iter()) {
            let ratio = pchi2 / pc;
            assert!(
                (ratio - 2.0).abs() < 1e-10,
                "Expected Chi2/Count power ratio ≈ 2, got {ratio}"
            );
        }

        let m_flat: Vec<f64> = vec![1.0; n];
        let eval_chi2_flat = make_eval(MultiColorPeriodogramNormalisation::Chi2);
        let mut mcts_flat = {
            let mut map = std::collections::BTreeMap::new();
            map.insert(
                StringPassband::from("g"),
                TimeSeries::new_without_weight(t.as_slice(), m_flat.as_slice()),
            );
            map.insert(
                StringPassband::from("r"),
                TimeSeries::new_without_weight(t.as_slice(), m_flat.as_slice()),
            );
            MultiColorTimeSeries::from_map(map)
        };
        assert!(
            eval_chi2_flat.eval_multicolor(&mut mcts_flat).is_err(),
            "Chi2 with all-flat bands should return an error"
        );
    }

    fn make_g_passband() -> BTreeSet<StringPassband> {
        [StringPassband::from("g")].into_iter().collect()
    }

    // Case 1: !t_required && !sorting_required — raw band_ts evaluated, no phase folding.
    // Amplitude only needs magnitudes; phase_fold_or_compute returns None and the extractor
    // is called directly on the original (unfolded) band time series.
    #[test]
    fn phase_feature_case1_no_t_no_sort_amplitude() {
        use crate::PeriodogramPowerDirect;
        use crate::data::TimeSeries;
        use crate::features::Amplitude;

        let period = 0.3_f64;
        let n = 50usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let m: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti / period).sin())
            .collect();

        let mut eval = McPeriodogram::new(
            1,
            MultiColorPeriodogramNormalisation::Count,
            make_g_passband(),
        );
        eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());
        eval.set_phase_bands(vec![StringPassband::from("g")]);
        eval.add_phase_feature(Amplitude::default().into());

        let mut map = std::collections::BTreeMap::new();
        map.insert(
            StringPassband::from("g"),
            TimeSeries::new_without_weight(t.as_slice(), m.as_slice()),
        );
        let mut mcts = MultiColorTimeSeries::from_map(map);

        let result = eval.eval_multicolor(&mut mcts).unwrap();
        assert_eq!(result.len(), eval.size_hint());
        // Amplitude = (max - min) / 2; must equal the full-band value since no folding
        let expected_amp = (m.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - m.iter().cloned().fold(f64::INFINITY, f64::min))
            / 2.0;
        assert!(
            (result[2] - expected_amp).abs() < 1e-12,
            "amplitude {}, expected {}",
            result[2],
            expected_amp,
        );
    }

    // Case 2: t_required && !sorting_required — phase_compute_ts path (phases as time, no sort).
    // TimeMean returns the mean of its "time" array; when phases are used as time the result
    // must lie in (0, 1) and differ from the mean of the original timestamps.
    #[test]
    fn phase_feature_case2_t_required_no_sort_time_mean() {
        use crate::PeriodogramPowerDirect;
        use crate::data::TimeSeries;
        use crate::features::TimeMean;

        let period = 0.3_f64;
        let n = 50usize;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let m: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti / period).sin())
            .collect();

        let mut eval = McPeriodogram::new(
            1,
            MultiColorPeriodogramNormalisation::Count,
            make_g_passband(),
        );
        eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());
        eval.set_phase_bands(vec![StringPassband::from("g")]);
        eval.add_phase_feature(TimeMean::default().into());

        let mut map = std::collections::BTreeMap::new();
        map.insert(
            StringPassband::from("g"),
            TimeSeries::new_without_weight(t.as_slice(), m.as_slice()),
        );
        let mut mcts = MultiColorTimeSeries::from_map(map);

        let result = eval.eval_multicolor(&mut mcts).unwrap();
        assert_eq!(result.len(), eval.size_hint());
        // Mean of phases must lie in (0, 1)
        let phase_mean = result[2];
        assert!(
            phase_mean > 0.0 && phase_mean < 1.0,
            "mean phase {phase_mean} not in (0, 1)",
        );
        // Must differ from mean of the original timestamps
        let mean_t = t.iter().sum::<f64>() / t.len() as f64;
        assert!(
            (phase_mean - mean_t).abs() > 1e-6,
            "mean phase {phase_mean} suspiciously equal to mean t {mean_t}",
        );
    }

    // Case 4: sorting_required && t_required, near-duplicate phases → Bins(1e-6) applied.
    // t=[0,1,2,3,4] with period=1 → all phases collapse to 0.0; binning merges them into
    // one point.  MaximumSlope requires ≥2 points, so eval_or_fill returns the fill value.
    #[test]
    fn phase_feature_case4_near_duplicate_phases_fill_not_panic() {
        use crate::PeriodogramPowerDirect;
        use crate::data::TimeSeries;
        use crate::features::MaximumSlope;

        let t = vec![0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let m = vec![1.0_f64, 1.5, 0.5, 1.2, 0.8];

        let mut eval = McPeriodogram::new(
            1,
            MultiColorPeriodogramNormalisation::Count,
            make_g_passband(),
        );
        eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());
        // Fixed grid: single frequency whose period equals 1.0 → all phases collapse to 0.0
        eval.set_freq_grid(crate::periodogram::FreqGrid::linear(
            std::f64::consts::TAU,
            0.1,
            1,
        ));
        eval.set_phase_bands(vec![StringPassband::from("g")]);
        eval.add_phase_feature(MaximumSlope::default().into());

        let mut map = std::collections::BTreeMap::new();
        map.insert(
            StringPassband::from("g"),
            TimeSeries::new_without_weight(t.as_slice(), m.as_slice()),
        );
        let mut mcts = MultiColorTimeSeries::from_map(map);

        let fill = f64::NAN;
        let result = eval.eval_or_fill_multicolor(&mut mcts, fill).unwrap();
        // result = [period_0, snr_0, period_folded_g_maximum_slope]
        assert_eq!(result.len(), eval.size_hint());
        // Phase feature must be fill (only 1 bin after merge, too few for MaximumSlope)
        assert!(result[2].is_nan(), "expected fill NaN, got {}", result[2]);
    }

    #[test]
    fn single_band_phase_feature_sine_recovery() {
        use crate::PeriodogramPowerDirect;
        use crate::data::TimeSeries;
        use crate::features::LaflerKinmanStringLength;
        use rand::prelude::*;

        let period = 0.17_f64;
        let n = 200usize;
        let mut rng = StdRng::seed_from_u64(0);
        let mut t: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
        t.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let m: Vec<f64> = t
            .iter()
            .map(|&ti| 3.0 * (2.0 * std::f64::consts::PI * ti / period).sin() + 4.0)
            .collect();

        let mut eval = McPeriodogram::new(
            1,
            MultiColorPeriodogramNormalisation::Count,
            make_g_passband(),
        );
        eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());
        eval.set_phase_bands(vec![StringPassband::from("g")]);
        eval.add_phase_feature(LaflerKinmanStringLength::new().into());

        let mut map = std::collections::BTreeMap::new();
        map.insert(
            StringPassband::from("g"),
            TimeSeries::new_without_weight(t.as_slice(), m.as_slice()),
        );
        let mut mcts = MultiColorTimeSeries::from_map(map);

        let result = eval.eval_multicolor(&mut mcts).unwrap();
        // result = [period_0, snr_0, period_folded_g_lafler_kinman_string_length]
        assert_eq!(result.len(), 3);
        let recovered_period = result[0];
        assert!(
            (recovered_period - period).abs() / period < 0.05,
            "period recovery failed: got {recovered_period}, expected {period}"
        );
        let theta = result[2];
        assert!(
            theta < 0.01,
            "phase-folded LaflerKinmanStringLength = {theta}, expected < 0.01 for smooth curve"
        );
    }

    /// Verify that MultiColorPeriodogram ignores bands not in its passband set.
    ///
    /// Setup: three bands g, r, i with a sinusoidal signal in g, constant in r, constant in i.
    /// An evaluator configured for only {g, r} must produce the same result regardless of
    /// whether the i-band data is present in the MultiColorTimeSeries.
    #[test]
    fn subsamples_to_requested_passbands() {
        use crate::data::TimeSeries;
        use crate::{LinearFreqGrid, PeriodogramPowerDirect};

        let n = 20usize;
        let period = 0.3_f64;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        let m_g: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti / period).sin())
            .collect();
        let m_r: Vec<f64> = vec![0.0; n];
        let m_i: Vec<f64> = vec![99.0; n]; // large constant — should be ignored

        let make_eval = || {
            let passbands: BTreeSet<StringPassband> = ["g", "r"]
                .iter()
                .map(|&s| StringPassband::from(s))
                .collect();
            let mut eval =
                McPeriodogram::new(1, MultiColorPeriodogramNormalisation::Count, passbands);
            eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());
            eval.set_freq_grid(LinearFreqGrid::new(1.0 / 2.0, 1.0 / 0.1, 200));
            eval
        };

        let eval_gr = make_eval();
        let eval_gri = make_eval();

        // mcts with only g and r
        let mut mcts_gr = {
            let mut map = std::collections::BTreeMap::new();
            map.insert(
                StringPassband::from("g"),
                TimeSeries::new_without_weight(t.as_slice(), m_g.as_slice()),
            );
            map.insert(
                StringPassband::from("r"),
                TimeSeries::new_without_weight(t.as_slice(), m_r.as_slice()),
            );
            MultiColorTimeSeries::from_map(map)
        };

        // mcts with g, r, and extra i band — evaluator should ignore i
        let mut mcts_gri = {
            let mut map = std::collections::BTreeMap::new();
            map.insert(
                StringPassband::from("g"),
                TimeSeries::new_without_weight(t.as_slice(), m_g.as_slice()),
            );
            map.insert(
                StringPassband::from("r"),
                TimeSeries::new_without_weight(t.as_slice(), m_r.as_slice()),
            );
            map.insert(
                StringPassband::from("i"),
                TimeSeries::new_without_weight(t.as_slice(), m_i.as_slice()),
            );
            MultiColorTimeSeries::from_map(map)
        };

        let result_gr = eval_gr.eval_multicolor(&mut mcts_gr).unwrap();
        let result_gri = eval_gri.eval_multicolor(&mut mcts_gri).unwrap();
        assert_eq!(
            result_gr, result_gri,
            "Extra i band must not affect the result"
        );
    }
}
