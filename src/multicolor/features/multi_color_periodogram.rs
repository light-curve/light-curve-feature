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
/// times. Individual band powers are weighted and summed according to the
/// chosen [`MultiColorPeriodogramNormalisation`] strategy.
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
/// let mut eval = MultiColorPeriodogram::<StringPassband, f64, Feature<f64>>::new(
///     1,
///     MultiColorPeriodogramNormalisation::Count,
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
        let mut mcp = Self::new_with_monochrome(p.monochrome, p.normalization);
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
    /// Create a new multi-colour periodogram.
    ///
    /// `peaks` sets the number of highest-power peaks whose period and
    /// signal-to-noise ratio are returned as features.
    pub fn new(peaks: usize, normalization: MultiColorPeriodogramNormalisation) -> Self {
        let monochrome = Periodogram::with_name_description(
            peaks,
            "multicolor_periodogram",
            "of multi-color periodogram (interpreting frequency as time, power as magnitude)",
        );
        Self::new_with_monochrome(monochrome, normalization)
    }

    fn new_with_monochrome(
        monochrome: Periodogram<T, F>,
        normalization: MultiColorPeriodogramNormalisation,
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
    /// Must be called before [Self::add_phase_feature]. The passband set
    /// switches from `AllAvailable` to `FixedSet` for phase evaluation;
    /// spectrum evaluation still uses all available bands.
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

impl<P, T, F> Default for MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn default() -> Self {
        Self::new(
            Self::default_peaks(),
            MultiColorPeriodogramNormalisation::Count,
        )
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
    fn power_from_periodogram<'slf, 'a, 'mcts>(
        &self,
        p: &periodogram::Periodogram<T>,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<Array1<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        let mapping = mcts.mapping_mut();
        let ts_weights = {
            let mut a: Array1<_> = match self.normalization {
                MultiColorPeriodogramNormalisation::Count => {
                    mapping.values().map(|ts| ts.lenf()).collect()
                }
                MultiColorPeriodogramNormalisation::Chi2 => {
                    mapping.values_mut().map(|ts| ts.get_m_chi2()).collect()
                }
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
            .values_mut()
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
    }

    /// Compute the combined multi-band Lomb-Scargle power spectrum.
    pub fn power<'slf, 'a, 'mcts>(
        &self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<Array1<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        let p = self
            .monochrome
            .periodogram_from_t(mcts.flat_mut().t.as_slice())
            .map_err(|e| MultiColorEvaluatorError::UnderlyingEvaluatorError(e.into()))?;
        self.power_from_periodogram(&p, mcts)
    }

    /// Compute the combined multi-band power spectrum together with the frequency grid.
    pub fn freq_power<'slf, 'a, 'mcts>(
        &self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<(Array1<T>, Array1<T>), MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        let p = self
            .monochrome
            .periodogram_from_t(mcts.flat_mut().t.as_slice())
            .map_err(|e| MultiColorEvaluatorError::UnderlyingEvaluatorError(e.into()))?;
        let power = self.power_from_periodogram(&p, mcts)?;
        let freq = (0..power.len()).map(|i| p.freq(i)).collect();
        Ok((freq, power))
    }

    fn transform_mcts_to_ts<'a>(
        &self,
        mcts: &mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<TmArrays<T>, MultiColorEvaluatorError> {
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
        &PassbandSet::AllAvailable
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
            for (band, maybe_ts) in mcts
                .mapping_mut()
                .iter_matched_passbands_mut(self.phase_bands.iter())
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
                            eval_phase_ts(&self.phase_extractor, &mut phase_ts).map_err(|e| {
                                MultiColorEvaluatorError::MonochromeEvaluatorError {
                                    error: e,
                                    passband: band.name().to_string(),
                                }
                            })?,
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
                for (_band, maybe_ts) in mcts
                    .mapping_mut()
                    .iter_matched_passbands_mut(self.phase_bands.iter())
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

    type McPeriodogram = MultiColorPeriodogram<StringPassband, f64, Feature<f64>>;

    fn check_finite_with_norm(norm: MultiColorPeriodogramNormalisation) {
        let eval = McPeriodogram::new(1, norm);
        for (name, mclc) in light_curve_feature_test_util::RRLYR_LIGHT_CURVES_MAG_F64.iter() {
            let (t, m, w, bands) = mclc.clone().into_quadruple();
            let passbands: Vec<StringPassband> = bands
                .iter()
                .map(|b| StringPassband::from(b.as_str()))
                .collect();
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

        let mut eval = McPeriodogram::new(2, MultiColorPeriodogramNormalisation::Count);

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
        eval.set_freq_grid(grid);
        eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());

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

    #[test]
    fn serde_json_default() {
        let eval = McPeriodogram::default();
        let json = serde_json::to_string(&eval).unwrap();
        let eval2: McPeriodogram = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string(&eval2).unwrap());
    }

    #[test]
    fn serde_json_chi2_norm() {
        let eval = McPeriodogram::new(2, MultiColorPeriodogramNormalisation::Chi2);
        let json = serde_json::to_string(&eval).unwrap();
        let eval2: McPeriodogram = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string(&eval2).unwrap());
    }

    #[test]
    fn serde_json_direct_algorithm() {
        use crate::PeriodogramPowerDirect;
        let mut eval = McPeriodogram::default();
        eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());
        let json = serde_json::to_string(&eval).unwrap();
        let eval2: McPeriodogram = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string(&eval2).unwrap());
    }

    #[test]
    fn phase_feature_names_and_size() {
        use crate::features::LaflerKinman;

        let mut eval = McPeriodogram::new(1, MultiColorPeriodogramNormalisation::Count);
        eval.set_phase_bands(vec![StringPassband::from("g"), StringPassband::from("r")]);
        eval.add_phase_feature(LaflerKinman::new().into());

        let names = eval.get_names();
        // spectrum: multicolor_periodogram_period_0, multicolor_periodogram_period_s_to_n_0
        assert_eq!(names[0], "multicolor_periodogram_period_0");
        // phase per band
        assert!(
            names.contains(&"period_folded_g_lafler_kinman"),
            "names = {names:?}"
        );
        assert!(
            names.contains(&"period_folded_r_lafler_kinman"),
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
            let mut eval = McPeriodogram::new(1, norm);
            eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());
            eval.set_freq_grid(LinearFreqGrid::new(1.0 / 2.0, 1.0 / 0.1, 200));
            eval
        };

        let mut mcts_count = make_mcts();
        let mut mcts_chi2 = make_mcts();

        let eval_count = make_eval(MultiColorPeriodogramNormalisation::Count);
        let eval_chi2 = make_eval(MultiColorPeriodogramNormalisation::Chi2);

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
        let eval_chi2_flat = make_eval(MultiColorPeriodogramNormalisation::Chi2);
        assert!(
            eval_chi2_flat.eval_multicolor(&mut mcts_flat).is_err(),
            "Chi2 with all-flat bands should return an error"
        );
    }

    #[test]
    fn single_band_phase_feature_sine_recovery() {
        use crate::PeriodogramPowerDirect;
        use crate::data::TimeSeries;
        use crate::features::LaflerKinman;
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

        let mut map = std::collections::BTreeMap::new();
        map.insert(
            StringPassband::from("g"),
            TimeSeries::new_without_weight(t.as_slice(), m.as_slice()),
        );
        let mut mcts = MultiColorTimeSeries::from_map(map);

        let mut eval = McPeriodogram::new(1, MultiColorPeriodogramNormalisation::Count);
        eval.set_periodogram_algorithm(PeriodogramPowerDirect.into());
        eval.set_phase_bands(vec![StringPassband::from("g")]);
        eval.add_phase_feature(LaflerKinman::new().into());

        let result = eval.eval_multicolor(&mut mcts).unwrap();
        // result = [period_0, snr_0, period_folded_g_lafler_kinman]
        assert_eq!(result.len(), 3);
        let recovered_period = result[0];
        assert!(
            (recovered_period - period).abs() / period < 0.05,
            "period recovery failed: got {recovered_period}, expected {period}"
        );
        let theta = result[2];
        assert!(
            theta < 0.01,
            "phase-folded LaflerKinman = {theta}, expected < 0.01 for smooth curve"
        );
    }
}
