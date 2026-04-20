use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::TmArrays;
use crate::evaluator::{
    EvaluatorInfo, EvaluatorInfoTrait, FeatureEvaluator, FeatureNamesDescriptionsTrait, OwnedArrays,
};
use crate::features::{Periodogram, PeriodogramPeaks};
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
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound(
    serialize = "P: PassbandTrait, T: Float, F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>, <F as TryInto<PeriodogramPeaks>>::Error: Debug",
    deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float, F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>, <F as TryInto<PeriodogramPeaks>>::Error: Debug",
))]
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
}

impl<P, T, F> MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks>,
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
        Self {
            monochrome,
            normalization,
            passband_set: PassbandSet::FixedSet(passbands.into_iter().collect()),
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
    ///
    /// A larger resolution allows peak periods to be found with better
    /// precision.
    ///
    /// Returns [`None`] if the underlying frequency-grid strategy is
    /// [`FreqGridStrategy::Fixed`]; otherwise applies the change and returns
    /// [`Some`].
    pub fn set_freq_resolution(&mut self, resolution: f32) -> Option<&mut Self> {
        self.monochrome.set_freq_resolution(resolution)?;
        Some(self)
    }

    /// Set the maximum-frequency multiplier.
    ///
    /// The maximum frequency is the Nyquist frequency multiplied by this
    /// factor. A larger factor lets [`PeriodogramPowerFft`] find higher
    /// frequencies more precisely, but also risks spurious peaks.
    ///
    /// Returns [`None`] if the underlying frequency-grid strategy is
    /// [`FreqGridStrategy::Fixed`]; otherwise applies the change and returns
    /// [`Some`].
    pub fn set_max_freq_factor(&mut self, max_freq_factor: f32) -> Option<&mut Self> {
        self.monochrome.set_max_freq_factor(max_freq_factor)?;
        Some(self)
    }

    /// Set the Nyquist frequency strategy.
    ///
    /// Returns [`None`] if the underlying frequency-grid strategy is
    /// [`FreqGridStrategy::Fixed`]; otherwise applies the change and returns
    /// [`Some`].
    pub fn set_nyquist(&mut self, nyquist: impl Into<NyquistFreq>) -> Option<&mut Self> {
        self.monochrome.set_nyquist(nyquist)?;
        Some(self)
    }

    /// Set a fixed frequency grid, overriding the dynamic Nyquist-based grid.
    ///
    /// Use [`crate::LinearFreqGrid`] with [`crate::PeriodogramPowerDirect`] to
    /// search a custom frequency range, e.g. to avoid ground-based daily
    /// aliases.
    pub fn set_freq_grid(
        &mut self,
        freq_grid: impl Into<crate::periodogram::FreqGrid<T>>,
    ) -> &mut Self {
        self.monochrome.set_freq_grid(freq_grid);
        self
    }

    /// Set a new frequency-grid strategy (either dynamic or fixed).
    ///
    /// This is the most general way to control the frequency grid; see
    /// [`FreqGridStrategy`] for the available variants.
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

    /// Extend the set of features extracted from the periodogram.
    pub fn add_feature(&mut self, feature: F) -> &mut Self {
        self.monochrome.add_feature(feature);
        self
    }
}

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
        let PassbandSet::FixedSet(passband_set) = &self.passband_set;
        let t_filtered: Vec<T> = {
            let mapping = mcts.mapping_mut();
            mapping
                .iter_matched_passbands_mut(passband_set.iter())
                .filter_map(|(_, ts)| ts)
                .flat_map(|ts| ts.t.as_slice().to_vec())
                .collect()
        };
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
        let PassbandSet::FixedSet(passband_set) = &self.passband_set;
        let mapping = mcts.mapping_mut();
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
}

impl<P, T, F> MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
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

impl<P, T, F> EvaluatorInfoTrait for MultiColorPeriodogram<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn get_info(&self) -> &EvaluatorInfo {
        self.monochrome.get_info()
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
        self.monochrome.get_names()
    }

    fn get_descriptions(&self) -> Vec<&str> {
        self.monochrome.get_descriptions()
    }
}

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
        self.monochrome
            .feature_extractor_ref()
            .eval(&mut ts)
            .map_err(From::from)
    }

    /// Returns vector of feature values and fill invalid components with given value
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
        Ok(self
            .monochrome
            .feature_extractor_ref()
            .eval_or_fill(&mut ts, fill_value))
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

        // Compute the maximum time baseline across the test light curves so the
        // frequency step adapts to the actual data rather than a hard-coded constant.
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

        // Use Direct periodogram with a linear frequency grid covering RRLyrae periods
        // (0.2-1.0 days), skipping ground-based 1-day aliases.
        let resolution = 10.0_f64;
        let min_freq = std::f64::consts::TAU / 1.0; // max period = 1 day
        let max_freq = std::f64::consts::TAU / 0.2; // min period = 0.2 days
        let step = std::f64::consts::TAU / (resolution * baseline);
        let size = ((max_freq - min_freq) / step).ceil() as usize + 1;
        let grid = LinearFreqGrid::new(min_freq, step, size);

        let tolerance = 0.01; // 1%

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
            // result = [period_0, snr_0, period_1, snr_1, ...]
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

    /// Verify Chi2 normalization with unity weights weights by magnitude variance,
    /// which is different from Count normalization (which weights by observation count).
    ///
    /// Setup: two bands with the same number of observations.
    ///   - Band "g": sinusoidal — non-zero variance, gets all the Chi2 weight.
    ///   - Band "r": constant  — zero variance, gets weight 0 under Chi2.
    ///
    /// With Count:  power = 0.5·power_g + 0.5·power_r
    /// With Chi2:   power = 1.0·power_g  (r contributes 0)
    ///
    /// Since power_r ≈ 0 for a constant signal, the two combined power spectra
    /// differ by a factor of ~2.  We also verify that all-flat bands are rejected.
    #[test]
    fn chi2_norm_unity_weights_differ_from_count() {
        use crate::data::TimeSeries;
        use crate::{LinearFreqGrid, PeriodogramPowerDirect};

        let n = 20usize;
        let period = 0.3_f64;
        let t: Vec<f64> = (0..n).map(|i| i as f64 * 0.05).collect();
        // Band g: sinusoidal — non-zero variance
        let m_g: Vec<f64> = t
            .iter()
            .map(|&ti| (2.0 * std::f64::consts::PI * ti / period).sin())
            .collect();
        // Band r: constant — zero variance → chi2 = 0 with unity weights
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

        let mut mcts_count = make_mcts();
        let mut mcts_chi2 = make_mcts();

        let eval_count = make_eval(MultiColorPeriodogramNormalisation::Count);
        let eval_chi2 = make_eval(MultiColorPeriodogramNormalisation::Chi2);

        // Compare raw power arrays (not extracted peaks, whose SNR would cancel the scale factor)
        let power_count = eval_count.power(&mut mcts_count).unwrap();
        let power_chi2 = eval_chi2.power(&mut mcts_chi2).unwrap();

        // Both should be finite
        assert!(
            power_count.iter().all(|v| v.is_finite()),
            "Count: non-finite power"
        );
        assert!(
            power_chi2.iter().all(|v| v.is_finite()),
            "Chi2: non-finite power"
        );

        // The constant band r contributes zero LS power, so:
        //   Count power ≈ 0.5 · power_g
        //   Chi2  power ≈ 1.0 · power_g
        // → Chi2 power should be ≈ 2× Count power at every frequency.
        for (&pc, &pchi2) in power_count.iter().zip(power_chi2.iter()) {
            let ratio = pchi2 / pc;
            assert!(
                (ratio - 2.0).abs() < 1e-10,
                "Expected Chi2/Count power ratio ≈ 2, got {ratio}"
            );
        }

        // All-flat bands → Chi2 must return an error
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

        let result_gr = make_eval().eval_multicolor(&mut mcts_gr).unwrap();
        let result_gri = make_eval().eval_multicolor(&mut mcts_gri).unwrap();
        assert_eq!(
            result_gr, result_gri,
            "Extra i band must not affect the result"
        );
    }
}
