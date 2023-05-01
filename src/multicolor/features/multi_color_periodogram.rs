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
use crate::periodogram::{self, NyquistFreq, PeriodogramPower};

use ndarray::Array1;
use std::fmt::Debug;

/// Normalisation of periodogram across passbands
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub enum MultiColorPeriodogramNormalisation {
    /// Weight individual periodograms by the number of observations in each passband.
    /// Useful if no weight is given to observations
    Count,
    /// Weight individual periodograms by $\chi^2 = \sum \left(\frac{m_i - \bar{m}}{\delta_i}\right)^2$
    ///
    /// Be aware that if no weight are given to observations
    /// (i.e. via [TimeSeries::new_without_weight]) unity weights are assumed and this is NOT
    /// equivalent to [::Count], but weighting by magnitude variance.
    Chi2,
}

/// Multi-passband periodogram
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(
    bound = "T: Float, F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>, <F as TryInto<PeriodogramPeaks>>::Error: Debug,"
)]
pub struct MultiColorPeriodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    // We use it to not reimplement some internals
    monochrome: Periodogram<T, F>,
    normalization: MultiColorPeriodogramNormalisation,
}

impl<T, F> MultiColorPeriodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks>,
{
    pub fn new(peaks: usize, normalization: MultiColorPeriodogramNormalisation) -> Self {
        let monochrome = Periodogram::with_name_description(
            peaks,
            "multicolor_periodogram",
            "of multi-color periodogram (interpreting frequency as time, power as magnitude)",
        );
        Self {
            monochrome,
            normalization,
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

    /// Set frequency resolution
    ///
    /// The larger frequency resolution allows to find peak period with better precision
    pub fn set_freq_resolution(&mut self, resolution: f32) -> &mut Self {
        self.monochrome.set_freq_resolution(resolution);
        self
    }

    /// Multiply maximum (Nyquist) frequency
    ///
    /// Maximum frequency is Nyquist frequncy multiplied by this factor. The larger factor allows
    /// to find larger frequency and makes [PeriodogramPowerFft] more precise. However large
    /// frequencies can show false peaks
    pub fn set_max_freq_factor(&mut self, max_freq_factor: f32) -> &mut Self {
        self.monochrome.set_max_freq_factor(max_freq_factor);
        self
    }

    /// Define Nyquist frequency
    pub fn set_nyquist(&mut self, nyquist: NyquistFreq) -> &mut Self {
        self.monochrome.set_nyquist(nyquist);
        self
    }

    /// Extend a feature to extract from periodogram
    pub fn add_feature(&mut self, feature: F) -> &mut Self {
        self.monochrome.add_feature(feature);
        self
    }

    pub fn set_periodogram_algorithm(
        &mut self,
        periodogram_power: PeriodogramPower<T>,
    ) -> &mut Self {
        self.monochrome.set_periodogram_algorithm(periodogram_power);
        self
    }
}

impl<T, F> MultiColorPeriodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn power_from_periodogram<'slf, 'a, 'mcts, P>(
        &self,
        p: &periodogram::Periodogram<T>,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<Array1<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
        P: PassbandTrait,
    {
        let ts_weights = {
            let mut a: Array1<_> = match self.normalization {
                MultiColorPeriodogramNormalisation::Count => {
                    mcts.mapping_mut().values().map(|ts| ts.lenf()).collect()
                }
                MultiColorPeriodogramNormalisation::Chi2 => mcts
                    .mapping_mut()
                    .values_mut()
                    .map(|ts| ts.get_m_chi2())
                    .collect(),
            };
            let norm = a.sum();
            if norm.is_zero() {
                match self.normalization {
                    MultiColorPeriodogramNormalisation::Count => {
                        return Err(MultiColorEvaluatorError::all_time_series_short(
                            mcts.mapping_mut(),
                            self.min_ts_length(),
                        ));
                    }
                    MultiColorPeriodogramNormalisation::Chi2 => {
                        return Err(MultiColorEvaluatorError::AllTimeSeriesAreFlat);
                    }
                }
            }
            a /= norm;
            a
        };
        mcts.mapping_mut()
            .values_mut()
            .zip(ts_weights.iter())
            .filter(|(ts, _ts_weight)| self.monochrome.check_ts_length(ts).is_ok())
            .map(|(ts, &ts_weight)| {
                let mut power = p.power(ts);
                power *= ts_weight;
                power
            })
            .reduce(|mut acc, power| {
                acc += &power;
                acc
            })
            .ok_or_else(|| {
                MultiColorEvaluatorError::all_time_series_short(
                    mcts.mapping_mut(),
                    self.min_ts_length(),
                )
            })
    }

    pub fn power<'slf, 'a, 'mcts, P>(
        &self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<Array1<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
        P: PassbandTrait,
    {
        self.power_from_periodogram(
            &self.monochrome.periodogram(mcts.flat_mut().t.as_slice()),
            mcts,
        )
    }

    pub fn freq_power<'slf, 'a, 'mcts, P>(
        &self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<(Array1<T>, Array1<T>), MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
        P: PassbandTrait,
    {
        let p = self.monochrome.periodogram(mcts.flat_mut().t.as_slice());
        let power = self.power_from_periodogram(&p, mcts)?;
        let freq = (0..power.len()).map(|i| p.freq_by_index(i)).collect();
        Ok((freq, power))
    }
}

impl<T, F> MultiColorPeriodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn transform_mcts_to_ts<P>(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
    ) -> Result<TmArrays<T>, MultiColorEvaluatorError>
    where
        P: PassbandTrait,
    {
        let (freq, power) = self.freq_power(mcts)?;
        Ok(TmArrays { t: freq, m: power })
    }
}

impl<T, F> EvaluatorInfoTrait for MultiColorPeriodogram<T, F>
where
    T: Float,
    F: FeatureEvaluator<T> + From<PeriodogramPeaks> + TryInto<PeriodogramPeaks>,
    <F as TryInto<PeriodogramPeaks>>::Error: Debug,
{
    fn get_info(&self) -> &EvaluatorInfo {
        self.monochrome.get_info()
    }
}

impl<T, F> FeatureNamesDescriptionsTrait for MultiColorPeriodogram<T, F>
where
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

impl<P, T, F> MultiColorPassbandSetTrait<P> for MultiColorPeriodogram<T, F>
where
    T: Float,
    P: PassbandTrait,
    F: FeatureEvaluator<T>,
{
    fn get_passband_set(&self) -> &PassbandSet<P> {
        &PassbandSet::AllAvailable
    }
}

impl<P, T, F> MultiColorEvaluator<P, T> for MultiColorPeriodogram<T, F>
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
            .feature_extractor
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
            .feature_extractor
            .eval_or_fill(&mut ts, fill_value))
    }
}
