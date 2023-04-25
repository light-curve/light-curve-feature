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
    properties: Box<EvaluatorProperties>,
}

impl<T, F> MultiColorPeriodogram<T, F>
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
        let unnormed_power = mcts
            .mapping_mut()
            .values_mut()
            .filter(|ts| self.monochrome.check_ts_length(ts).is_ok())
            .map(|ts| p.power(ts) * ts.lenf())
            .reduce(|acc, x| acc + x)
            .ok_or_else(|| {
                MultiColorEvaluatorError::all_time_series_short(
                    mcts.mapping_mut(),
                    self.monochrome.min_ts_length(),
                )
            })?;
        Ok(unnormed_power / mcts.total_lenf())
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
        &self.properties.info
    }
}

impl<T, F> FeatureNamesDescriptionsTrait for MultiColorPeriodogram<T, F>
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
