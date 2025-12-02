use crate::evaluator::*;
use crate::extractor::FeatureExtractor;
use crate::features::*;
use crate::float_trait::Float;
use crate::time_series::TimeSeries;
use crate::transformers::Transformer;

use enum_dispatch::enum_dispatch;
use std::fmt::Debug;
use std::hash::{Hash, Hasher};

/// All features are available as variants of this enum
///
/// Consider to import [crate::FeatureEvaluator] as well
#[enum_dispatch(FeatureEvaluator<T>, FeatureNamesDescriptionsTrait, EvaluatorInfoTrait)]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
#[serde(bound = "T: Float")]
#[non_exhaustive]
pub enum Feature<T>
where
    T: Float,
{
    // extractor
    FeatureExtractor(FeatureExtractor<T, Self>),
    // features
    Amplitude,
    AndersonDarlingNormal,
    Bins(Bins<T, Self>),
    BazinFit,
    BeyondNStd(BeyondNStd<T>),
    Cusum,
    Duration,
    Eta,
    EtaE,
    ExcessVariance,
    InterPercentileRange,
    Kurtosis,
    LinearFit,
    LinearTrend,
    LinexpFit,
    MagnitudePercentageRatio,
    MaximumSlope,
    MaximumTimeInterval,
    MinimumTimeInterval,
    Mean,
    MeanVariance,
    Median,
    MedianAbsoluteDeviation,
    MedianBufferRangePercentage(MedianBufferRangePercentage<T>),
    ObservationCount,
    OtsuSplit,
    PercentAmplitude,
    PercentDifferenceMagnitudePercentile,
    Periodogram(Periodogram<T, Self>),
    _PeriodogramPeaks,
    ReducedChi2,
    Roms,
    Skew,
    StandardDeviation,
    StetsonK,
    TimeMean,
    TimeStandardDeviation,
    Transformed(Transformed<T, Self, Transformer<T>>),
    VillarFit,
    WeightedMean,
}

impl<T: Float> Hash for Feature<T> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Feature::FeatureExtractor(v) => v.hash(state),
            Feature::Amplitude(v) => v.hash(state),
            Feature::AndersonDarlingNormal(v) => v.hash(state),
            Feature::Bins(v) => v.hash(state),
            Feature::BazinFit(v) => v.hash(state),
            Feature::BeyondNStd(v) => v.hash(state),
            Feature::Cusum(v) => v.hash(state),
            Feature::Duration(v) => v.hash(state),
            Feature::Eta(v) => v.hash(state),
            Feature::EtaE(v) => v.hash(state),
            Feature::ExcessVariance(v) => v.hash(state),
            Feature::InterPercentileRange(v) => v.hash(state),
            Feature::Kurtosis(v) => v.hash(state),
            Feature::LinearFit(v) => v.hash(state),
            Feature::LinearTrend(v) => v.hash(state),
            Feature::LinexpFit(v) => v.hash(state),
            Feature::MagnitudePercentageRatio(v) => v.hash(state),
            Feature::MaximumSlope(v) => v.hash(state),
            Feature::MaximumTimeInterval(v) => v.hash(state),
            Feature::MinimumTimeInterval(v) => v.hash(state),
            Feature::Mean(v) => v.hash(state),
            Feature::MeanVariance(v) => v.hash(state),
            Feature::Median(v) => v.hash(state),
            Feature::MedianAbsoluteDeviation(v) => v.hash(state),
            Feature::MedianBufferRangePercentage(v) => v.hash(state),
            Feature::ObservationCount(v) => v.hash(state),
            Feature::OtsuSplit(v) => v.hash(state),
            Feature::PercentAmplitude(v) => v.hash(state),
            Feature::PercentDifferenceMagnitudePercentile(v) => v.hash(state),
            Feature::Periodogram(v) => v.hash(state),
            Feature::_PeriodogramPeaks(v) => v.hash(state),
            Feature::ReducedChi2(v) => v.hash(state),
            Feature::Roms(v) => v.hash(state),
            Feature::Skew(v) => v.hash(state),
            Feature::StandardDeviation(v) => v.hash(state),
            Feature::StetsonK(v) => v.hash(state),
            Feature::TimeMean(v) => v.hash(state),
            Feature::TimeStandardDeviation(v) => v.hash(state),
            Feature::Transformed(v) => v.hash(state),
            Feature::VillarFit(v) => v.hash(state),
            Feature::WeightedMean(v) => v.hash(state),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_feature_partial_eq() {
        // Test unit struct features
        let amplitude1: Feature<f64> = Amplitude::default().into();
        let amplitude2: Feature<f64> = Amplitude::default().into();
        assert_eq!(amplitude1, amplitude2);

        let mean1: Feature<f64> = Mean::default().into();
        let mean2: Feature<f64> = Mean::default().into();
        assert_eq!(mean1, mean2);

        // Test that different features are not equal
        assert_ne!(amplitude1, mean1);

        // Test parametric features
        let beyond1: Feature<f64> = BeyondNStd::default().into();
        let beyond2: Feature<f64> = BeyondNStd::default().into();
        assert_eq!(beyond1, beyond2);

        let beyond3: Feature<f64> = BeyondNStd::new(2.0).into();
        assert_ne!(beyond1, beyond3);
    }
}
