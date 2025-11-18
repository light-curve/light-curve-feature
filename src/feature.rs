use crate::evaluator::*;
use crate::extractor::FeatureExtractor;
use crate::features::*;
use crate::float_trait::Float;
use crate::time_series::TimeSeries;
use crate::transformers::Transformer;

use enum_dispatch::enum_dispatch;
use std::fmt::Debug;

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
    Adf,
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
