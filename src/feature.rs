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
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
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
