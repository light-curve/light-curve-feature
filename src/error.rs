use crate::periodogram::PeriodogramPowerError;

/// Error returned from [crate::FeatureEvaluator]
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EvaluatorError {
    #[error("time-series' length {actual} is smaller than the minimum required length {minimum}")]
    ShortTimeSeries { actual: usize, minimum: usize },

    #[error("feature value is undefined for a flat time series")]
    FlatTimeSeries,

    #[error("zero division: {0}")]
    ZeroDivision(&'static str),

    #[error("periodogram error: {0}")]
    Periodogram(#[from] PeriodogramPowerError),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SortedArrayError {
    #[error("SortedVec constructors accept sorted arrays only")]
    Unsorted,
    #[error("SortedVec constructors accept contiguous arrays only")]
    NonContiguous,
}
