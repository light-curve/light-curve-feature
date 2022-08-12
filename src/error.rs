use crate::PassbandTrait;

use std::collections::BTreeSet;

/// Error returned from [crate::FeatureEvaluator]
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum EvaluatorError {
    #[error("time-series' length {actual} is smaller than the minimum required length {minimum}")]
    ShortTimeSeries { actual: usize, minimum: usize },

    #[error("feature value is undefined for a flat time series")]
    FlatTimeSeries,

    #[error("zero division: {0}")]
    ZeroDivision(&'static str),
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MultiColorEvaluatorError {
    #[error("Passband {passband} time-series caused error: {error:?}")]
    MonochromeEvaluatorError {
        passband: String,
        error: EvaluatorError,
    },

    #[error("Wrong passbands {actual:?}, {desired:?} are desired")]
    WrongPassbandsError {
        actual: BTreeSet<String>,
        desired: BTreeSet<String>,
    },
}

impl MultiColorEvaluatorError {
    pub fn wrong_passbands_error<'a, P>(
        actual: impl Iterator<Item = &'a P>,
        desired: impl Iterator<Item = &'a P>,
    ) -> Self
    where
        P: PassbandTrait + 'a,
    {
        Self::WrongPassbandsError {
            actual: actual.map(|p| p.name().into()).collect(),
            desired: desired.map(|p| p.name().into()).collect(),
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SortedArrayError {
    #[error("SortedVec constructors accept sorted arrays only")]
    Unsorted,

    #[error("SortedVec constructors accept contiguous arrays only")]
    NonContiguous,
}
