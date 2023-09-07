use crate::data::multi_color_time_series::MappedMultiColorTimeSeries;
use crate::float_trait::Float;
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

    #[error("No time-series long enough: maximum length found is {maximum_actual}, while minimum required is {minimum_required}")]
    AllTimeSeriesAreShort {
        maximum_actual: usize,
        minimum_required: usize,
    },

    #[error(r#"Underlying feature caused an error: "{0:?}""#)]
    UnderlyingEvaluatorError(#[from] EvaluatorError),

    #[error("All time-series are flat")]
    AllTimeSeriesAreFlat,
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

    pub fn all_time_series_short<P, T>(
        mapped: &MappedMultiColorTimeSeries<P, T>,
        minimum_required: usize,
    ) -> Self
    where
        P: PassbandTrait,
        T: Float,
    {
        Self::AllTimeSeriesAreShort {
            maximum_actual: mapped.iter_ts().map(|ts| ts.lenu()).max().unwrap_or(0),
            minimum_required,
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
