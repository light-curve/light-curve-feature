#![doc = include_str!("../README.md")]

#[cfg(test)]
#[macro_use]
mod tests;

#[macro_use]
mod macros;

mod evaluator;
pub use evaluator::{EvaluatorInfoTrait, FeatureEvaluator, FeatureNamesDescriptionsTrait};

mod error;
pub use error::EvaluatorError;

mod extractor;
pub use extractor::FeatureExtractor;

mod feature;
pub use feature::Feature;

pub mod features;
pub use features::*;

mod float_trait;
pub use float_trait::Float;

mod lnerfc;

mod nl_fit;
#[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
pub use nl_fit::CeresCurveFit;
pub use nl_fit::CobylaCurveFit;
pub use nl_fit::CurveFitResult;
#[cfg(feature = "gsl")]
pub use nl_fit::LmsderCurveFit;
pub use nl_fit::evaluator::FitFeatureEvaluatorGettersTrait;
pub use nl_fit::{CurveFitAlgorithm, McmcCurveFit};
pub use nl_fit::{LnPrior, LnPrior1D, prior};

#[doc(hidden)]
pub mod periodogram;
pub use periodogram::sin_cos_iterator::RecurrentSinCos;
pub use periodogram::{
    AverageNyquistFreq, FixedNyquistFreq, MedianNyquistFreq, NyquistFreq, PeriodogramPower,
    PeriodogramPowerDirect, PeriodogramPowerFft, QuantileNyquistFreq,
};

pub mod prelude;

mod sorted_array;

mod straight_line_fit;
#[doc(hidden)]
pub use straight_line_fit::fit_straight_line;

pub mod transformers;
pub use transformers::{Transformer, TransformerTrait};

mod peak_indices;
#[doc(hidden)]
pub use peak_indices::peak_indices;

mod time_series;
pub use time_series::{DataSample, TimeSeries};

mod types;

pub use ndarray;
