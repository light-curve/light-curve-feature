#![doc = include_str!("../README.md")]

extern crate core;

#[cfg(test)]
#[macro_use]
mod tests;

#[macro_use]
mod macros;

mod data;
pub use data::{DataSample, TimeSeries};

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

mod multicolor;
pub use multicolor::*;

mod nl_fit;
pub use nl_fit::evaluator::FitFeatureEvaluatorGettersTrait;
#[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
pub use nl_fit::CeresCurveFit;
#[cfg(feature = "gsl")]
pub use nl_fit::LmsderCurveFit;
pub use nl_fit::{prior, LnPrior, LnPrior1D};
pub use nl_fit::{CurveFitAlgorithm, McmcCurveFit};

mod number_ending;
pub(crate) use number_ending::number_ending;

#[doc(hidden)]
pub mod periodogram;
pub use periodogram::recurrent_sin_cos::RecurrentSinCos;
pub use periodogram::{
    AverageNyquistFreq, FixedNyquistFreq, MedianNyquistFreq, NyquistFreq, PeriodogramPower,
    PeriodogramPowerDirect, PeriodogramPowerFft, QuantileNyquistFreq,
};

pub mod prelude;

mod straight_line_fit;
#[doc(hidden)]
pub use straight_line_fit::fit_straight_line;

pub mod transformers;
pub use transformers::{Transformer, TransformerTrait};

mod peak_indices;
#[doc(hidden)]
pub use peak_indices::peak_indices;

mod types;

pub use ndarray;
