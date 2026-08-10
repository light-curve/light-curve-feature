// Licu, the project's mascot, in the rustdoc sidebar and the browser tab.
// These have to be URLs: rustdoc drops them straight into <img src> and
// <link rel="icon">, and copies nothing next to the generated pages, so a path
// relative to the repository would resolve against docs.rs and 404. They point
// at this crate's own copy under assets/logo/ rather than at the branding
// repository, so the two cannot drift apart.
//
// The adaptive mark is the right file for both: it is loaded as an isolated
// document that the surrounding page cannot style, so the only way it can suit
// a light and a dark theme is by switching on prefers-color-scheme itself.
#![doc(
    html_logo_url = "https://raw.githubusercontent.com/light-curve/light-curve-feature/main/assets/logo/mark-adaptive.svg",
    html_favicon_url = "https://raw.githubusercontent.com/light-curve/light-curve-feature/main/assets/logo/mark-adaptive.svg"
)]
#![doc = include_str!("../README.md")]

extern crate core;

#[cfg(test)]
#[macro_use]
mod tests;

#[macro_use]
mod macros;

mod data;
pub use data::{DataSample, MultiColorTimeSeries, TimeSeries};

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

pub mod multicolor;
pub use multicolor::*;

mod nl_fit;
#[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
pub use nl_fit::CeresCurveFit;
pub use nl_fit::CurveFitResult;
#[cfg(feature = "gsl")]
pub use nl_fit::LmsderCurveFit;
pub use nl_fit::NutsCurveFit;
pub use nl_fit::evaluator::FitFeatureEvaluatorGettersTrait;
pub use nl_fit::{CurveFitAlgorithm, McmcCurveFit};
pub use nl_fit::{LnPrior, LnPrior1D, prior};

mod number_ending;
pub(crate) use number_ending::number_ending;

#[doc(hidden)]
pub mod periodogram;
pub use periodogram::sin_cos_iterator::RecurrentSinCos;
pub use periodogram::{
    AverageNyquistFreq, FixedNyquistFreq, FreqGrid, LinearFreqGrid, MedianNyquistFreq, NyquistFreq,
    PeriodogramPower, PeriodogramPowerDirect, PeriodogramPowerFft, QuantileNyquistFreq,
};

pub mod prelude;

mod parabola_fit;
#[doc(hidden)]
pub use parabola_fit::fit_parabola;

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
