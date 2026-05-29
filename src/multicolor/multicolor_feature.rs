use crate::data::{MultiColorTimeSeries, TimeSeries};
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{EvaluatorInfoTrait, FeatureNamesDescriptionsTrait};
use crate::float_trait::Float;
use crate::multicolor::features::{
    ColorOfMaximum, ColorOfMedian, ColorOfMinimum, ColorSpread, MultiColorPeriodogram,
};
use crate::multicolor::multicolor_evaluator::*;
use crate::multicolor::{MultiColorBins, MultiColorExtractor, PerBandFeature};

use enum_dispatch::enum_dispatch;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[enum_dispatch(MultiColorEvaluator<P, T>, FeatureNamesDescriptionsTrait, EvaluatorInfoTrait, MultiColorPassbandSetTrait<P>)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound(deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float"))]
#[non_exhaustive]
pub enum MultiColorFeature<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    // Extractor
    MultiColorExtractor(MultiColorExtractor<P, T>),
    // Monochrome Features
    PerBandFeature(PerBandFeature<P, T, Feature<T>>),
    // Features
    ColorOfMaximum(ColorOfMaximum<P>),
    ColorOfMedian(ColorOfMedian<P>),
    ColorOfMinimum(ColorOfMinimum<P>),
    ColorSpread(ColorSpread<P>),
    MultiColorPeriodogram(MultiColorPeriodogram<P, T, Feature<T>>),
    MultiColorBins(MultiColorBins<P, T>),
}

impl<P, T> MultiColorFeature<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    pub fn from_per_band_feature<F>(feature: F, passbands: Vec<P>) -> Self
    where
        F: Into<Feature<T>>,
    {
        PerBandFeature::new(feature.into(), passbands).into()
    }
}
