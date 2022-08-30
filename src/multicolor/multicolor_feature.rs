use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{EvaluatorInfo, EvaluatorInfoTrait, FeatureNamesDescriptionsTrait};
use crate::feature::Feature;
use crate::float_trait::Float;
use crate::multicolor::features::ColorOfMedian;
use crate::multicolor::multicolor_evaluator::*;
use crate::multicolor::{MonochromeFeature, MultiColorExtractor};

use enum_dispatch::enum_dispatch;
pub use lazy_static::lazy_static;
pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
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
    MultiColorExtractor(MultiColorExtractor<P, T, MultiColorFeature<P, T>>),
    // Monochrome Features
    MonochromeFeature(MonochromeFeature<P, T, Feature<T>>),
    // Features
    ColorOfMedian(ColorOfMedian<P>),
}

impl<P, T> MultiColorFeature<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    pub fn from_monochrome_feature<F>(feature: F, passband_set: BTreeSet<P>) -> Self
    where
        F: Into<Feature<T>>,
    {
        MonochromeFeature::new(feature.into(), passband_set).into()
    }
}
