use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{EvaluatorInfo, EvaluatorInfoTrait, FeatureNamesDescriptionsTrait};
use crate::float_trait::Float;
use crate::multicolor::multicolor_evaluator::*;

use itertools::Itertools;
pub use lazy_static::lazy_static;
pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::marker::PhantomData;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    into = "MultiColorExtractorParameters<MCF>",
    from = "MultiColorExtractorParameters<MCF>",
    bound(
        serialize = "P: PassbandTrait, T: Float, MCF: MultiColorEvaluator<P, T>",
        deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float, MCF: MultiColorEvaluator<P, T> + Deserialize<'de>"
    )
)]
pub struct MultiColorExtractor<P, T, MCF>
where
    P: Ord,
{
    features: Vec<MCF>,
    info: Box<EvaluatorInfo>,
    passband_set: PassbandSet<P>,
    phantom: PhantomData<T>,
}

impl<P, T, MCF> MultiColorExtractor<P, T, MCF>
where
    P: PassbandTrait,
    T: Float,
    MCF: MultiColorEvaluator<P, T>,
{
    pub fn new(features: Vec<MCF>) -> Self {
        let passband_set = {
            let set: BTreeSet<_> = features
                .iter()
                .filter_map(|f| match f.get_passband_set() {
                    PassbandSet::AllAvailable => None,
                    PassbandSet::FixedSet(set) => Some(set),
                })
                .flatten()
                .cloned()
                .collect();
            if set.is_empty() {
                PassbandSet::AllAvailable
            } else {
                PassbandSet::FixedSet(set)
            }
        };

        let info = EvaluatorInfo {
            size: features.iter().map(|x| x.size_hint()).sum(),
            min_ts_length: features
                .iter()
                .map(|x| x.min_ts_length())
                .max()
                .unwrap_or(0),
            t_required: features.iter().any(|x| x.is_t_required()),
            m_required: features.iter().any(|x| x.is_m_required()),
            w_required: features.iter().any(|x| x.is_w_required()),
            sorting_required: features.iter().any(|x| x.is_sorting_required()),
            variability_required: features.iter().any(|x| x.is_variability_required()),
        }
        .into();

        Self {
            features,
            passband_set,
            info,
            phantom: PhantomData,
        }
    }
}

impl<P, T, MCF> FeatureNamesDescriptionsTrait for MultiColorExtractor<P, T, MCF>
where
    P: Ord,
    MCF: FeatureNamesDescriptionsTrait,
{
    /// Get feature names
    fn get_names(&self) -> Vec<&str> {
        self.features.iter().flat_map(|x| x.get_names()).collect()
    }

    /// Get feature descriptions
    fn get_descriptions(&self) -> Vec<&str> {
        self.features
            .iter()
            .flat_map(|x| x.get_descriptions())
            .collect()
    }
}

impl<P, T, MCF> EvaluatorInfoTrait for MultiColorExtractor<P, T, MCF>
where
    P: Ord,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.info
    }
}

impl<P, T, F> MultiColorPassbandSetTrait<P> for MultiColorExtractor<P, T, F>
where
    P: PassbandTrait,
{
    fn get_passband_set(&self) -> &PassbandSet<P> {
        &self.passband_set
    }
}

impl<P, T, MCF> MultiColorEvaluator<P, T> for MultiColorExtractor<P, T, MCF>
where
    P: PassbandTrait,
    T: Float,
    MCF: MultiColorEvaluator<P, T>,
{
    fn eval_multicolor_no_mcts_check(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError> {
        let mut vec = Vec::with_capacity(self.size_hint());
        for x in &self.features {
            vec.extend(x.eval_multicolor(mcts)?);
        }
        Ok(vec)
    }

    fn eval_or_fill_multicolor(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
        fill_value: T,
    ) -> Result<Vec<T>, MultiColorEvaluatorError> {
        self.features
            .iter()
            .map(|x| x.eval_or_fill_multicolor(mcts, fill_value))
            .flatten_ok()
            .collect()
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "MultiColorExtractor")]
struct MultiColorExtractorParameters<MCF> {
    features: Vec<MCF>,
}

impl<P, T, MCF> From<MultiColorExtractor<P, T, MCF>> for MultiColorExtractorParameters<MCF>
where
    P: PassbandTrait,
    T: Float,
    MCF: MultiColorEvaluator<P, T>,
{
    fn from(f: MultiColorExtractor<P, T, MCF>) -> Self {
        Self {
            features: f.features,
        }
    }
}

impl<P, T, MCF> From<MultiColorExtractorParameters<MCF>> for MultiColorExtractor<P, T, MCF>
where
    P: PassbandTrait,
    T: Float,
    MCF: MultiColorEvaluator<P, T>,
{
    fn from(p: MultiColorExtractorParameters<MCF>) -> Self {
        Self::new(p.features)
    }
}

impl<P, T, MCF> JsonSchema for MultiColorExtractor<P, T, MCF>
where
    P: PassbandTrait,
    T: Float,
    MCF: JsonSchema,
{
    json_schema!(MultiColorExtractorParameters<MCF>, true);
}
