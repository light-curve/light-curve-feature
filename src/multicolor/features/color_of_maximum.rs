use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{EvaluatorInfo, EvaluatorInfoTrait, FeatureNamesDescriptionsTrait};
use crate::float_trait::Float;
use crate::multicolor::multicolor_evaluator::*;
use crate::multicolor::{PassbandSet, PassbandTrait};

pub use lazy_static::lazy_static;
pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt::Debug;

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound(deserialize = "P: PassbandTrait + Deserialize<'de>"))]
pub struct ColorOfMaximum<P>
where
    P: Ord,
{
    passband_set: PassbandSet<P>,
    passbands: [P; 2],
    name: String,
    description: String,
}

impl<P> ColorOfMaximum<P>
where
    P: PassbandTrait,
{
    pub fn new(passbands: [P; 2]) -> Self {
        let set: BTreeSet<_> = passbands.clone().into();
        Self {
            passband_set: set.into(),
            name: format!("color_max_{}_{}", passbands[0].name(), passbands[1].name()),
            description: format!(
                "difference of maximum value magnitudes {}-{}",
                passbands[0].name(),
                passbands[1].name()
            ),
            passbands,
        }
    }
}

lazy_info!(
    COLOR_OF_MAXIMUM_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
    variability_required: false,
);

impl<P> EvaluatorInfoTrait for ColorOfMaximum<P>
where
    P: Ord,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &COLOR_OF_MAXIMUM_INFO
    }
}

impl<P> FeatureNamesDescriptionsTrait for ColorOfMaximum<P>
where
    P: Ord,
{
    fn get_names(&self) -> Vec<&str> {
        vec![self.name.as_str()]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![self.description.as_str()]
    }
}

impl<P> MultiColorPassbandSetTrait<P> for ColorOfMaximum<P>
where
    P: PassbandTrait,
{
    fn get_passband_set(&self) -> &PassbandSet<P> {
        &self.passband_set
    }
}

impl<P, T> MultiColorEvaluator<P, T> for ColorOfMaximum<P>
where
    P: PassbandTrait,
    T: Float,
{
    fn eval_multicolor_no_mcts_check(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError> {
        let mut maxima = [T::zero(); 2];
        for ((_passband, mcts), maximum) in mcts
            .iter_matched_passbands_mut(self.passbands.iter())
            .zip(maxima.iter_mut())
        {
            let mcts = mcts.expect("MultiColorTimeSeries must have all required passbands");
            *maximum = mcts.m.get_max()
        }
        Ok(vec![maxima[0] - maxima[1]])
    }
}
