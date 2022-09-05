use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{
    EvaluatorInfo, EvaluatorInfoTrait, FeatureEvaluator, FeatureNamesDescriptionsTrait,
};
use crate::features::Median;
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
pub struct ColorOfMedian<P>
where
    P: Ord,
{
    passband_set: PassbandSet<P>,
    passbands: [P; 2],
    median: Median,
    name: String,
    description: String,
}

impl<P> ColorOfMedian<P>
where
    P: PassbandTrait,
{
    pub fn new(passbands: [P; 2]) -> Self {
        let set: BTreeSet<_> = passbands.clone().into();
        Self {
            passband_set: set.into(),
            name: format!(
                "color_median_{}_{}",
                passbands[0].name(),
                passbands[1].name()
            ),
            description: format!(
                "difference of median magnitudes {}-{}",
                passbands[0].name(),
                passbands[1].name()
            ),
            passbands,
            median: Median {},
        }
    }
}

lazy_info!(
    COLOR_OF_MEDIAN_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
    variability_required: false,
);

impl<P> EvaluatorInfoTrait for ColorOfMedian<P>
where
    P: Ord,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &COLOR_OF_MEDIAN_INFO
    }
}

impl<P> FeatureNamesDescriptionsTrait for ColorOfMedian<P>
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

impl<P> MultiColorPassbandSetTrait<P> for ColorOfMedian<P>
where
    P: PassbandTrait,
{
    fn get_passband_set(&self) -> &PassbandSet<P> {
        &self.passband_set
    }
}

impl<P, T> MultiColorEvaluator<P, T> for ColorOfMedian<P>
where
    P: PassbandTrait,
    T: Float,
{
    fn eval_multicolor_no_mcts_check(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError> {
        let mut medians = [T::zero(); 2];
        for (median, passband) in medians.iter_mut().zip(self.passbands.iter()) {
            *median = self
                .median
                .eval(mcts.get_mut(passband).expect(
                    "we checked all needed passbands are in mcts, but we still cannot find one",
                ))
                .map_err(|error| MultiColorEvaluatorError::MonochromeEvaluatorError {
                    passband: passband.name().into(),
                    error,
                })?[0]
        }
        Ok(vec![medians[0] - medians[1]])
    }
}
