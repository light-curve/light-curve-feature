pub use crate::data::MultiColorTimeSeries;
pub use crate::error::MultiColorEvaluatorError;
pub use crate::evaluator::{
    EvaluatorError, EvaluatorInfo, EvaluatorInfoTrait, EvaluatorProperties, FeatureEvaluator,
    FeatureNamesDescriptionsTrait,
};
pub use crate::feature::Feature;
pub use crate::float_trait::Float;
pub use crate::multicolor::PassbandTrait;

use enum_dispatch::enum_dispatch;
pub use lazy_static::lazy_static;
pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

#[enum_dispatch]
pub trait MultiColorPassbandSetTrait<P>
where
    P: PassbandTrait,
{
    fn get_passband_set(&self) -> &PassbandSet<P>;
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound(deserialize = "P: PassbandTrait + Deserialize<'de>"))]
#[non_exhaustive]
pub enum PassbandSet<P>
where
    P: Ord,
{
    FixedSet(BTreeSet<P>),
    AllAvailable,
}

impl<P> From<BTreeSet<P>> for PassbandSet<P>
where
    P: Ord,
{
    fn from(value: BTreeSet<P>) -> Self {
        Self::FixedSet(value)
    }
}

#[enum_dispatch]
pub trait MultiColorEvaluator<P, T>:
    FeatureNamesDescriptionsTrait
    + EvaluatorInfoTrait
    + MultiColorPassbandSetTrait<P>
    + Clone
    + Serialize
where
    P: PassbandTrait,
    T: Float,
{
    /// Vector of feature values or `EvaluatorError`
    fn eval_multicolor(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError>;

    /// Returns vector of feature values and fill invalid components with given value
    fn eval_or_fill_multicolor(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
        fill_value: T,
    ) -> Result<Vec<T>, MultiColorEvaluatorError> {
        self.check_mcts_shape(mcts)?;
        Ok(match self.eval_multicolor(mcts) {
            Ok(v) => v,
            Err(_) => vec![fill_value; self.size_hint()],
        })
    }

    fn check_mcts_shape(
        &self,
        mcts: &MultiColorTimeSeries<P, T>,
    ) -> Result<BTreeMap<P, usize>, MultiColorEvaluatorError> {
        self.check_mcts_passabands(mcts)?;
        self.check_every_ts_length(mcts)
    }

    fn check_mcts_passabands(
        &self,
        mcts: &MultiColorTimeSeries<P, T>,
    ) -> Result<(), MultiColorEvaluatorError> {
        match self.get_passband_set() {
            PassbandSet::AllAvailable => Ok(()),
            PassbandSet::FixedSet(self_passbands) => {
                if mcts
                    .keys()
                    .all(|mcts_passband| self_passbands.contains(mcts_passband))
                {
                    Ok(())
                } else {
                    Err(MultiColorEvaluatorError::wrong_passbands_error(
                        mcts.keys(),
                        self_passbands.iter(),
                    ))
                }
            }
        }
    }

    /// Checks if each component of [MultiColorTimeSeries] has enough points to evaluate the feature
    fn check_every_ts_length(
        &self,
        mcts: &MultiColorTimeSeries<P, T>,
    ) -> Result<BTreeMap<P, usize>, MultiColorEvaluatorError> {
        // Use try_reduce when stabilizes
        // https://github.com/rust-lang/rust/issues/87053
        mcts.iter()
            .map(|(passband, ts)| {
                let length = ts.lenu();
                if length < self.min_ts_length() {
                    Err(MultiColorEvaluatorError::MonochromeEvaluatorError {
                        error: EvaluatorError::ShortTimeSeries {
                            actual: length,
                            minimum: self.min_ts_length(),
                        },
                        passband: passband.name().into(),
                    })
                } else {
                    Ok((passband.clone(), length))
                }
            })
            .collect()
    }
}
