use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{
    EvaluatorInfo, EvaluatorInfoTrait, EvaluatorProperties, FeatureEvaluator,
    FeatureNamesDescriptionsTrait,
};
use crate::float_trait::Float;
use crate::multicolor::multicolor_evaluator::*;

use itertools::Itertools;
pub use lazy_static::lazy_static;
pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Multi-color feature which evaluates non-color dependent feature for each passband.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound(
    deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float, F: FeatureEvaluator<T>"
))]
pub struct MonochromeFeature<P, T, F>
where
    P: Ord,
{
    feature: F,
    passband_set: PassbandSet<P>,
    properties: Box<EvaluatorProperties>,
    phantom: PhantomData<T>,
}

impl<P, T, F> MonochromeFeature<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T>,
{
    /// Creates a new instance of `MonochromeFeature`.
    ///
    /// # Arguments
    /// - `feature` - non-multi-color feature to evaluate for each passband.
    /// - `passband_set` - set of passbands to evaluate the feature for.
    pub fn new(feature: F, passband_set: BTreeSet<P>) -> Self {
        let names = passband_set
            .iter()
            .cartesian_product(feature.get_names())
            .map(|(passband, name)| format!("{}_{}", name, passband.name()))
            .collect();
        let descriptions = passband_set
            .iter()
            .cartesian_product(feature.get_descriptions())
            .map(|(passband, description)| format!("{}, passband {}", description, passband.name()))
            .collect();
        let info = {
            let mut info = feature.get_info().clone();
            info.size *= passband_set.len();
            info
        };
        Self {
            properties: EvaluatorProperties {
                info,
                names,
                descriptions,
            }
            .into(),
            feature,
            passband_set: passband_set.into(),
            phantom: PhantomData,
        }
    }
}

impl<P, T, F> FeatureNamesDescriptionsTrait for MonochromeFeature<P, T, F>
where
    P: Ord,
{
    fn get_names(&self) -> Vec<&str> {
        self.properties.names.iter().map(String::as_str).collect()
    }

    fn get_descriptions(&self) -> Vec<&str> {
        self.properties
            .descriptions
            .iter()
            .map(String::as_str)
            .collect()
    }
}

impl<P, T, F> EvaluatorInfoTrait for MonochromeFeature<P, T, F>
where
    P: Ord,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}

impl<P, T, F> MultiColorPassbandSetTrait<P> for MonochromeFeature<P, T, F>
where
    P: PassbandTrait,
{
    fn get_passband_set(&self) -> &PassbandSet<P> {
        &self.passband_set
    }
}

impl<P, T, F> MultiColorEvaluator<P, T> for MonochromeFeature<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn eval_multicolor_no_mcts_check<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        match &self.passband_set {
            PassbandSet::FixedSet(set) => {
                mcts.mapping_mut().iter_matched_passbands_mut(set.iter())
                    .map(|(passband, ts)| {
                        self.feature.eval_no_ts_check(
                                ts.expect("we checked all needed passbands are in mcts, but we still cannot find one")
                        ).map_err(|error| MultiColorEvaluatorError::MonochromeEvaluatorError {
                            passband: passband.name().into(),
                            error,
                        })
                    }).flatten_ok().collect()
            }
            PassbandSet::AllAvailable => panic!("passband_set must be FixedSet variant here"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::features::Mean;
    use crate::multicolor::passband::MonochromePassband;
    use crate::Feature;

    #[test]
    fn test_monochrome_feature() {
        let feature: MonochromeFeature<MonochromePassband<_>, f64, Feature<_>> =
            MonochromeFeature::new(
                Mean::default().into(),
                [
                    MonochromePassband::new(4700e-8, "g"),
                    MonochromePassband::new(6200e-8, "r"),
                ]
                .into_iter()
                .collect(),
            );
        assert_eq!(feature.get_names(), vec!["mean_g", "mean_r"]);
        assert_eq!(
            feature.get_descriptions(),
            vec!["mean magnitude, passband g", "mean magnitude, passband r"]
        );
        assert_eq!(feature.get_info().size, 2);
    }
}
