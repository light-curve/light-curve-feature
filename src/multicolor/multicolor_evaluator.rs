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
use itertools::Itertools;
pub use lazy_static::lazy_static;
pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
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

enum InternalMctsError {
    MultiColorEvaluatorError(MultiColorEvaluatorError),
    InternalWrongPassbandSet,
}

impl InternalMctsError {
    fn into_multi_color_evaluator_error<P, T>(
        self,
        mcts: &MultiColorTimeSeries<P, T>,
        ps: &PassbandSet<P>,
    ) -> MultiColorEvaluatorError
    where
        P: PassbandTrait,
        T: Float,
    {
        match self {
            InternalMctsError::MultiColorEvaluatorError(e) => e,
            InternalMctsError::InternalWrongPassbandSet => {
                MultiColorEvaluatorError::wrong_passbands_error(
                    mcts.keys(),
                    match ps {
                        PassbandSet::FixedSet(ps) => ps.iter(),
                        PassbandSet::AllAvailable => {
                            panic!("PassbandSet cannot be ::AllAvailable here")
                        }
                    },
                )
            }
        }
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
    /// Version of [MultiColorEvaluator::eval_multicolor] without basic [MultiColorTimeSeries] checks
    fn eval_multicolor_no_mcts_check(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError>;

    /// Vector of feature values or `EvaluatorError`
    fn eval_multicolor(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError> {
        self.check_mcts(mcts)?;
        self.eval_multicolor_no_mcts_check(mcts)
    }

    /// Returns vector of feature values and fill invalid components with given value
    fn eval_or_fill_multicolor(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
        fill_value: T,
    ) -> Result<Vec<T>, MultiColorEvaluatorError> {
        Ok(match self.eval_multicolor(mcts) {
            Ok(v) => v,
            Err(_) => vec![fill_value; self.size_hint()],
        })
    }

    /// Check [MultiColorTimeSeries] to have required passbands and individual [TimeSeries] are valid
    fn check_mcts(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
    ) -> Result<(), MultiColorEvaluatorError> {
        mcts.iter_passband_set_mut(self.get_passband_set())
            .map(|(p, maybe_ts)| {
                maybe_ts
                    .ok_or(InternalMctsError::InternalWrongPassbandSet)
                    .and_then(|ts| {
                        self.check_ts(ts).map_err(|error| {
                            InternalMctsError::MultiColorEvaluatorError(
                                MultiColorEvaluatorError::MonochromeEvaluatorError {
                                    error,
                                    passband: p.name().into(),
                                },
                            )
                        })
                    })
                    .map(|_| ())
            })
            .try_collect()
            .map_err(|err| err.into_multi_color_evaluator_error(mcts, self.get_passband_set()))
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::data::TimeSeries;
    use crate::multicolor::MonochromePassband;

    use std::collections::BTreeMap;

    #[derive(Clone, Debug, Serialize)]
    struct TestTimeMultiColorFeature {
        passband_set: PassbandSet<MonochromePassband<'static, f64>>,
    }

    lazy_info!(
        TEST_TIME_FEATURE_INFO,
        TestTimeMultiColorFeature,
        size: 1,
        min_ts_length: 1,
        t_required: true,
        m_required: false,
        w_required: false,
        sorting_required: true,
        variability_required: false,
    );

    impl FeatureNamesDescriptionsTrait for TestTimeMultiColorFeature {
        fn get_names(&self) -> Vec<&str> {
            vec!["zero"]
        }

        fn get_descriptions(&self) -> Vec<&str> {
            vec!["zero"]
        }
    }

    impl MultiColorPassbandSetTrait<MonochromePassband<'static, f64>> for TestTimeMultiColorFeature {
        fn get_passband_set(&self) -> &PassbandSet<MonochromePassband<'static, f64>> {
            &self.passband_set
        }
    }

    impl<T> MultiColorEvaluator<MonochromePassband<'static, f64>, T> for TestTimeMultiColorFeature
    where
        T: Float,
    {
        fn eval_multicolor_no_mcts_check(
            &self,
            _mcts: &mut MultiColorTimeSeries<MonochromePassband<'static, f64>, T>,
        ) -> Result<Vec<T>, MultiColorEvaluatorError> {
            Ok(vec![T::zero()])
        }
    }

    #[test]
    fn test_check_mcts_passbands() {
        let t = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let m = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let passband_b_capital = MonochromePassband::new(4400e-8, "B");
        let passband_v_capital = MonochromePassband::new(5500e-8, "V");
        let passband_r_capital = MonochromePassband::new(6400e-8, "R");
        let mut mcts = {
            let mut passbands = BTreeMap::new();
            passbands.insert(
                passband_b_capital.clone(),
                TimeSeries::new_without_weight(&t, &m),
            );
            passbands.insert(
                passband_v_capital.clone(),
                TimeSeries::new_without_weight(&t, &m),
            );
            MultiColorTimeSeries::new(passbands)
        };

        let feature = TestTimeMultiColorFeature {
            passband_set: PassbandSet::AllAvailable,
        };
        assert!(feature.eval_multicolor(&mut mcts).is_ok());

        let feature = TestTimeMultiColorFeature {
            passband_set: PassbandSet::FixedSet(
                [passband_b_capital.clone(), passband_v_capital.clone()].into(),
            ),
        };
        assert!(feature.eval_multicolor(&mut mcts).is_ok());

        let feature = TestTimeMultiColorFeature {
            passband_set: PassbandSet::FixedSet([passband_b_capital.clone()].into()),
        };
        assert!(feature.eval_multicolor(&mut mcts).is_ok());

        let feature = TestTimeMultiColorFeature {
            passband_set: PassbandSet::FixedSet([passband_r_capital.clone()].into()),
        };
        assert!(feature.eval_multicolor(&mut mcts).is_err());

        let feature = TestTimeMultiColorFeature {
            passband_set: PassbandSet::FixedSet(
                [
                    passband_b_capital.clone(),
                    passband_r_capital.clone(),
                    passband_r_capital.clone(),
                ]
                .into(),
            ),
        };
        assert!(feature.eval_multicolor(&mut mcts).is_err());
    }
}
