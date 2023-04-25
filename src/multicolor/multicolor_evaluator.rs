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

/// Trait for getting alphabetically sorted passbands
#[enum_dispatch]
pub trait MultiColorPassbandSetTrait<P>
where
    P: PassbandTrait,
{
    /// Get passband set for this evaluator
    fn get_passband_set(&self) -> &PassbandSet<P>;
}

/// Enum for passband set, which can be either fixed set or all available passbands.
/// This is used for [MultiColorEvaluator]s, which can be evaluated on all available passbands
/// (for example [MultiColorPeriodogram](super::features::MultiColorPeriodogram)) or on fixed set of
/// passbands (for example [ColorOfMaximum](super::ColorOfMaximum)).
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound(deserialize = "P: PassbandTrait + Deserialize<'de>"))]
#[non_exhaustive]
pub enum PassbandSet<P>
where
    P: Ord,
{
    /// Fixed set of passbands
    FixedSet(BTreeSet<P>),
    /// All available passbands
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

/// Helper error for [MultiColorEvaluator]
enum InternalMctsError {
    MultiColorEvaluatorError(MultiColorEvaluatorError),
    InternalWrongPassbandSet,
}

impl InternalMctsError {
    fn into_multi_color_evaluator_error<'mcts, 'a, 'ps, P, T>(
        self,
        mcts: &'mcts MultiColorTimeSeries<'a, P, T>,
        ps: &'ps PassbandSet<P>,
    ) -> MultiColorEvaluatorError
    where
        'ps: 'a,
        'a: 'mcts,
        P: PassbandTrait,
        T: Float,
    {
        match self {
            InternalMctsError::MultiColorEvaluatorError(e) => e,
            InternalMctsError::InternalWrongPassbandSet => {
                MultiColorEvaluatorError::wrong_passbands_error(
                    mcts.passbands(),
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

/// Trait for multi-color feature evaluators
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
    fn eval_multicolor_no_mcts_check<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts;

    /// Vector of feature values or `EvaluatorError`
    fn eval_multicolor<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
        P: 'a,
    {
        self.check_mcts(mcts)?;
        self.eval_multicolor_no_mcts_check(mcts)
    }

    /// Returns vector of feature values and fill invalid components with given value
    fn eval_or_fill_multicolor<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
        fill_value: T,
    ) -> Result<Vec<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
        P: 'a,
    {
        Ok(match self.eval_multicolor(mcts) {
            Ok(v) => v,
            Err(_) => vec![fill_value; self.size_hint()],
        })
    }

    /// Check [MultiColorTimeSeries] to have required passbands and individual [TimeSeries] are valid
    fn check_mcts<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<(), MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
        P: 'a,
    {
        mcts.mapping_mut()
            .iter_passband_set_mut(self.get_passband_set())
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
        fn eval_multicolor_no_mcts_check<'slf, 'a, 'mcts>(
            &'slf self,
            _mcts: &'mcts mut MultiColorTimeSeries<'a, MonochromePassband<'static, f64>, T>,
        ) -> Result<Vec<T>, MultiColorEvaluatorError>
        where
            'slf: 'a,
            'a: 'mcts,
        {
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
            let mut mapping = BTreeMap::new();
            mapping.insert(
                passband_b_capital.clone(),
                TimeSeries::new_without_weight(&t, &m),
            );
            mapping.insert(
                passband_v_capital.clone(),
                TimeSeries::new_without_weight(&t, &m),
            );
            MultiColorTimeSeries::from_map(mapping)
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
