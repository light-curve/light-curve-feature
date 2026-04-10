use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{EvaluatorInfoTrait, FeatureEvaluator, FeatureNamesDescriptionsTrait};
use crate::features::Median;
use crate::float_trait::Float;
use crate::multicolor::multicolor_evaluator::*;
use crate::multicolor::{PassbandSet, PassbandTrait};

use lazy_static::lazy_static;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt::Debug;

/// Difference of median magnitudes in two passbands
///
/// Note that median is calculated for each passband separately
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
    fn eval_multicolor_no_mcts_check<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        let mut medians = [T::zero(); 2];
        for ((passband, mcts), median) in mcts
            .mapping_mut()
            .iter_matched_passbands_mut(self.passbands.iter())
            .zip(medians.iter_mut())
        {
            let mcts = mcts.expect("MultiColorTimeSeries must have all required passbands");
            *median = self.median.eval(mcts).map_err(|error| {
                MultiColorEvaluatorError::MonochromeEvaluatorError {
                    passband: passband.name().into(),
                    error,
                }
            })?[0]
        }
        Ok(vec![medians[0] - medians[1]])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MultiColorTimeSeries, StringPassband};

    #[test]
    fn color_of_median_values() {
        let eval = ColorOfMedian::new([StringPassband::from("g"), StringPassband::from("r")]);
        // g band: [4.0, 6.0, 5.0] -> median = 5.0; r band: [1.0, 3.0, 2.0] -> median = 2.0
        let t = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let m = vec![4.0_f64, 6.0, 5.0, 1.0, 3.0, 2.0];
        let w = vec![1.0_f64; 6];
        let bands: Vec<StringPassband> = vec!["g", "g", "g", "r", "r", "r"]
            .into_iter()
            .map(StringPassband::from)
            .collect();
        let mut mcts = MultiColorTimeSeries::from_flat(t, m, w, bands);
        let result = eval.eval_multicolor(&mut mcts).unwrap();
        assert!((result[0] - (5.0 - 2.0)).abs() < 1e-10);
    }

    #[test]
    fn color_of_median_names() {
        let eval = ColorOfMedian::new([StringPassband::from("g"), StringPassband::from("r")]);
        assert_eq!(eval.get_names(), vec!["color_median_g_r"]);
        assert_eq!(eval.size_hint(), 1);
    }

    #[test]
    fn color_of_median_serde() {
        let eval = ColorOfMedian::new([StringPassband::from("g"), StringPassband::from("r")]);
        let json = serde_json::to_string(&eval).unwrap();
        let eval2: ColorOfMedian<StringPassband> = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string(&eval2).unwrap());
    }
}
