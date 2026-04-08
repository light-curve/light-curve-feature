use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{EvaluatorInfoTrait, FeatureNamesDescriptionsTrait};
use crate::float_trait::Float;
use crate::multicolor::multicolor_evaluator::*;
use crate::multicolor::multicolor_feature::MultiColorFeature;

use itertools::Itertools;
pub use schemars::JsonSchema;
pub use serde::Serialize;
use std::collections::BTreeSet;
use std::fmt::Debug;

/// Bulk feature evaluator.
///
/// Evaluates multiple [`MultiColorFeature`]s on a [`MultiColorTimeSeries`] and returns
/// a flat vector of all feature values.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    into = "MultiColorExtractorParameters<P, T>",
    from = "MultiColorExtractorParameters<P, T>",
    bound(
        serialize = "P: PassbandTrait, T: Float",
        deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float"
    )
)]
pub struct MultiColorExtractor<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    features: Vec<MultiColorFeature<P, T>>,
    info: Box<EvaluatorInfo>,
    passband_set: PassbandSet<P>,
}

impl<P, T> MultiColorExtractor<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    /// Create a new [MultiColorExtractor]
    ///
    /// # Arguments
    /// `features` - A vector of multi-color features to be evaluated
    pub fn new(features: Vec<MultiColorFeature<P, T>>) -> Self {
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
        }
    }
}

impl<P, T> FeatureNamesDescriptionsTrait for MultiColorExtractor<P, T>
where
    P: PassbandTrait,
    T: Float,
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

impl<P, T> EvaluatorInfoTrait for MultiColorExtractor<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.info
    }
}

impl<P, T> MultiColorPassbandSetTrait<P> for MultiColorExtractor<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn get_passband_set(&self) -> &PassbandSet<P> {
        &self.passband_set
    }
}

impl<P, T> MultiColorEvaluator<P, T> for MultiColorExtractor<P, T>
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
        let mut vec = Vec::with_capacity(self.size_hint());
        for x in &self.features {
            vec.extend(x.eval_multicolor_no_mcts_check(mcts)?);
        }
        Ok(vec)
    }

    fn eval_or_fill_multicolor<'slf, 'a, 'mcts>(
        &'slf self,
        mcts: &'mcts mut MultiColorTimeSeries<'a, P, T>,
        fill_value: T,
    ) -> Result<Vec<T>, MultiColorEvaluatorError>
    where
        'slf: 'a,
        'a: 'mcts,
    {
        self.features
            .iter()
            .map(|x| x.eval_or_fill_multicolor(mcts, fill_value))
            .flatten_ok()
            .collect()
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(
    rename = "MultiColorExtractor",
    bound(
        serialize = "P: PassbandTrait, T: Float",
        deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float"
    )
)]
struct MultiColorExtractorParameters<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    features: Vec<MultiColorFeature<P, T>>,
}

impl<P, T> From<MultiColorExtractor<P, T>> for MultiColorExtractorParameters<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(f: MultiColorExtractor<P, T>) -> Self {
        Self {
            features: f.features,
        }
    }
}

impl<P, T> From<MultiColorExtractorParameters<P, T>> for MultiColorExtractor<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(p: MultiColorExtractorParameters<P, T>) -> Self {
        Self::new(p.features)
    }
}

impl<P, T> JsonSchema for MultiColorExtractor<P, T>
where
    P: PassbandTrait,
    T: Float,
    MultiColorFeature<P, T>: JsonSchema,
{
    json_schema!(MultiColorExtractorParameters<P, T>, true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multicolor::features::{ColorOfMaximum, ColorOfMinimum};
    use crate::{MultiColorTimeSeries, StringPassband};

    type McExtractor = MultiColorExtractor<StringPassband, f64>;

    #[test]
    fn extractor_combines_features() {
        let passbands = [StringPassband::from("g"), StringPassband::from("r")];
        let features = vec![
            ColorOfMaximum::new(passbands.clone()).into(),
            ColorOfMinimum::new(passbands.clone()).into(),
        ];
        let extractor = McExtractor::new(features);

        assert_eq!(extractor.size_hint(), 2);
        assert_eq!(
            extractor.get_names(),
            vec!["color_max_g_r", "color_min_g_r"]
        );

        // g band: [4.0, 5.0, 6.0] max=6, min=4; r band: [1.0, 3.0, 2.0] max=3, min=1
        let t = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let m = vec![4.0_f64, 5.0, 6.0, 1.0, 3.0, 2.0];
        let w = vec![1.0_f64; 6];
        let bands: Vec<StringPassband> = vec!["g", "g", "g", "r", "r", "r"]
            .into_iter()
            .map(StringPassband::from)
            .collect();
        let mut mcts = MultiColorTimeSeries::from_flat(t, m, w, bands);
        let result = extractor.eval_multicolor(&mut mcts).unwrap();
        assert!((result[0] - (6.0 - 3.0)).abs() < 1e-10); // color_max_g_r
        assert!((result[1] - (4.0 - 1.0)).abs() < 1e-10); // color_min_g_r
    }

    #[test]
    fn extractor_empty() {
        let extractor = McExtractor::new(vec![]);
        assert_eq!(extractor.size_hint(), 0);
        assert_eq!(extractor.get_names(), Vec::<&str>::new());
    }

    #[test]
    fn extractor_serde() {
        let passbands = [StringPassband::from("g"), StringPassband::from("r")];
        let features = vec![ColorOfMaximum::new(passbands).into()];
        let extractor = McExtractor::new(features);
        let json = serde_json::to_string(&extractor).unwrap();
        let extractor2: McExtractor = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string(&extractor2).unwrap());
    }
}
