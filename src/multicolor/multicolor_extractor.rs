use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{EvaluatorInfoTrait, FeatureNamesDescriptionsTrait};
use crate::float_trait::Float;
use crate::multicolor::multicolor_evaluator::*;

use itertools::Itertools;
pub use schemars::JsonSchema;
pub use serde::Serialize;
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Bulk feature evaluator.
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
    /// Create a new [MultiColorExtractor]
    ///
    /// # Arguments
    /// `features` - A vector of multi-color features to be evaluated
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::multicolor::features::{ColorOfMaximum, ColorOfMinimum};
    use crate::{MultiColorFeature, MultiColorTimeSeries, StringPassband};

    type McFeature = MultiColorFeature<StringPassband, f64>;
    type McExtractor = MultiColorExtractor<StringPassband, f64, McFeature>;

    #[test]
    fn extractor_combines_features() {
        let passbands = [StringPassband::from("g"), StringPassband::from("r")];
        let features: Vec<McFeature> = vec![
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
        let features: Vec<McFeature> = vec![ColorOfMaximum::new(passbands).into()];
        let extractor = McExtractor::new(features);
        let json = serde_json::to_string(&extractor).unwrap();
        let extractor2: McExtractor = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string(&extractor2).unwrap());
    }
}
