use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{
    EvaluatorInfoTrait, EvaluatorProperties, FeatureEvaluator, FeatureNamesDescriptionsTrait,
};
use crate::float_trait::Float;
use crate::multicolor::multicolor_evaluator::*;

use itertools::Itertools;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::fmt::Debug;
use std::marker::PhantomData;

/// Multi-color feature which evaluates a monochrome feature independently for each passband.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound(
    deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float, F: FeatureEvaluator<T>"
))]
pub struct PerBandFeature<P, T, F>
where
    P: Ord,
{
    feature: F,
    passband_set: PassbandSet<P>,
    properties: Box<EvaluatorProperties>,
    phantom: PhantomData<T>,
}

impl<P, T, F> PerBandFeature<P, T, F>
where
    P: PassbandTrait,
    T: Float,
    F: FeatureEvaluator<T>,
{
    /// Creates a new instance of `PerBandFeature`.
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

impl<P, T, F> FeatureNamesDescriptionsTrait for PerBandFeature<P, T, F>
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

impl<P, T, F> EvaluatorInfoTrait for PerBandFeature<P, T, F>
where
    P: Ord,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}

impl<P, T, F> MultiColorPassbandSetTrait<P> for PerBandFeature<P, T, F>
where
    P: PassbandTrait,
{
    fn get_passband_set(&self) -> &PassbandSet<P> {
        &self.passband_set
    }
}

impl<P, T, F> MultiColorEvaluator<P, T> for PerBandFeature<P, T, F>
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
        let PassbandSet::FixedSet(set) = &self.passband_set;
        mcts.mapping_mut()
            .iter_matched_passbands_mut(set.iter())
            .map(|(passband, ts)| {
                self.feature
                    .eval_no_ts_check(ts.expect(
                        "we checked all needed passbands are in mcts, but we still cannot find one",
                    ))
                    .map_err(|error| MultiColorEvaluatorError::MonochromeEvaluatorError {
                        passband: passband.name().into(),
                        error,
                    })
            })
            .flatten_ok()
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::Feature;
    use crate::MultiColorTimeSeries;
    use crate::StringPassband;
    use crate::features::Mean;
    use crate::multicolor::multicolor_evaluator::MultiColorEvaluator;
    use crate::multicolor::passband::MonochromePassband;

    #[test]
    fn test_per_band_feature() {
        let feature: PerBandFeature<MonochromePassband<_>, f64, Feature<_>> = PerBandFeature::new(
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

    #[test]
    fn test_per_band_feature_values() {
        use crate::data::TimeSeries;
        use std::collections::BTreeMap;

        let passband_g = MonochromePassband::new(4700e-8, "g");
        let passband_r = MonochromePassband::new(6200e-8, "r");

        let t = vec![0.0_f64, 1.0, 2.0];
        let m_g = vec![1.0_f64, 2.0, 3.0]; // mean = 2.0
        let m_r = vec![4.0_f64, 5.0, 6.0]; // mean = 5.0

        let mut mcts = {
            let mut map = BTreeMap::new();
            map.insert(passband_g.clone(), TimeSeries::new_without_weight(&t, &m_g));
            map.insert(passband_r.clone(), TimeSeries::new_without_weight(&t, &m_r));
            MultiColorTimeSeries::from_map(map)
        };

        let feature: PerBandFeature<MonochromePassband<_>, f64, Feature<_>> = PerBandFeature::new(
            Mean::default().into(),
            [passband_g.clone(), passband_r.clone()]
                .into_iter()
                .collect(),
        );

        let result = feature.eval_multicolor(&mut mcts).unwrap();
        // Passbands are ordered by wavelength (g before r), so mean_g=2.0 comes first
        assert_eq!(result, vec![2.0_f64, 5.0_f64]);
    }

    #[test]
    fn test_per_band_feature_subsamples_bands() {
        // Data has g, r, i but feature only requests g and r — i band must be ignored.
        let t = vec![0.0_f64; 6];
        let m = vec![1.0, 2.0, 4.0, 5.0, 10.0, 20.0];
        let w = vec![1.0_f64; 6];
        let bands: Vec<StringPassband> = ["g", "g", "r", "r", "i", "i"]
            .iter()
            .map(|&s| StringPassband::from(s))
            .collect();
        let mut mcts = MultiColorTimeSeries::from_flat(t, m, w, bands);

        let feature: PerBandFeature<StringPassband, f64, Feature<f64>> = PerBandFeature::new(
            Mean::default().into(),
            ["g", "r"]
                .iter()
                .map(|&s| StringPassband::from(s))
                .collect(),
        );
        let result = feature.eval_multicolor(&mut mcts).unwrap();
        // mean_g = 1.5, mean_r = 4.5; i band ignored
        assert!((result[0] - 1.5).abs() < 1e-10);
        assert!((result[1] - 4.5).abs() < 1e-10);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_per_band_feature_serde() {
        let feature: PerBandFeature<StringPassband, f64, Feature<f64>> = PerBandFeature::new(
            Mean::default().into(),
            ["g", "r"]
                .iter()
                .map(|&s| StringPassband::from(s))
                .collect(),
        );
        let json = serde_json::to_string(&feature).unwrap();
        let feature2: PerBandFeature<StringPassband, f64, Feature<f64>> =
            serde_json::from_str(&json).unwrap();
        assert_eq!(feature.get_names(), feature2.get_names());
        assert_eq!(feature.get_descriptions(), feature2.get_descriptions());
    }
}
