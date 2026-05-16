use crate::data::{MultiColorTimeSeries, TimeSeries};
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{
    EvaluatorInfo, EvaluatorInfoTrait, EvaluatorProperties, FeatureNamesDescriptionsTrait,
    TmwArrays,
};
use crate::features::bins::bin_time_series;
use crate::float_trait::Float;
use crate::multicolor::multicolor_evaluator::*;
use crate::multicolor::{MultiColorExtractor, PassbandSet, PassbandTrait};

use conv::ConvUtil;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Debug;

use super::MultiColorFeature;

/// Multi-color meta-feature that bins each passband's time series independently,
/// then evaluates inner multi-color features on the collection of binned per-band series.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    into = "MultiColorBinsParameters<P, T>",
    from = "MultiColorBinsParameters<P, T>",
    bound(
        serialize = "P: PassbandTrait + Serialize, T: Float",
        deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float"
    )
)]
pub struct MultiColorBins<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    window: f64,
    offset: f64,
    feature_extractor: MultiColorExtractor<P, T>,
    properties: Box<EvaluatorProperties>,
}

impl<P, T> MultiColorBins<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    pub fn new(window: f64, offset: f64) -> Self {
        assert!(window > 0.0, "window must be positive");
        assert!(
            window.is_finite() && window.abs() <= f32::MAX as f64,
            "window value {} is out of range for f32",
            window
        );
        assert!(
            offset.is_finite() && offset.abs() <= f32::MAX as f64,
            "offset value {} is out of range for f32",
            offset
        );
        Self {
            window,
            offset,
            feature_extractor: MultiColorExtractor::new(vec![]),
            properties: EvaluatorProperties {
                info: EvaluatorInfo {
                    size: 0,
                    min_ts_length: 1,
                    t_required: true,
                    m_required: true,
                    w_required: true,
                    sorting_required: true,
                    variability_required: false,
                },
                names: vec![],
                descriptions: vec![],
            }
            .into(),
        }
    }

    pub fn add_feature(&mut self, feature: MultiColorFeature<P, T>) -> &mut Self {
        let window = self.window;
        let offset = self.offset;
        self.properties.info.size += feature.size_hint();
        self.properties.info.min_ts_length = self
            .properties
            .info
            .min_ts_length
            .max(feature.min_ts_length());
        self.properties.info.variability_required |= feature.is_variability_required();
        self.properties.names.extend(
            feature
                .get_names()
                .iter()
                .map(|name| format!("bins_window{window:.1}_offset{offset:.1}_{name}")),
        );
        self.properties
            .descriptions
            .extend(feature.get_descriptions().iter().map(|desc| {
                format!("{desc} for binned time-series with window {window} and offset {offset}")
            }));
        self.feature_extractor.add_feature(feature);
        self
    }

    pub fn default_window() -> f64 {
        1.0
    }

    pub fn default_offset() -> f64 {
        0.0
    }
}

impl<P, T> EvaluatorInfoTrait for MultiColorBins<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}

impl<P, T> FeatureNamesDescriptionsTrait for MultiColorBins<P, T>
where
    P: PassbandTrait,
    T: Float,
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

impl<P, T> MultiColorPassbandSetTrait<P> for MultiColorBins<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn get_passband_set(&self) -> &PassbandSet<P> {
        self.feature_extractor.get_passband_set()
    }
}

impl<P, T> MultiColorEvaluator<P, T> for MultiColorBins<P, T>
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
        let window: T = self.window.approx_as::<T>().unwrap();
        let offset: T = self.offset.approx_as::<T>().unwrap();

        let PassbandSet(set) = self.feature_extractor.get_passband_set();

        let binned_arrays: BTreeMap<P, TmwArrays<T>> = mcts
            .mapping_mut()
            .iter_matched_passbands_mut(set.iter())
            .map(|(passband, maybe_ts)| {
                let ts = maybe_ts.expect("passband checked before eval_multicolor_no_mcts_check");
                let tmw = bin_time_series(ts, window, offset).map_err(|error| {
                    MultiColorEvaluatorError::MonochromeEvaluatorError {
                        passband: passband.name().into(),
                        error,
                    }
                })?;
                Ok((passband.clone(), tmw))
            })
            .collect::<Result<_, MultiColorEvaluatorError>>()?;

        let binned_map: BTreeMap<P, TimeSeries<'_, T>> = binned_arrays
            .iter()
            .map(|(p, tmw)| {
                (
                    p.clone(),
                    TimeSeries::new(tmw.t.view(), tmw.m.view(), tmw.w.view()),
                )
            })
            .collect();

        let mut binned_mcts = MultiColorTimeSeries::from_map(binned_map);
        self.feature_extractor
            .eval_multicolor_no_mcts_check(&mut binned_mcts)
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(
    rename = "MultiColorBins",
    bound(
        serialize = "P: PassbandTrait + Serialize, T: Float",
        deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float"
    )
)]
struct MultiColorBinsParameters<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    window: f64,
    offset: f64,
    feature_extractor: MultiColorExtractor<P, T>,
}

impl<P, T> From<MultiColorBins<P, T>> for MultiColorBinsParameters<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(b: MultiColorBins<P, T>) -> Self {
        Self {
            window: b.window,
            offset: b.offset,
            feature_extractor: b.feature_extractor,
        }
    }
}

impl<P, T> From<MultiColorBinsParameters<P, T>> for MultiColorBins<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(p: MultiColorBinsParameters<P, T>) -> Self {
        let mut bins = Self::new(p.window, p.offset);
        p.feature_extractor
            .get_features()
            .iter()
            .cloned()
            .for_each(|f| {
                bins.add_feature(f);
            });
        bins
    }
}

impl<P, T> JsonSchema for MultiColorBins<P, T>
where
    P: PassbandTrait + JsonSchema,
    T: Float,
{
    json_schema!(MultiColorBinsParameters<P, T>, false);
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::data::TimeSeries;
    use crate::multicolor::features::ColorOfMaximum;
    use crate::{MultiColorTimeSeries, StringPassband};

    use std::collections::BTreeMap;

    fn make_mcts<'a>(
        t: &'a [f64],
        m_g: &'a [f64],
        m_r: &'a [f64],
        w: &'a [f64],
    ) -> MultiColorTimeSeries<'a, StringPassband, f64> {
        let mut map = BTreeMap::new();
        map.insert(StringPassband::from("g"), TimeSeries::new(t, m_g, w));
        map.insert(StringPassband::from("r"), TimeSeries::new(t, m_r, w));
        MultiColorTimeSeries::from_map(map)
    }

    // g-band: t=[0.0, 0.1, 1.0, 1.1, 2.0], m=[1,3,5,7,9], w=1
    // r-band: same t, m=[2,4,6,8,10], w=1
    // binned (window=1, offset=0):
    //   bin [0,1): g->mean(1,3)=2; r->mean(2,4)=3
    //   bin [1,2): g->mean(5,7)=6; r->mean(6,8)=7
    //   bin [2,3): g->9;           r->10
    // ColorOfMaximum(g,r): max_g=9, max_r=10 → 9-10=-1
    #[test]
    fn multicolor_bins_values() {
        let t = [0.0_f64, 0.1, 1.0, 1.1, 2.0];
        let m_g = [1.0_f64, 3.0, 5.0, 7.0, 9.0];
        let m_r = [2.0_f64, 4.0, 6.0, 8.0, 10.0];
        let w = [1.0_f64; 5];

        let mut eval = MultiColorBins::new(1.0, 0.0);
        eval.add_feature(
            ColorOfMaximum::new([StringPassband::from("g"), StringPassband::from("r")]).into(),
        );
        let mut mcts = make_mcts(&t, &m_g, &m_r, &w);
        let result = eval.eval_multicolor(&mut mcts).unwrap();
        assert_eq!(result.len(), 1);
        assert!((result[0] - (-1.0_f64)).abs() < 1e-10, "got {}", result[0]);
    }

    #[test]
    fn multicolor_bins_names() {
        let mut eval = MultiColorBins::<StringPassband, f64>::new(1.0, 0.0);
        eval.add_feature(
            ColorOfMaximum::new([StringPassband::from("g"), StringPassband::from("r")]).into(),
        );
        assert_eq!(
            eval.get_names(),
            vec!["bins_window1.0_offset0.0_color_max_g_r"]
        );
    }

    #[test]
    fn multicolor_bins_serde() {
        let mut eval = MultiColorBins::<StringPassband, f64>::new(1.0, 0.0);
        eval.add_feature(
            ColorOfMaximum::new([StringPassband::from("g"), StringPassband::from("r")]).into(),
        );
        let json = serde_json::to_string(&eval).unwrap();
        let eval2: MultiColorBins<StringPassband, f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(eval.get_names(), eval2.get_names());
    }
}
