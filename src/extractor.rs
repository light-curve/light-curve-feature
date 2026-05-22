use crate::data::TimeSeries;
use crate::error::EvaluatorError;
use crate::evaluator::*;
use crate::feature::Feature;
use crate::float_trait::Float;

use std::marker::PhantomData;

macro_const! {
    const DOC: &str = r#"
Bulk feature extractor

- Depends on: as reuired by feature evaluators
- Minimum number of observations: as required by feature evaluators
- Number of features: total for all feature evaluators
"#;
}

#[doc = DOC!()]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(
    into = "FeatureExtractorParameters<F>",
    from = "FeatureExtractorParameters<F>",
    bound = "T: Float, F: FeatureEvaluator<T>"
)]
pub struct FeatureExtractor<T, F> {
    features: Vec<F>,
    info: Box<EvaluatorInfo>,
    phantom: PhantomData<T>,
}

impl<T, F> FeatureExtractor<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    pub fn new(features: Vec<F>) -> Self {
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
            info,
            features,
            phantom: PhantomData,
        }
    }

    pub fn get_features(&self) -> &Vec<F> {
        &self.features
    }

    pub fn into_vec(self) -> Vec<F> {
        self.features
    }

    pub fn add_feature(&mut self, feature: F) {
        self.info.size += feature.size_hint();
        self.info.min_ts_length = self.info.min_ts_length.max(feature.min_ts_length());
        self.info.t_required |= feature.is_t_required();
        self.info.m_required |= feature.is_m_required();
        self.info.w_required |= feature.is_w_required();
        self.info.sorting_required |= feature.is_sorting_required();
        self.info.variability_required |= feature.is_variability_required();
        self.features.push(feature);
    }
}

impl<T> FeatureExtractor<T, Feature<T>>
where
    T: Float,
{
    /// Specialized version of [FeatureExtractor::new] for [Feature]
    pub fn from_features(features: Vec<Feature<T>>) -> Self {
        Self::new(features)
    }
}

impl<T, F> FeatureExtractor<T, F> {
    pub const fn doc() -> &'static str {
        DOC
    }
}

impl<T, F> EvaluatorInfoTrait for FeatureExtractor<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.info
    }
}

impl<T, F> FeatureNamesDescriptionsTrait for FeatureExtractor<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
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

impl<T, F> FeatureEvaluator<T> for FeatureExtractor<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        let mut vec = Vec::with_capacity(self.size_hint());
        for x in &self.features {
            vec.extend(x.eval_no_ts_check(ts)?);
        }
        Ok(vec)
    }

    fn eval_or_fill(&self, ts: &mut TimeSeries<T>, fill_value: T) -> Vec<T> {
        self.features
            .iter()
            .flat_map(|x| x.eval_or_fill(ts, fill_value))
            .collect()
    }
}

#[cfg(test)]
impl<T, F> Default for FeatureExtractor<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn default() -> Self {
        Self::new(vec![])
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "FeatureExtractor")]
struct FeatureExtractorParameters<F> {
    features: Vec<F>,
}

impl<T, F> From<FeatureExtractor<T, F>> for FeatureExtractorParameters<F> {
    fn from(f: FeatureExtractor<T, F>) -> Self {
        Self {
            features: f.features,
        }
    }
}

impl<T, F> From<FeatureExtractorParameters<F>> for FeatureExtractor<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn from(p: FeatureExtractorParameters<F>) -> Self {
        Self::new(p.features)
    }
}

impl<T, F> JsonSchema for FeatureExtractor<T, F>
where
    F: JsonSchema,
{
    json_schema!(FeatureExtractorParameters<F>, true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Feature;
    use crate::tests::*;

    use approx::assert_relative_eq;
    use serde_test::{Token, assert_ser_tokens};

    serialization_name_test!(FeatureExtractor<f64, Feature<f64>>);

    serde_json_test!(
        feature_extractor_ser_json_de,
        FeatureExtractor<f64, Feature<f64>>,
        FeatureExtractor::new(vec![crate::Amplitude{}.into(), crate::BeyondNStd::new(2.0).into()]),
    );

    check_doc_static_method!(feature_extractor_doc_static_method, FeatureExtractor<f64, Feature<f64>>);

    #[test]
    fn serialization_empty() {
        let fe: FeatureExtractor<f64, Feature<_>> = FeatureExtractor::new(vec![]);
        assert_ser_tokens(
            &fe,
            &[
                //
                Token::Struct {
                    len: 1,
                    name: "FeatureExtractor",
                },
                //
                Token::String("features"),
                Token::Seq { len: Some(0) },
                Token::SeqEnd,
                //
                Token::StructEnd,
            ],
        )
    }

    // Integration test: multiple features evaluated together produce correct concatenated values
    #[test]
    fn multi_feature_eval_values() {
        let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let m = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0_f64; 5];
        let mut ts = TimeSeries::new(&t[..], &m[..], &w[..]);

        let fe: FeatureExtractor<f64, Feature<f64>> = FeatureExtractor::new(vec![
            crate::Amplitude::new().into(),
            crate::Mean::new().into(),
        ]);

        let values = fe.eval(&mut ts).unwrap();
        assert_eq!(values.len(), 2, "should produce one value per feature");
        // Amplitude = (max - min) / 2 = (5 - 1) / 2 = 2.0
        assert_relative_eq!(values[0], 2.0, epsilon = 1e-10);
        // Mean = (1+2+3+4+5)/5 = 3.0
        assert_relative_eq!(values[1], 3.0, epsilon = 1e-10);
    }

    // Integration test: names and descriptions are correctly aggregated from all sub-features
    #[test]
    fn multi_feature_names_and_descriptions_aggregated() {
        let fe: FeatureExtractor<f64, Feature<f64>> = FeatureExtractor::new(vec![
            crate::Amplitude::new().into(),
            crate::Mean::new().into(),
            crate::StandardDeviation::new().into(),
        ]);

        let names = fe.get_names();
        let descs = fe.get_descriptions();

        assert_eq!(names.len(), 3);
        assert_eq!(descs.len(), 3);
        assert_eq!(fe.size_hint(), 3);
        // Names should be in the same order as features
        assert_eq!(names[0], "amplitude");
        assert_eq!(names[1], "mean");
        assert_eq!(names[2], "standard_deviation");
        // Descriptions must be non-empty strings
        assert!(descs.iter().all(|d| !d.is_empty()));
    }

    // Integration test: info flags are OR'd / max'd correctly when features are combined
    #[test]
    fn info_aggregated_correctly() {
        // Amplitude: t_required=false, sorting_required=false, min_ts_length=1, size=1
        // LinearTrend: t_required=true, sorting_required=true, min_ts_length=3, size=3
        let fe: FeatureExtractor<f64, Feature<f64>> = FeatureExtractor::new(vec![
            crate::Amplitude::new().into(),
            crate::LinearTrend::new().into(),
        ]);

        assert!(
            fe.is_t_required(),
            "t_required should be true when any feature requires it"
        );
        assert!(
            fe.is_sorting_required(),
            "sorting_required should be true when any feature requires it"
        );
        assert_eq!(
            fe.min_ts_length(),
            3,
            "min_ts_length should be the maximum across features"
        );
        assert_eq!(
            fe.size_hint(),
            1 + 3,
            "size should be the sum across features"
        );
    }

    // Integration test: add_feature correctly updates all info fields
    #[test]
    fn add_feature_updates_info_correctly() {
        let mut fe: FeatureExtractor<f64, Feature<f64>> =
            FeatureExtractor::new(vec![crate::Amplitude::new().into()]);

        assert_eq!(fe.size_hint(), 1);
        assert!(!fe.is_t_required());
        assert!(!fe.is_sorting_required());
        assert_eq!(fe.min_ts_length(), 1);

        fe.add_feature(crate::LinearTrend::new().into());

        assert_eq!(fe.size_hint(), 4);
        assert!(fe.is_t_required());
        assert!(fe.is_sorting_required());
        assert_eq!(fe.min_ts_length(), 3);
    }

    // Integration test: eval returns ShortTimeSeries when time series is too short
    #[test]
    fn eval_returns_error_on_short_ts() {
        // LinearTrend requires at least 3 points
        let fe: FeatureExtractor<f64, Feature<f64>> = FeatureExtractor::new(vec![
            crate::Amplitude::new().into(),
            crate::LinearTrend::new().into(),
        ]);

        let t = [0.0_f64, 1.0];
        let m = [1.0_f64, 2.0];
        let w = [1.0_f64, 1.0];
        let mut ts = TimeSeries::new(&t[..], &m[..], &w[..]);

        let result = fe.eval(&mut ts);
        assert!(
            matches!(
                result,
                Err(EvaluatorError::ShortTimeSeries {
                    actual: 2,
                    minimum: 3
                })
            ),
            "expected ShortTimeSeries error, got: {:?}",
            result
        );
    }

    // Integration test: eval_or_fill fills only the failing feature's outputs independently.
    // Each sub-feature in the extractor fails/fills independently, so features that succeed
    // return their values while those that fail return fill values.
    #[test]
    fn eval_or_fill_fills_only_failing_feature() {
        // Amplitude needs 1 point (succeeds), LinearTrend needs 3 points (fails on ts of len 2)
        let fe: FeatureExtractor<f64, Feature<f64>> = FeatureExtractor::new(vec![
            crate::Amplitude::new().into(),   // size 1, succeeds
            crate::LinearTrend::new().into(), // size 3, fails on short ts
        ]);

        let t = [0.0_f64, 1.0];
        let m = [1.0_f64, 3.0];
        let w = [1.0_f64, 1.0];
        let mut ts = TimeSeries::new(&t[..], &m[..], &w[..]);

        let values = fe.eval_or_fill(&mut ts, f64::NAN);
        assert_eq!(values.len(), 4, "should always return size_hint() values");
        // Amplitude succeeds: (3-1)/2 = 1.0
        assert_relative_eq!(values[0], 1.0, epsilon = 1e-10);
        // LinearTrend fails (short ts): all 3 outputs are fill value
        assert!(
            values[1..].iter().all(|v| v.is_nan()),
            "failed feature outputs should be fill value"
        );
    }

    // Integration test: eval_or_fill fills all values for a single feature that fails
    #[test]
    fn eval_or_fill_fills_all_on_single_failing_feature() {
        // OtsuSplit requires variability (variability_required=true)
        let fe: FeatureExtractor<f64, Feature<f64>> =
            FeatureExtractor::new(vec![crate::OtsuSplit::new().into()]);

        let t = [0.0_f64, 1.0, 2.0, 3.0];
        let m = [3.0_f64; 4];
        let w = [1.0_f64; 4];
        let mut ts = TimeSeries::new(&t[..], &m[..], &w[..]);

        let values = fe.eval_or_fill(&mut ts, -999.0);
        assert_eq!(values.len(), fe.size_hint());
        assert!(
            values.iter().all(|&v| v == -999.0),
            "all outputs should be fill value"
        );
    }

    // Integration test: eval_or_fill returns actual values on valid input (not fill)
    #[test]
    fn eval_or_fill_returns_values_on_valid_ts() {
        let fe: FeatureExtractor<f64, Feature<f64>> = FeatureExtractor::new(vec![
            crate::Amplitude::new().into(),
            crate::Mean::new().into(),
        ]);

        let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let m = [1.0_f64, 2.0, 3.0, 4.0, 5.0];
        let w = [1.0_f64; 5];
        let mut ts = TimeSeries::new(&t[..], &m[..], &w[..]);

        let values = fe.eval_or_fill(&mut ts, f64::NAN);
        assert_eq!(values.len(), 2);
        assert!(
            values.iter().all(|v| v.is_finite()),
            "values should be finite"
        );
    }

    // Integration test: eval result length always matches size_hint and names/descriptions length
    #[test]
    fn eval_result_length_consistent_with_size_hint() {
        let fe: FeatureExtractor<f64, Feature<f64>> = FeatureExtractor::new(vec![
            crate::Amplitude::new().into(),
            crate::LinearTrend::new().into(),
            crate::Mean::new().into(),
        ]);

        let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let m = [1.0_f64, 3.0, 2.0, 5.0, 4.0];
        let w = [1.0_f64; 5];
        let mut ts = TimeSeries::new(&t[..], &m[..], &w[..]);

        let values = fe.eval(&mut ts).unwrap();
        assert_eq!(values.len(), fe.size_hint());
        assert_eq!(values.len(), fe.get_names().len());
        assert_eq!(values.len(), fe.get_descriptions().len());
    }

    // Integration test: variability_required feature returns FlatTimeSeries error
    #[test]
    fn eval_returns_flat_ts_error_for_constant_magnitude() {
        // OtsuSplit requires variability (variability_required=true)
        let fe: FeatureExtractor<f64, Feature<f64>> =
            FeatureExtractor::new(vec![crate::OtsuSplit::new().into()]);

        let t = [0.0_f64, 1.0, 2.0, 3.0];
        let m = [3.0_f64; 4];
        let w = [1.0_f64; 4];
        let mut ts = TimeSeries::new(&t[..], &m[..], &w[..]);

        assert!(
            matches!(fe.eval(&mut ts), Err(EvaluatorError::FlatTimeSeries)),
            "expected FlatTimeSeries error for constant magnitude input"
        );
    }

    // Integration test: full pipeline evaluation on real-world-like data
    #[test]
    fn full_pipeline_on_realistic_data() {
        let mut rng = StdRng::seed_from_u64(42);
        let n = 50;
        let t: Vec<f64> = sorted(&randvec::<f64>(&mut rng, n))
            .into_iter()
            .enumerate()
            .map(|(i, _)| i as f64)
            .collect();
        let m = randvec::<f64>(&mut rng, n);
        let w = positive_randvec::<f64>(&mut rng, n);

        let fe: FeatureExtractor<f64, Feature<f64>> = FeatureExtractor::new(vec![
            crate::Amplitude::new().into(),
            crate::Mean::new().into(),
            crate::StandardDeviation::new().into(),
            crate::LinearTrend::new().into(),
            crate::MedianAbsoluteDeviation::new().into(),
        ]);

        let expected_size = fe.size_hint();
        let mut ts = TimeSeries::new(&t, &m, &w);
        let values = fe.eval(&mut ts).unwrap();

        assert_eq!(values.len(), expected_size);
        assert_eq!(values.len(), fe.get_names().len());
        assert!(
            values.iter().all(|v| v.is_finite()),
            "all pipeline values should be finite"
        );
    }
}
