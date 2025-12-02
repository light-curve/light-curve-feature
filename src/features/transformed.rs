use crate::evaluator::*;
use crate::transformers::TransformerTrait;

use std::hash::Hash;
use std::marker::PhantomData;
use thiserror::Error;

#[derive(Error, Debug)]
pub enum TransformedConstructionError {
    #[error("Size mismatch between feature output and supported transformer input")]
    SizeMismatch,
}

macro_const! {
    const DOC: &str = r"
Feature extractor transforming output of other feature extractors

- Depends on: what underlying feature depends on
- Minimum number of observations: what underlying feature requires
- Number of features: a combination of underlying feature and transformer
#";
}

#[doc = DOC!()]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(
    into = "TransformedParameters<F, Tr>",
    try_from = "TransformedParameters<F, Tr>",
    bound = "T: Float, F: FeatureEvaluator<T>, Tr: TransformerTrait<T>"
)]
pub struct Transformed<T, F, Tr> {
    // We need to store the feature in a box to avoid a recursive type in `Feature`
    feature: Box<F>,
    transformer: Tr,
    properties: Box<EvaluatorProperties>,
    phantom: PhantomData<T>,
}

impl<T, F, Tr> Transformed<T, F, Tr>
where
    T: Float,
    F: FeatureEvaluator<T>,
    Tr: TransformerTrait<T>,
{
    pub fn new(feature: F, transformer: Tr) -> Result<Self, TransformedConstructionError> {
        if !transformer.is_size_valid(feature.size_hint()) {
            return Err(TransformedConstructionError::SizeMismatch);
        }
        let info = EvaluatorInfo {
            size: transformer.size_hint(feature.size_hint()),
            min_ts_length: feature.min_ts_length(),
            t_required: feature.is_t_required(),
            m_required: feature.is_m_required(),
            w_required: feature.is_w_required(),
            sorting_required: feature.is_sorting_required(),
        };
        let names = transformer.names(&feature.get_names());
        let descriptions = transformer.descriptions(&feature.get_descriptions());
        let properties = EvaluatorProperties {
            info,
            names,
            descriptions,
        }
        .into();
        Ok(Self {
            feature: feature.into(),
            transformer,
            properties,
            phantom: PhantomData,
        })
    }

    pub const fn doc() -> &'static str {
        DOC!()
    }
}

impl<T, F, Tr> EvaluatorInfoTrait for Transformed<T, F, Tr>
where
    T: Float,
    F: FeatureEvaluator<T>,
    Tr: TransformerTrait<T>,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}

impl<T, F, Tr> FeatureNamesDescriptionsTrait for Transformed<T, F, Tr>
where
    T: Float,
    F: FeatureEvaluator<T>,
    Tr: TransformerTrait<T>,
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

impl<T, F, Tr> FeatureEvaluator<T> for Transformed<T, F, Tr>
where
    T: Float,
    F: FeatureEvaluator<T>,
    Tr: TransformerTrait<T>,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        Ok(self.transformer.transform(self.feature.eval(ts)?))
    }

    // We keep default implementation of eval_or_fill

    fn check_ts_length(&self, ts: &TimeSeries<T>) -> Result<usize, EvaluatorError> {
        self.feature.check_ts_length(ts)
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Transformed")]
struct TransformedParameters<F, Tr> {
    feature: F,
    transformer: Tr,
}

impl<T, F, Tr> From<Transformed<T, F, Tr>> for TransformedParameters<F, Tr> {
    fn from(f: Transformed<T, F, Tr>) -> Self {
        Self {
            feature: *f.feature,
            transformer: f.transformer,
        }
    }
}

impl<T, F, Tr> TryFrom<TransformedParameters<F, Tr>> for Transformed<T, F, Tr>
where
    T: Float,
    F: FeatureEvaluator<T>,
    Tr: TransformerTrait<T>,
{
    type Error = TransformedConstructionError;

    fn try_from(p: TransformedParameters<F, Tr>) -> Result<Self, Self::Error> {
        Self::new(p.feature, p.transformer)
    }
}

impl<T, F, Tr> JsonSchema for Transformed<T, F, Tr>
where
    T: Float,
    F: FeatureEvaluator<T>,
    Tr: TransformerTrait<T>,
{
    json_schema!(TransformedParameters<F, Tr>, false);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::features::{Amplitude, BazinFit, Cusum, Kurtosis, LinexpFit, ReducedChi2};
    use crate::tests::*;
    use crate::transformers::{
        Transformer, arcsinh::ArcsinhTransformer, bazin_fit::BazinFitTransformer,
        identity::IdentityTransformer, lg::LgTransformer, linexp_fit::LinexpFitTransformer,
        ln1p::Ln1pTransformer,
    };

    eval_info_test!(
        info_default_transformed_reduced_chi2,
        Transformed::new(ReducedChi2::new().into(), Ln1pTransformer::new().into()).unwrap()
    );

    eval_info_test!(
        info_default_transformed_bazin_fit,
        Transformed::new(
            BazinFit::default().into(),
            BazinFitTransformer::default().into()
        )
        .unwrap()
    );

    eval_info_test!(
        info_default_transformed_linexp_fit,
        Transformed::new(
            LinexpFit::default().into(),
            LinexpFitTransformer::default().into()
        )
        .unwrap()
    );

    serialization_name_test!(
        Transformed<f64, Feature<f64>, Transformer<f64>>,
        Transformed::<f64, Feature<f64>, Transformer<f64>>::new(
            Kurtosis::new().into(),
            ArcsinhTransformer::new().into()
        ).unwrap()
    );

    serde_json_test!(
        ser_json_de,
        Transformed<f64, Feature<f64>, Transformer<f64>>,
        Transformed::new(Amplitude::new().into(), LgTransformer::new().into()).unwrap()
    );

    check_doc_static_method!(
        doc_static_method,
        Transformed::<f64, Feature<f64>, Transformer<f64>>
    );

    check_finite!(
        check_values_finite_identity_transformer,
        Transformed::<f64, Feature<f64>, Transformer<f64>>::new(
            Cusum::new().into(),
            IdentityTransformer::new().into()
        )
        .unwrap()
    );
}
