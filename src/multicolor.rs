use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{
    EvaluatorError, EvaluatorInfo, EvaluatorInfoTrait, EvaluatorProperties, FeatureEvaluator,
    FeatureNamesDescriptionsTrait,
};
use crate::feature::Feature;
use crate::float_trait::Float;

use enum_dispatch::enum_dispatch;
use itertools::Itertools;
pub use lazy_static::lazy_static;
pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;
use std::marker::PhantomData;

pub trait PassbandTrait: Debug + Clone + Send + Sync + Ord + Serialize + JsonSchema {
    fn name(&self) -> &str;
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct MonochromePassband<'a, T> {
    pub name: &'a str,
    pub wavelength: T,
}

impl<'a, T> MonochromePassband<'a, T>
where
    T: Float,
{
    pub fn new(wavelength: T, name: &'a str) -> Self {
        assert!(
            wavelength.is_normal(),
            "wavelength must be a positive normal number"
        );
        assert!(
            wavelength.is_sign_positive(),
            "wavelength must be a positive normal number"
        );
        Self { wavelength, name }
    }
}

impl<'a, T> PartialEq for MonochromePassband<'a, T>
where
    T: Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.wavelength.eq(&other.wavelength)
    }
}

impl<'a, T> Eq for MonochromePassband<'a, T> where T: Float {}

impl<'a, T> PartialOrd for MonochromePassband<'a, T>
where
    T: Float,
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        (self.wavelength).partial_cmp(&other.wavelength)
    }
}

impl<'a, T> Ord for MonochromePassband<'a, T>
where
    T: Float,
{
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<'a, T> PassbandTrait for MonochromePassband<'a, T>
where
    T: Float,
{
    fn name(&self) -> &str {
        self.name
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct NoPassband {}

impl PassbandTrait for NoPassband {
    fn name(&self) -> &str {
        ""
    }
}

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
    /// Vector of feature values or `EvaluatorError`
    fn eval_multicolor(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError>;

    /// Returns vector of feature values and fill invalid components with given value
    fn eval_or_fill_multicolor(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
        fill_value: T,
    ) -> Result<Vec<T>, MultiColorEvaluatorError> {
        self.check_mcts_shape(mcts)?;
        Ok(match self.eval_multicolor(mcts) {
            Ok(v) => v,
            Err(_) => vec![fill_value; self.size_hint()],
        })
    }

    fn check_mcts_shape(
        &self,
        mcts: &MultiColorTimeSeries<P, T>,
    ) -> Result<BTreeMap<P, usize>, MultiColorEvaluatorError> {
        self.check_mcts_passabands(mcts)?;
        self.check_every_ts_length(mcts)
    }

    fn check_mcts_passabands(
        &self,
        mcts: &MultiColorTimeSeries<P, T>,
    ) -> Result<(), MultiColorEvaluatorError> {
        match self.get_passband_set() {
            PassbandSet::AllAvailable => Ok(()),
            PassbandSet::FixedSet(self_passbands) => {
                if mcts
                    .keys()
                    .all(|mcts_passband| self_passbands.contains(mcts_passband))
                {
                    Ok(())
                } else {
                    Err(MultiColorEvaluatorError::wrong_passbands_error(
                        mcts.keys(),
                        self_passbands.iter(),
                    ))
                }
            }
        }
    }

    /// Checks if each component of [MultiColorTimeSeries] has enough points to evaluate the feature
    fn check_every_ts_length(
        &self,
        mcts: &MultiColorTimeSeries<P, T>,
    ) -> Result<BTreeMap<P, usize>, MultiColorEvaluatorError> {
        // Use try_reduce when stabilizes
        // https://github.com/rust-lang/rust/issues/87053
        mcts.iter()
            .map(|(passband, ts)| {
                let length = ts.lenu();
                if length < self.min_ts_length() {
                    Err(MultiColorEvaluatorError::MonochromeEvaluatorError {
                        error: EvaluatorError::ShortTimeSeries {
                            actual: length,
                            minimum: self.min_ts_length(),
                        },
                        passband: passband.name().into(),
                    })
                } else {
                    Ok((passband.clone(), length))
                }
            })
            .collect()
    }
}

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
    fn eval_multicolor(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError> {
        self.check_mcts_passabands(mcts)?;
        let mut vec = Vec::with_capacity(self.size_hint());
        for x in &self.features {
            vec.extend(x.eval_multicolor(mcts)?);
        }
        Ok(vec)
    }

    fn eval_or_fill_multicolor(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
        fill_value: T,
    ) -> Result<Vec<T>, MultiColorEvaluatorError> {
        self.check_mcts_passabands(mcts)?;
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
    fn eval_multicolor(
        &self,
        mcts: &mut MultiColorTimeSeries<P, T>,
    ) -> Result<Vec<T>, MultiColorEvaluatorError> {
        self.check_mcts_passabands(mcts)?;
        match &self.passband_set {
            PassbandSet::FixedSet(set) => set
                .iter()
                .map(|passband| {
                    self.feature.eval(mcts.get_mut(passband).expect(
                        "we checked all needed passbands are in mcts, but we still cannot find one",
                    )).map_err(|error| MultiColorEvaluatorError::MonochromeEvaluatorError {
                        passband: passband.name().into(),
                        error,
                    })
                })
                .flatten_ok()
                .collect(),
            PassbandSet::AllAvailable => panic!("passband_set must be FixedSet variant here"),
        }
    }
}

#[enum_dispatch(MultiColorEvaluator<P, T>, FeatureNamesDescriptionsTrait, EvaluatorInfoTrait, MultiColorPassbandSetTrait<P>)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound(deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float"))]
#[non_exhaustive]
pub enum MultiColorFeature<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    // Extractor
    MultiColorExtractor(MultiColorExtractor<P, T, MultiColorFeature<P, T>>),
    // Monochrome Features
    MonochromeFeature(MonochromeFeature<P, T, Feature<T>>),
    // Features
    ColorOfMedian(color_median::ColorOfMedian<P>),
}

impl<P, T> MultiColorFeature<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    pub fn from_monochrome_feature<F>(feature: F, passband_set: BTreeSet<P>) -> Self
    where
        F: Into<Feature<T>>,
    {
        MonochromeFeature::new(feature.into(), passband_set).into()
    }
}

/// Example of multicolor light-curve feature evaluator
mod color_median {
    use super::*;
    use crate::{FeatureEvaluator, Median};

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
        COLOR_MEDIAN_INFO,
        size: 1,
        min_ts_length: 1,
        t_required: false,
        m_required: true,
        w_required: false,
        sorting_required: false,
    );

    impl<P> EvaluatorInfoTrait for ColorOfMedian<P>
    where
        P: Ord,
    {
        fn get_info(&self) -> &EvaluatorInfo {
            &COLOR_MEDIAN_INFO
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
        fn eval_multicolor(
            &self,
            mcts: &mut MultiColorTimeSeries<P, T>,
        ) -> Result<Vec<T>, MultiColorEvaluatorError> {
            self.check_mcts_passabands(mcts)?;
            let mut medians = [T::zero(); 2];
            for (median, passband) in medians.iter_mut().zip(self.passbands.iter()) {
                *median = self
                    .median
                    .eval(mcts.get_mut(passband).expect(
                        "we checked all needed passbands are in mcts, but we still cannot find one",
                    ))
                    .map_err(|error| MultiColorEvaluatorError::MonochromeEvaluatorError {
                        passband: passband.name().into(),
                        error,
                    })?[0]
            }
            Ok(vec![medians[0] - medians[1]])
        }
    }
}
