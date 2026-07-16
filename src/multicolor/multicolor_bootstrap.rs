use crate::data::{MultiColorTimeSeries, TimeSeries};
use crate::error::{EvaluatorError, MultiColorEvaluatorError};
use crate::evaluator::{
    EvaluatorInfo, EvaluatorInfoTrait, EvaluatorProperties, FeatureNamesDescriptionsTrait,
};
use crate::features::bootstrap::{
    BootstrapFeatureError, BootstrapUncertainty, aggregate_bootstrap,
};
use crate::float_trait::Float;
use crate::multicolor::multicolor_evaluator::*;
use crate::multicolor::{MultiColorExtractor, MultiColorFeature, PassbandSet, PassbandTrait};

use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::fmt::Debug;

/// Per-passband resample buffers: `(t, m, w)` value columns, keyed by a borrowed passband.
type BandBuffers<'p, P, T> = BTreeMap<&'p P, (Vec<T>, Vec<T>, Vec<T>)>;

/// Per-band resampling strategy for [MultiColorBootstrap].
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BandStrategy {
    /// Resample each passband independently with replacement, preserving every per-band count.
    ///
    /// This is a *stratified* bootstrap: it conditions on the per-band sample sizes, so it does
    /// not capture the variance from the random allocation of observations across bands. By
    /// construction every resample preserves per-band validity, so none is ever rejected.
    #[default]
    Stratified,
    /// Classical i.i.d. bootstrap: resample all observations jointly with replacement.
    ///
    /// Resamples that fail the wrapped features' requirements (e.g. a band falling below its
    /// minimum) are rejected (via `check_mcts`) and redrawn, up to a bounded budget of
    /// `n_bootstrap * max_attempts_factor` total draws — there is no unbounded loop. If the budget
    /// is exhausted the valid resamples collected so far are used; if fewer than two valid
    /// resamples are obtained the feature returns an error (handled by `eval_or_fill` like any
    /// feature), never `NaN`.
    Rejection { max_attempts_factor: usize },
}

impl BandStrategy {
    #[inline]
    pub fn default_max_attempts_factor() -> usize {
        100
    }
}

/// Multi-color [Bootstrap](crate::Bootstrap): estimates multi-color feature uncertainties by
/// bagging the multi-band light curve. See [BandStrategy] for the per-band resampling options.
///
/// As in the single-band case, sub-features that require sorting, or that require variability,
/// are rejected by [MultiColorBootstrap::add_feature].
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    into = "MultiColorBootstrapParameters<P, T>",
    try_from = "MultiColorBootstrapParameters<P, T>",
    bound(
        serialize = "P: PassbandTrait + Serialize, T: Float",
        deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float"
    )
)]
pub struct MultiColorBootstrap<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    n_bootstrap: usize,
    seed: u64,
    uncertainty: BootstrapUncertainty,
    band_strategy: BandStrategy,
    feature_extractor: MultiColorExtractor<P, T>,
    properties: Box<EvaluatorProperties>,
}

impl<P, T> MultiColorBootstrap<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    pub fn new(
        n_bootstrap: usize,
        seed: u64,
        uncertainty: BootstrapUncertainty,
        band_strategy: BandStrategy,
    ) -> Self {
        assert!(n_bootstrap >= 2, "n_bootstrap must be at least 2");
        uncertainty.validate();
        if let BandStrategy::Rejection {
            max_attempts_factor,
        } = &band_strategy
        {
            assert!(
                *max_attempts_factor >= 1,
                "max_attempts_factor must be at least 1"
            );
        }
        Self {
            n_bootstrap,
            seed,
            uncertainty,
            band_strategy,
            feature_extractor: MultiColorExtractor::new(vec![]),
            properties: EvaluatorProperties {
                info: EvaluatorInfo {
                    size: 0,
                    min_ts_length: 1,
                    t_required: false,
                    m_required: false,
                    w_required: false,
                    sorting_required: false,
                    variability_required: false,
                },
                names: vec![],
                descriptions: vec![],
            }
            .into(),
        }
    }

    /// Add a multi-color feature to estimate the bootstrap uncertainty of.
    ///
    /// # Errors
    /// Returns [BootstrapFeatureError] if the feature requires sorting (bagging duplicates
    /// points, which is ill-defined for time-interval-dividing features and biases
    /// consecutive-difference-based features) or requires variability (a resample may be
    /// constant).
    pub fn add_feature(
        &mut self,
        feature: MultiColorFeature<P, T>,
    ) -> Result<&mut Self, BootstrapFeatureError> {
        if feature.is_sorting_required() {
            return Err(BootstrapFeatureError::SortingRequired);
        }
        if feature.is_variability_required() {
            return Err(BootstrapFeatureError::VariabilityRequired);
        }

        let multiplier = 1 + self.uncertainty.len();
        self.properties.info.size += feature.size_hint() * multiplier;
        self.properties.info.min_ts_length =
            usize::max(self.properties.info.min_ts_length, feature.min_ts_length());
        self.properties.info.t_required |= feature.is_t_required();
        self.properties.info.m_required |= feature.is_m_required();
        self.properties.info.w_required |= feature.is_w_required();
        self.properties.info.sorting_required |= feature.is_sorting_required();

        for name in feature.get_names() {
            let names = self.uncertainty.value_and_uncertainty_names(name);
            self.properties.names.extend(names);
        }
        for desc in feature.get_descriptions() {
            let descriptions = self.uncertainty.value_and_uncertainty_descriptions(desc);
            self.properties.descriptions.extend(descriptions);
        }
        self.feature_extractor.add_feature(feature);
        Ok(self)
    }

    #[inline]
    pub fn default_n_bootstrap() -> usize {
        100
    }

    #[inline]
    pub fn default_seed() -> u64 {
        0
    }

    /// Build a `passband -> TimeSeries` map from per-band resample buffers.
    ///
    /// Bands absent from a resample (empty buffers) are dropped rather than inserted empty, so
    /// `check_mcts` sees them as missing (matching the previous flat-based behaviour). For
    /// `Stratified` every group is always non-empty, so the filter is a no-op there.
    fn collect_map<'p, 'b>(buffers: &'b BandBuffers<'p, P, T>) -> BTreeMap<P, TimeSeries<'b, T>> {
        buffers
            .iter()
            .filter(|(_, (t_buf, _, _))| !t_buf.is_empty())
            .map(|(&band, (t_buf, m_buf, w_buf))| {
                (
                    band.clone(),
                    TimeSeries::new(&t_buf[..], &m_buf[..], &w_buf[..]),
                )
            })
            .collect()
    }
}

impl<P, T> EvaluatorInfoTrait for MultiColorBootstrap<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}

impl<P, T> FeatureNamesDescriptionsTrait for MultiColorBootstrap<P, T>
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

impl<P, T> MultiColorPassbandSetTrait<P> for MultiColorBootstrap<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn get_passband_set(&self) -> &PassbandSet<P> {
        self.feature_extractor.get_passband_set()
    }
}

impl<P, T> MultiColorEvaluator<P, T> for MultiColorBootstrap<P, T>
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
        // Feature values on the original light curve.
        let original = self.feature_extractor.eval_multicolor_no_mcts_check(mcts)?;
        let inner_size = original.len();

        // Read directly from the mapping representation: zero passband clones and zero t/m/w
        // copies for the source data (most multi-color features use the mapping representation
        // anyway, so this is at worst a no-op conversion, not an extra one).
        mcts.with_mapping_mut(|_| {});
        let mapping = mcts.mapping().expect("mapping was just ensured");
        let n: usize = mapping.iter_ts().map(TimeSeries::lenu).sum();

        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut columns: Vec<Vec<T>> = vec![Vec::with_capacity(self.n_bootstrap); inner_size];

        // Per-band resample buffers, reused across iterations (cleared and refilled by index,
        // no per-resample array allocation). Building the resample via `from_map` below clones
        // each passband once per resample per present band (K clones), not once per observation
        // (n clones) as the flat representation would require — `P` may be expensive to clone.
        // No sub-feature requires sorting (add_feature rejects those), so rows needn't be
        // time-ordered.
        let mut band_buffers: BandBuffers<P, T> = mapping
            .iter()
            .map(|(&band, ts)| {
                let len = ts.lenu();
                (
                    band,
                    (
                        Vec::with_capacity(len),
                        Vec::with_capacity(len),
                        Vec::with_capacity(len),
                    ),
                )
            })
            .collect();

        let n_resamples = match &self.band_strategy {
            BandStrategy::Stratified => {
                for _ in 0..self.n_bootstrap {
                    for (&band, ts) in mapping.iter() {
                        let (t_buf, m_buf, w_buf) =
                            band_buffers.get_mut(band).expect("band was pre-populated");
                        t_buf.clear();
                        m_buf.clear();
                        w_buf.clear();
                        let len = ts.lenu();
                        for _ in 0..len {
                            let i = rng.random_range(0..len);
                            t_buf.push(ts.t.sample[i]);
                            m_buf.push(ts.m.sample[i]);
                            w_buf.push(ts.w.sample[i]);
                        }
                    }
                    let map = Self::collect_map(&band_buffers);
                    let mut resample = MultiColorTimeSeries::from_map(map);
                    // Per-band counts are preserved, so validity is guaranteed; skip the check.
                    let values = self
                        .feature_extractor
                        .eval_multicolor_no_mcts_check(&mut resample)?;
                    for (column, &value) in columns.iter_mut().zip(&values) {
                        column.push(value);
                    }
                }
                self.n_bootstrap
            }
            BandStrategy::Rejection {
                max_attempts_factor,
            } => {
                let budget = self.n_bootstrap.saturating_mul(*max_attempts_factor);
                let mut collected = 0usize;
                let mut attempts = 0usize;
                while collected < self.n_bootstrap && attempts < budget {
                    attempts += 1;
                    for (t_buf, m_buf, w_buf) in band_buffers.values_mut() {
                        t_buf.clear();
                        m_buf.clear();
                        w_buf.clear();
                    }
                    for _ in 0..n {
                        // Draw a point uniformly across all bands: pick a global index, then
                        // locate which band it falls into (K is small, so a linear scan is fine).
                        let mut i = rng.random_range(0..n);
                        for (&band, ts) in mapping.iter() {
                            let len = ts.lenu();
                            if i < len {
                                let (t_buf, m_buf, w_buf) =
                                    band_buffers.get_mut(band).expect("band was pre-populated");
                                t_buf.push(ts.t.sample[i]);
                                m_buf.push(ts.m.sample[i]);
                                w_buf.push(ts.w.sample[i]);
                                break;
                            }
                            i -= len;
                        }
                    }
                    let map = Self::collect_map(&band_buffers);
                    let mut resample = MultiColorTimeSeries::from_map(map);
                    // Use the framework's validity check as the rejection criterion.
                    if self.feature_extractor.check_mcts(&mut resample).is_err() {
                        continue;
                    }
                    let values = self
                        .feature_extractor
                        .eval_multicolor_no_mcts_check(&mut resample)?;
                    for (column, &value) in columns.iter_mut().zip(&values) {
                        column.push(value);
                    }
                    collected += 1;
                }
                collected
            }
        };

        // Like any feature, return an error when the uncertainty cannot be estimated (too few
        // valid resamples). The caller's `eval_or_fill` then substitutes its fill value.
        if inner_size > 0 && n_resamples < 2 {
            return Err(EvaluatorError::InsufficientResamples {
                actual: n_resamples,
                minimum: 2,
            }
            .into());
        }

        Ok(aggregate_bootstrap(
            &original,
            &mut columns,
            &self.uncertainty,
        ))
    }
}

#[derive(Clone, Serialize, Deserialize, JsonSchema)]
#[serde(
    rename = "MultiColorBootstrap",
    bound(
        serialize = "P: PassbandTrait + Serialize, T: Float",
        deserialize = "P: PassbandTrait + Deserialize<'de>, T: Float"
    )
)]
struct MultiColorBootstrapParameters<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    n_bootstrap: usize,
    seed: u64,
    uncertainty: BootstrapUncertainty,
    band_strategy: BandStrategy,
    feature_extractor: MultiColorExtractor<P, T>,
}

impl<P, T> From<MultiColorBootstrap<P, T>> for MultiColorBootstrapParameters<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(b: MultiColorBootstrap<P, T>) -> Self {
        Self {
            n_bootstrap: b.n_bootstrap,
            seed: b.seed,
            uncertainty: b.uncertainty,
            band_strategy: b.band_strategy,
            feature_extractor: b.feature_extractor,
        }
    }
}

impl<P, T> TryFrom<MultiColorBootstrapParameters<P, T>> for MultiColorBootstrap<P, T>
where
    P: PassbandTrait,
    T: Float,
{
    type Error = BootstrapFeatureError;

    fn try_from(p: MultiColorBootstrapParameters<P, T>) -> Result<Self, Self::Error> {
        let mut bootstrap = Self::new(p.n_bootstrap, p.seed, p.uncertainty, p.band_strategy);
        for feature in p.feature_extractor.get_features().iter().cloned() {
            bootstrap.add_feature(feature)?;
        }
        Ok(bootstrap)
    }
}

impl<P, T> JsonSchema for MultiColorBootstrap<P, T>
where
    P: PassbandTrait + JsonSchema,
    T: Float,
{
    json_schema!(MultiColorBootstrapParameters<P, T>, false);
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

    fn color_of_maximum() -> MultiColorFeature<StringPassband, f64> {
        ColorOfMaximum::new([StringPassband::from("g"), StringPassband::from("r")]).into()
    }

    #[test]
    fn stratified_size_value_and_names() {
        let mut b =
            MultiColorBootstrap::new(50, 0, BootstrapUncertainty::Std, BandStrategy::Stratified);
        b.add_feature(color_of_maximum()).unwrap();
        assert_eq!(b.size_hint(), 2); // value + sigma
        assert_eq!(
            b.get_names(),
            &["bootstrap_color_max_g_r", "bootstrap_color_max_g_r_sigma"]
        );

        let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0];
        let m_g = [1.0_f64, 3.0, 2.0, 5.0, 4.0];
        let m_r = [2.0_f64, 4.0, 6.0, 1.0, 3.0];
        let w = [1.0_f64; 5];
        let mut mcts = make_mcts(&t, &m_g, &m_r, &w);
        let out = b.eval_multicolor(&mut mcts).unwrap();

        // Original value matches the plain color feature (max_g - max_r = 5 - 6 = -1).
        let plain_feature = color_of_maximum();
        let mut plain_mcts = make_mcts(&t, &m_g, &m_r, &w);
        let plain = plain_feature.eval_multicolor(&mut plain_mcts).unwrap()[0];
        assert!((out[0] - plain).abs() < 1e-12);
        assert!(out[1].is_finite() && out[1] >= 0.0);
    }

    #[test]
    fn rejection_runs_and_is_bounded() {
        let mut b = MultiColorBootstrap::new(
            30,
            1,
            BootstrapUncertainty::quantiles([0.16, 0.84]),
            BandStrategy::Rejection {
                max_attempts_factor: 100,
            },
        );
        b.add_feature(color_of_maximum()).unwrap();
        assert_eq!(b.size_hint(), 3); // value + 2 quantiles

        let t = [0.0_f64, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let m_g = [1.0_f64, 3.0, 2.0, 5.0, 4.0, 6.0, 0.0, 7.0];
        let m_r = [2.0_f64, 4.0, 6.0, 1.0, 3.0, 5.0, 8.0, 2.0];
        let w = [1.0_f64; 8];
        let mut mcts = make_mcts(&t, &m_g, &m_r, &w);
        let out = b.eval_multicolor(&mut mcts).unwrap();
        assert_eq!(out.len(), 3);
        assert!(out[1] <= out[2]); // q16 <= q84
        assert!(out[1].is_finite() && out[2].is_finite());
    }

    #[test]
    fn serde_roundtrip() {
        let mut b = MultiColorBootstrap::<StringPassband, f64>::new(
            10,
            0,
            BootstrapUncertainty::Std,
            BandStrategy::Rejection {
                max_attempts_factor: 50,
            },
        );
        b.add_feature(color_of_maximum()).unwrap();
        let json = serde_json::to_string(&b).unwrap();
        let b2: MultiColorBootstrap<StringPassband, f64> = serde_json::from_str(&json).unwrap();
        assert_eq!(b.get_names(), b2.get_names());
    }
}
