use crate::evaluator::*;
use crate::extractor::FeatureExtractor;

use conv::ConvUtil;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

macro_const! {
    const DOC: &str = r"
Bootstrap uncertainty meta-feature

Estimates the uncertainty of feature values by bagging: it draws ``n_bootstrap`` resamples of the
light curve (sampling observations *with replacement*, keeping the original length) and evaluates
the wrapped feature(s) on each resample. For every wrapped feature value it returns the value on
the original light curve followed by an uncertainty summary over the resamples — either the sample
standard deviation or a set of quantiles (see [BootstrapUncertainty]).

[Bootstrap::add_feature] rejects features that cannot be evaluated on a resample:

* features requiring **both** time and time-sorting (they divide by time intervals, e.g.
  ``EtaE``, ``MaximumSlope``, ``LinearFit``, the curve fits) — bagging produces duplicate
  timestamps;
* features requiring **variability** (e.g. ``StetsonK``, ``Skew``, ``Kurtosis``, ``Eta``,
  ``Cusum``) — a resample may be constant.

Distribution features (``Amplitude``, ``StandardDeviation``, percentiles, the robust scale
estimators, …) and time-value-only features (``TimeMean``) are supported.

- Depends on: as required by sub-features
- Minimum number of observations: as required by sub-features, but at least **1**
- Number of features: ``(1 + n_uncertainty)`` per sub-feature value, where ``n_uncertainty`` is 1
  for the standard deviation or the number of quantile levels
";
}

/// How [Bootstrap] summarizes the spread of a feature over the resamples.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Default)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BootstrapUncertainty {
    /// Sample standard deviation (``ddof = 1``).
    #[default]
    Std,
    /// Quantile levels, each in ``[0, 1]`` (e.g. ``[0.16, 0.84]`` for a 1σ-like interval).
    Quantiles { levels: Vec<f32> },
}

impl BootstrapUncertainty {
    /// Number of uncertainty values produced per wrapped feature value.
    pub(crate) fn len(&self) -> usize {
        match self {
            Self::Std => 1,
            Self::Quantiles { levels } => levels.len(),
        }
    }

    /// Output names for one wrapped feature value named `base`: the value followed by its
    /// uncertainty descriptor(s). Shared by the single-band and multi-color meta-features.
    pub(crate) fn value_and_uncertainty_names(&self, base: &str) -> Vec<String> {
        let mut names = vec![format!("bootstrap_{base}")];
        match self {
            Self::Std => names.push(format!("bootstrap_{base}_sigma")),
            Self::Quantiles { levels } => {
                names.extend(
                    levels
                        .iter()
                        .map(|q| format!("bootstrap_{base}_quantile_{q}")),
                );
            }
        }
        names
    }

    /// Output descriptions matching [Self::value_and_uncertainty_names].
    pub(crate) fn value_and_uncertainty_descriptions(&self, base: &str) -> Vec<String> {
        let mut descriptions = vec![base.to_string()];
        match self {
            Self::Std => descriptions.push(format!("bootstrap standard deviation of {base}")),
            Self::Quantiles { levels } => {
                descriptions.extend(
                    levels
                        .iter()
                        .map(|q| format!("bootstrap {q} quantile of {base}")),
                );
            }
        }
        descriptions
    }

    pub(crate) fn validate(&self) {
        if let Self::Quantiles { levels } = self {
            assert!(!levels.is_empty(), "quantile levels must not be empty");
            for &q in levels {
                assert!(
                    q.is_finite() && (0.0..=1.0).contains(&q),
                    "quantile level {q} must be in [0, 1]"
                );
            }
        }
    }
}

#[doc = DOC!()]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(
    into = "BootstrapParameters<T, F>",
    from = "BootstrapParameters<T, F>",
    bound(deserialize = "T: Float, F: FeatureEvaluator<T>")
)]
pub struct Bootstrap<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    n_bootstrap: usize,
    seed: u64,
    uncertainty: BootstrapUncertainty,
    feature_extractor: FeatureExtractor<T, F>,
    properties: Box<EvaluatorProperties>,
}

impl<T, F> Bootstrap<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    pub fn new(n_bootstrap: usize, seed: u64, uncertainty: BootstrapUncertainty) -> Self {
        assert!(n_bootstrap >= 2, "n_bootstrap must be at least 2");
        uncertainty.validate();
        let info = EvaluatorInfo {
            size: 0,
            min_ts_length: 1,
            t_required: false,
            m_required: false,
            w_required: false,
            sorting_required: false,
            variability_required: false,
        };
        Self {
            properties: EvaluatorProperties {
                info,
                names: vec![],
                descriptions: vec![],
            }
            .into(),
            n_bootstrap,
            seed,
            uncertainty,
            feature_extractor: FeatureExtractor::new(vec![]),
        }
    }

    /// Add a feature to estimate the bootstrap uncertainty of.
    ///
    /// # Panics
    /// * If the feature requires both time and sorting: bagging produces duplicate timestamps,
    ///   which is ill-defined for features that divide by time intervals.
    /// * If the feature requires variability: a resample may be constant.
    pub fn add_feature(&mut self, feature: F) -> &mut Self {
        assert!(
            !(feature.is_t_required() && feature.is_sorting_required()),
            "Bootstrap cannot wrap a feature that requires both time and sorting \
             (it divides by time intervals, but bagging produces duplicate timestamps)"
        );
        assert!(
            !feature.is_variability_required(),
            "Bootstrap cannot wrap a feature that requires variability \
             (a resample may be constant)"
        );

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
        self
    }

    #[inline]
    pub fn default_n_bootstrap() -> usize {
        100
    }

    #[inline]
    pub fn default_seed() -> u64 {
        0
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

/// Sample standard deviation (`ddof = 1`) of finite values; `NaN` for fewer than two.
fn sample_std<T: Float>(values: &[T]) -> T {
    let n = values.len();
    if n < 2 {
        return T::nan();
    }
    let nf = n.approx_as::<T>().unwrap();
    let mean = values.iter().copied().sum::<T>() / nf;
    let var = values.iter().map(|&v| (v - mean).powi(2)).sum::<T>() / (nf - T::one());
    var.sqrt()
}

/// Combine per-value resample columns into the flat `[value, uncertainty…]` output, shared by
/// the single-band and multi-color bootstrap meta-features. `columns[j]` holds the resample
/// values for `original[j]` and is sorted in place for the quantile case.
pub(crate) fn aggregate_bootstrap<T: Float>(
    original: &[T],
    columns: &mut [Vec<T>],
    uncertainty: &BootstrapUncertainty,
) -> Vec<T> {
    let mut output = Vec::with_capacity(original.len() * (1 + uncertainty.len()));
    for (&value, column) in original.iter().zip(columns.iter_mut()) {
        output.push(value);
        match uncertainty {
            BootstrapUncertainty::Std => output.push(sample_std(column)),
            BootstrapUncertainty::Quantiles { levels } => {
                column.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
                for &q in levels {
                    output.push(quantile_sorted(column, q));
                }
            }
        }
    }
    output
}

/// Linear-interpolated quantile of `values` (already sorted ascending); `NaN` if empty.
fn quantile_sorted<T: Float>(values: &[T], q: f32) -> T {
    let n = values.len();
    if n == 0 {
        return T::nan();
    }
    if n == 1 {
        return values[0];
    }
    let pos = f64::from(q) * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    let frac = (pos - lo as f64).approx_as::<T>().unwrap();
    values[lo] + (values[hi] - values[lo]) * frac
}

impl<T, F> Default for Bootstrap<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn default() -> Self {
        Self::new(
            Self::default_n_bootstrap(),
            Self::default_seed(),
            BootstrapUncertainty::default(),
        )
    }
}

impl<T, F> EvaluatorInfoTrait for Bootstrap<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}

impl<T, F> FeatureNamesDescriptionsTrait for Bootstrap<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
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

impl<T, F> FeatureEvaluator<T> for Bootstrap<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        // Feature values on the original light curve.
        let original = self.feature_extractor.eval(ts)?;
        let inner_size = original.len();

        let n = ts.lenu();
        let t = ts.t.as_slice();
        let m = ts.m.as_slice();
        let w = ts.w.as_slice();

        let mut rng = StdRng::seed_from_u64(self.seed);

        // Per-inner-value resample results. By construction (no variability-, dt-requiring
        // sub-features) every resample evaluates successfully, so no values are dropped.
        let mut columns: Vec<Vec<T>> = vec![Vec::with_capacity(self.n_bootstrap); inner_size];
        let mut triples: Vec<(T, T, T)> = Vec::with_capacity(n);
        let (mut rt, mut rm, mut rw) = (vec![T::zero(); n], vec![T::zero(); n], vec![T::zero(); n]);

        for _ in 0..self.n_bootstrap {
            // Bagging: draw n observations with replacement, then sort by time.
            triples.clear();
            triples.extend((0..n).map(|_| {
                let i = rng.random_range(0..n);
                (t[i], m[i], w[i])
            }));
            triples.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            for (k, &(tt, mm, ww)) in triples.iter().enumerate() {
                rt[k] = tt;
                rm[k] = mm;
                rw[k] = ww;
            }

            let mut resample = TimeSeries::new(&rt, &rm, &rw);
            let values = self.feature_extractor.eval(&mut resample)?;
            for (column, &value) in columns.iter_mut().zip(&values) {
                column.push(value);
            }
        }

        Ok(aggregate_bootstrap(
            &original,
            &mut columns,
            &self.uncertainty,
        ))
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Bootstrap", bound = "T: Float, F: FeatureEvaluator<T>")]
struct BootstrapParameters<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    n_bootstrap: usize,
    seed: u64,
    uncertainty: BootstrapUncertainty,
    feature_extractor: FeatureExtractor<T, F>,
}

impl<T, F> From<Bootstrap<T, F>> for BootstrapParameters<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn from(f: Bootstrap<T, F>) -> Self {
        Self {
            n_bootstrap: f.n_bootstrap,
            seed: f.seed,
            uncertainty: f.uncertainty,
            feature_extractor: f.feature_extractor,
        }
    }
}

impl<T, F> From<BootstrapParameters<T, F>> for Bootstrap<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn from(p: BootstrapParameters<T, F>) -> Self {
        let mut bootstrap = Self::new(p.n_bootstrap, p.seed, p.uncertainty);
        for feature in p.feature_extractor.get_features().iter().cloned() {
            bootstrap.add_feature(feature);
        }
        bootstrap
    }
}

impl<T, F> JsonSchema for Bootstrap<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    json_schema!(BootstrapParameters<T, F>, false);
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::features::{Amplitude, EtaE, Skew, StandardDeviation};
    use crate::tests::*;

    serialization_name_test!(Bootstrap<f64, Feature<f64>>);

    serde_json_test!(
        bootstrap_ser_json_de,
        Bootstrap<f64, Feature<f64>>,
        {
            let mut b = Bootstrap::default();
            b.add_feature(Amplitude::default().into());
            b
        },
    );

    check_doc_static_method!(bootstrap_doc_static_method, Bootstrap<f64, Feature<f64>>);

    check_finite!(check_values_finite, {
        let mut b: Bootstrap<_, Feature<_>> = Bootstrap::default();
        b.add_feature(Amplitude::default().into());
        b.add_feature(StandardDeviation::default().into());
        b
    });

    #[test]
    fn size_and_value_std() {
        let mut b: Bootstrap<f64, Feature<f64>> = Bootstrap::default();
        b.add_feature(Amplitude::default().into());
        b.add_feature(StandardDeviation::default().into());
        assert_eq!(b.size_hint(), 4); // (value + sigma) x 2 features
        assert_eq!(
            b.get_names(),
            &[
                "bootstrap_amplitude",
                "bootstrap_amplitude_sigma",
                "bootstrap_standard_deviation",
                "bootstrap_standard_deviation_sigma"
            ]
        );

        let mut rng = StdRng::seed_from_u64(0);
        let m: Vec<f64> = (0..100).map(|_| rng.random_range(10.0..20.0)).collect();
        let t: Vec<f64> = (0..100).map(f64::from).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let out = b.eval(&mut ts).unwrap();

        // Original values match the plain features.
        let amplitude = Amplitude::default().eval(&mut ts).unwrap()[0];
        let std = StandardDeviation::default().eval(&mut ts).unwrap()[0];
        assert!((out[0] - amplitude).abs() < 1e-12);
        assert!((out[2] - std).abs() < 1e-12);
        // Sigmas are finite and positive.
        assert!(out[1] > 0.0 && out[1].is_finite());
        assert!(out[3] > 0.0 && out[3].is_finite());
    }

    #[test]
    fn quantiles_layout() {
        let mut b: Bootstrap<f64, Feature<f64>> = Bootstrap::new(
            200,
            0,
            BootstrapUncertainty::Quantiles {
                levels: vec![0.16, 0.84],
            },
        );
        b.add_feature(Amplitude::default().into());
        assert_eq!(b.size_hint(), 3); // value + 2 quantiles
        assert_eq!(
            b.get_names(),
            &[
                "bootstrap_amplitude",
                "bootstrap_amplitude_quantile_0.16",
                "bootstrap_amplitude_quantile_0.84"
            ]
        );

        let mut rng = StdRng::seed_from_u64(1);
        let m: Vec<f64> = (0..100).map(|_| rng.random_range(10.0..20.0)).collect();
        let t: Vec<f64> = (0..100).map(f64::from).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let out = b.eval(&mut ts).unwrap();
        // q16 <= q84
        assert!(out[1] <= out[2]);
        assert!(out[1].is_finite() && out[2].is_finite());
    }

    #[test]
    #[should_panic(expected = "both time and sorting")]
    fn rejects_time_and_sorting_feature() {
        let mut b: Bootstrap<f64, Feature<f64>> = Bootstrap::default();
        b.add_feature(EtaE::default().into());
    }

    #[test]
    #[should_panic(expected = "variability")]
    fn rejects_variability_feature() {
        // Skew requires variability (a resample may be constant).
        let mut b: Bootstrap<f64, Feature<f64>> = Bootstrap::default();
        b.add_feature(Skew::default().into());
    }
}
