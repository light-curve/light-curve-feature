use crate::evaluator::*;
use crate::extractor::FeatureExtractor;

use conv::ConvUtil;
use itertools::Itertools;
use ordered_float::NotNan;
use std::hash::Hash;
use unzip3::Unzip3;

macro_const! {
    const DOC: &str = r"
Sampled time series meta-feature

Binning time series to bins with width $\mathrm{window}$ with respect to some $\mathrm{offset}$.
$j-th$ bin interval is
$[j \cdot \mathrm{window} + \mathrm{offset}; (j + 1) \cdot \mathrm{window} + \mathrm{offset})$.
Binned time series is defined by
$$
t_j^* = (j + \frac12) \cdot \mathrm{window} + \mathrm{offset},
$$
$$
m_j^* = \frac{\sum{m_i / \delta_i^2}}{\sum{\delta_i^{-2}}},
$$
$$
\delta_j^* = \frac{N_j}{\sum{\delta_i^{-2}}},
$$
where $N_j$ is a number of sampling observations and all sums are over observations inside
considering bin. Bins takes any other feature evaluators to extract features from sample time series

- Depends on: **time**, **magnitude**, **magnitude error**
- Minimum number of observations: as required by sub-features, but at least **1**
- Number of features: as provided by sub-features
";
}

#[doc = DOC!()]
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(
    into = "BinsParameters<T, F>",
    from = "BinsParameters<T, F>",
    bound(deserialize = "T: Float, F: FeatureEvaluator<T>")
)]
pub struct Bins<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    window: NotNan<f64>,
    offset: NotNan<f64>,
    feature_extractor: FeatureExtractor<T, F>,
    properties: Box<EvaluatorProperties>,
}

impl<T, F> Hash for Bins<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.window.hash(state);
        self.offset.hash(state);
        self.feature_extractor.hash(state);
        self.properties.hash(state);
    }
}

impl<T, F> Bins<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    pub fn new(window: f64, offset: f64) -> Self {
        assert!(window > 0.0, "window must be positive");
        // Validate that window and offset can be converted to f32 without overflow
        // since T could be f32 or f64
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
        let window = NotNan::new(window).expect("window must not be NaN");
        let offset = NotNan::new(offset).expect("offset must not be NaN");
        let info = EvaluatorInfo {
            size: 0,
            min_ts_length: 1,
            t_required: true,
            m_required: true,
            w_required: true,
            sorting_required: true,
        };
        Self {
            properties: EvaluatorProperties {
                info,
                names: vec![],
                descriptions: vec![],
            }
            .into(),
            window,
            offset,
            feature_extractor: FeatureExtractor::new(vec![]),
        }
    }

    pub fn set_window(&mut self, window: f64) -> &mut Self {
        assert!(window > 0.0, "window must be positive");
        assert!(
            window.is_finite() && window.abs() <= f32::MAX as f64,
            "window value {} is out of range for f32",
            window
        );
        self.window = NotNan::new(window).expect("window must not be NaN");
        self
    }

    pub fn set_offset(&mut self, offset: f64) -> &mut Self {
        assert!(
            offset.is_finite() && offset.abs() <= f32::MAX as f64,
            "offset value {} is out of range for f32",
            offset
        );
        self.offset = NotNan::new(offset).expect("offset must not be NaN");
        self
    }

    /// Extend a feature to extract from binned time series
    pub fn add_feature(&mut self, feature: F) -> &mut Self {
        let window = self.window.into_inner();
        let offset = self.offset.into_inner();
        self.properties.info.size += feature.size_hint();
        self.properties.info.min_ts_length =
            usize::max(self.properties.info.min_ts_length, feature.min_ts_length());
        self.properties.names.extend(
            feature
                .get_names()
                .iter()
                .map(|name| format!("bins_window{window:.1}_offset{offset:.1}_{name}")),
        );
        self.properties
            .descriptions
            .extend(feature.get_descriptions().iter().map(|desc| {
                format!("{desc} for binned time-series with window {window} and offset {offset}",)
            }));
        self.feature_extractor.add_feature(feature);
        self
    }

    #[inline]
    pub fn default_window() -> f64 {
        1.0
    }

    #[inline]
    pub fn default_offset() -> f64 {
        0.0
    }
}

impl<T, F> Bins<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    pub const fn doc() -> &'static str {
        DOC
    }

    fn transform_ts(&self, ts: &mut TimeSeries<T>) -> Result<TmwArrays<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        // These conversions should never fail because we validated the range in new() and set methods
        let window = self.window.into_inner().approx_as::<T>().unwrap();
        let offset = self.offset.into_inner().approx_as::<T>().unwrap();
        let (t, m, w): (Vec<_>, Vec<_>, Vec<_>) =
            ts.t.as_slice()
                .iter()
                .copied()
                .zip(ts.m.as_slice().iter().copied())
                .zip(ts.w.as_slice().iter().copied())
                .map(|((t, m), w)| (t, m, w))
                .chunk_by(|(t, _, _)| ((*t - offset) / window).floor())
                .into_iter()
                .map(|(x, chunk)| {
                    let bin_t = (x + T::half()) * window;
                    let (n, bin_m, norm) = chunk
                        .fold((T::zero(), T::zero(), T::zero()), |acc, (_, m, w)| {
                            (acc.0 + T::one(), acc.1 + m * w, acc.2 + w)
                        });
                    let bin_m = bin_m / norm;
                    let bin_w = norm / n;
                    (bin_t, bin_m, bin_w)
                })
                .unzip3();
        Ok(TmwArrays {
            t: t.into(),
            m: m.into(),
            w: w.into(),
        })
    }
}

impl<T, F> Default for Bins<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn default() -> Self {
        Self::new(Self::default_window(), Self::default_offset())
    }
}

impl<T, F> EvaluatorInfoTrait for Bins<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}
impl<T, F> FeatureNamesDescriptionsTrait for Bins<T, F>
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

impl<T, F> FeatureEvaluator<T> for Bins<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    transformer_eval!();
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Bins", bound = "T: Float, F: FeatureEvaluator<T>")]
struct BinsParameters<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    window: f64,
    offset: f64,
    feature_extractor: FeatureExtractor<T, F>,
}

impl<T, F> From<Bins<T, F>> for BinsParameters<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn from(f: Bins<T, F>) -> Self {
        Self {
            window: f.window.into_inner(),
            offset: f.offset.into_inner(),
            feature_extractor: f.feature_extractor,
        }
    }
}

impl<T, F> From<BinsParameters<T, F>> for Bins<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    fn from(p: BinsParameters<T, F>) -> Self {
        let mut bins = Self::new(p.window, p.offset);
        p.feature_extractor
            .get_features()
            .iter()
            .cloned()
            .for_each(|feature| {
                bins.add_feature(feature);
            });
        bins
    }
}

impl<T, F> JsonSchema for Bins<T, F>
where
    T: Float,
    F: FeatureEvaluator<T>,
{
    json_schema!(BinsParameters<T, F>, false);
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::features::{Amplitude, EtaE, LinearFit};
    use crate::tests::*;

    serialization_name_test!(Bins<f64, Feature<f64>>);

    serde_json_test!(
        bins_ser_json_de,
        Bins<f64, Feature<f64>>,
        {
            let mut bins = Bins::default();
            bins.add_feature(Amplitude::default().into());
            bins
        },
    );

    eval_info_test!(bins_with_amplitude_info, {
        let mut bins = Bins::default();
        bins.add_feature(Amplitude::default().into());
        bins
    });

    #[test]
    fn bins_with_eta_e_info() {
        let eval = {
            let mut bins = Bins::new(1e-100, 0.0);
            bins.add_feature(EtaE::default().into());
            bins.into()
        };
        // Bins are tiny, so no actual binning happens and min_ts_length must be right
        // Wrong times must give wrong answer, so t_required must be checked
        eval_info_tests(
            eval,  // feature
            true,  // test_min_ts_length. Bins are tiny, so no actual binning happens
            true, // test_t_required. Times are essential for EtaE, wrong times must give wrong answers
            true, // m_required. EtaE needs magnitudes
            false, // test_w_required. EtaE doesn't need weights and no binning happens, so they are not used and test would fail
            true,  // test_sorting_required. Sorting is essential for binning
        );
    }

    #[test]
    fn bins_with_linear_fit_info() {
        let eval = {
            let mut bins = Bins::new(1.0, 0.0);
            bins.add_feature(LinearFit::default().into());
            bins.into()
        };
        // Bins are tiny, so no actual binning happens and min_ts_length must be right
        // Wrong times must give wrong answer, so t_required must be checked
        eval_info_tests(
            eval,  // feature
            false, // test_min_ts_length. Bins has significant size, so actual binning happens and min_ts_length must be lower limit
            true, // test_t_required. Times are essential for EtaE, wrong times must give wrong answers
            true, // m_required. EtaE needs magnitudes
            true, // test_w_required. EtaE doesn't need weights and no binning happens, so they are not used and test would fail
            true, // test_sorting_required. Sorting is essential for binning
        );
    }

    check_doc_static_method!(bins_doc_static_method, Bins<f64, Feature<f64>>);

    check_finite!(check_values_finite, {
        let mut bins: Bins<_, Feature<_>> = Bins::default();
        bins.add_feature(Amplitude::default().into());
        bins.add_feature(EtaE::default().into());
        bins
    });

    #[test]
    fn bins() {
        let t = [0.0_f32, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 5.0];
        let m = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let w = [10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0, 5.0, 10.0];
        let mut ts = TimeSeries::new(&t, &m, &w);

        let desired_t = [0.5, 1.5, 2.5, 5.5];
        let desired_m = [0.0, 2.0, 6.333333333333333, 10.0];
        let desired_w = [10.0, 6.666666666666667, 7.5, 10.0];

        let bins: Bins<_, Feature<_>> = Bins::new(1.0, 0.0);
        let actual_tmw = bins.transform_ts(&mut ts).unwrap();

        assert_eq!(actual_tmw.t.len(), actual_tmw.m.len());
        assert_eq!(actual_tmw.t.len(), actual_tmw.w.len());
        all_close(actual_tmw.t.as_slice().unwrap(), &desired_t, 1e-6);
        all_close(actual_tmw.m.as_slice().unwrap(), &desired_m, 1e-6);
        all_close(actual_tmw.w.as_slice().unwrap(), &desired_w, 1e-6);
    }

    #[test]
    fn bins_windows_and_offsets() {
        let t = [0.0_f32, 1.0, 1.1, 1.2, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 5.0];
        let m = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mut ts = TimeSeries::new_without_weight(&t, &m);

        let mut len = |window, offset| {
            let tmw = Bins::<_, Feature<_>>::new(window, offset)
                .transform_ts(&mut ts)
                .unwrap();
            assert_eq!(tmw.t.len(), tmw.m.len());
            assert_eq!(tmw.m.len(), tmw.w.len());
            tmw.t.len()
        };

        assert_eq!(len(2.0, 0.0), 3);
        assert_eq!(len(3.0, 0.0), 2);
        assert_eq!(len(10.0, 0.0), 1);
        assert_eq!(len(1.0, 0.1), 5);
        assert_eq!(len(1.0, 0.5), 5);
        assert_eq!(len(2.0, 1.0), 3);
    }

    // Add more Bins::get_info() tests for non-trivial cases
}
