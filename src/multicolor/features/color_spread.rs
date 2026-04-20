use crate::data::MultiColorTimeSeries;
use crate::error::MultiColorEvaluatorError;
use crate::evaluator::{EvaluatorInfo, EvaluatorInfoTrait, FeatureNamesDescriptionsTrait};
use crate::float_trait::Float;
use crate::multicolor::multicolor_evaluator::*;
use crate::multicolor::{PassbandSet, PassbandTrait};

use lazy_static::lazy_static;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Standard deviation of per-passband weighted mean magnitudes
///
/// For each passband $b$, the weighted mean magnitude is computed as
/// $$\mu_b = \frac{\sum_i w_i m_i}{\sum_i w_i},$$
/// where $w_i = 1/\sigma_i^2$ are the inverse-variance weights.
///
/// `ColorSpread` is the population standard deviation of these per-band means:
/// $$\text{ColorSpread} = \sqrt{\frac{1}{B}\sum_b \left(\mu_b - \bar\mu\right)^2},$$
/// where $\bar\mu = \frac{1}{B}\sum_b \mu_b$ and $B$ is the number of passbands.
///
/// A large value indicates a large spread of mean brightnesses across bands
/// (e.g. a red star bright in the infrared and faint in the blue).
/// A value of zero means all bands have the same mean magnitude.
///
/// The set of passbands to include must be specified at construction time via
/// [`ColorSpread::new`]. Input data is subsampled to the specified bands.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound(
    serialize = "P: PassbandTrait",
    deserialize = "P: PassbandTrait + Deserialize<'de>"
))]
pub struct ColorSpread<P>
where
    P: PassbandTrait,
{
    passband_set: PassbandSet<P>,
}

impl<P> ColorSpread<P>
where
    P: PassbandTrait,
{
    /// Create a new `ColorSpread` evaluator for the given set of passbands.
    ///
    /// Input [MultiColorTimeSeries](crate::data::MultiColorTimeSeries) is subsampled
    /// to the specified passbands when this feature is evaluated.
    pub fn new(passbands: impl IntoIterator<Item = P>) -> Self {
        Self {
            passband_set: PassbandSet::FixedSet(passbands.into_iter().collect()),
        }
    }
}

lazy_info!(
    COLOR_SPREAD_INFO,
    size: 1,
    min_ts_length: 1,
    t_required: false,
    m_required: true,
    w_required: true,
    sorting_required: false,
    variability_required: false,
);

impl<P> EvaluatorInfoTrait for ColorSpread<P>
where
    P: PassbandTrait,
{
    fn get_info(&self) -> &EvaluatorInfo {
        &COLOR_SPREAD_INFO
    }
}

impl<P> FeatureNamesDescriptionsTrait for ColorSpread<P>
where
    P: PassbandTrait,
{
    fn get_names(&self) -> Vec<&str> {
        vec!["color_spread"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["standard deviation of per-passband weighted mean magnitudes"]
    }
}

impl<P> MultiColorPassbandSetTrait<P> for ColorSpread<P>
where
    P: PassbandTrait,
{
    fn get_passband_set(&self) -> &PassbandSet<P> {
        &self.passband_set
    }
}

impl<P, T> MultiColorEvaluator<P, T> for ColorSpread<P>
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
        let PassbandSet::FixedSet(set) = &self.passband_set;
        let mapping = mcts.mapping_mut();

        // Compute weighted mean magnitude for each specified passband (in sorted order)
        let band_means: Vec<T> = set
            .iter()
            .map(|p| {
                let ts = mapping
                    .get_mut(p)
                    .expect("passband must be present after check_mcts");
                let m = ts.m.as_slice();
                let w = ts.w.as_slice();
                let sum_wm = m
                    .iter()
                    .zip(w.iter())
                    .fold(T::zero(), |acc, (&mi, &wi)| acc + wi * mi);
                let sum_w = w.iter().fold(T::zero(), |acc, &wi| acc + wi);
                sum_wm / sum_w
            })
            .collect();

        let n = T::from_usize(band_means.len()).expect("number of bands fits in float");
        let mean_of_means = band_means.iter().fold(T::zero(), |acc, &mu| acc + mu) / n;
        let variance = band_means.iter().fold(T::zero(), |acc, &mu| {
            let diff = mu - mean_of_means;
            acc + diff * diff
        }) / n;

        Ok(vec![variance.sqrt()])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MultiColorTimeSeries, StringPassband};
    use std::collections::BTreeSet;

    fn make_passbands(names: &[&str]) -> BTreeSet<StringPassband> {
        names.iter().map(|&s| StringPassband::from(s)).collect()
    }

    #[test]
    fn color_spread_values() {
        let eval = ColorSpread::new(make_passbands(&["g", "i", "r"]));
        // g band: [10.0, 12.0] w=[1.0, 1.0] -> mu_g = 11.0
        // i band: [14.0, 16.0] w=[1.0, 1.0] -> mu_i = 15.0
        // r band: [17.0, 19.0] w=[1.0, 1.0] -> mu_r = 18.0
        // mean_of_means = (11 + 15 + 18) / 3 = 44/3
        // variance = ((11 - 44/3)^2 + (15 - 44/3)^2 + (18 - 44/3)^2) / 3
        let t = vec![0.0_f64; 6];
        let m = vec![10.0, 12.0, 14.0, 16.0, 17.0, 19.0];
        let w = vec![1.0_f64; 6];
        let bands: Vec<StringPassband> = vec!["g", "g", "i", "i", "r", "r"]
            .into_iter()
            .map(StringPassband::from)
            .collect();
        let mut mcts = MultiColorTimeSeries::from_flat(t, m, w, bands);
        let result = eval.eval_multicolor(&mut mcts).unwrap();

        let mu_g = 11.0_f64;
        let mu_i = 15.0_f64;
        let mu_r = 18.0_f64;
        let mean_mu = (mu_g + mu_i + mu_r) / 3.0;
        let expected =
            (((mu_g - mean_mu).powi(2) + (mu_i - mean_mu).powi(2) + (mu_r - mean_mu).powi(2))
                / 3.0)
                .sqrt();
        assert!((result[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn color_spread_subsamples_bands() {
        // Data has g, i, r but we only request g and r — result should match two-band computation
        let eval = ColorSpread::new(make_passbands(&["g", "r"]));
        let t = vec![0.0_f64; 6];
        let m = vec![10.0, 12.0, 14.0, 16.0, 17.0, 19.0];
        let w = vec![1.0_f64; 6];
        let bands: Vec<StringPassband> = vec!["g", "g", "i", "i", "r", "r"]
            .into_iter()
            .map(StringPassband::from)
            .collect();
        let mut mcts = MultiColorTimeSeries::from_flat(t, m, w, bands);
        let result = eval.eval_multicolor(&mut mcts).unwrap();

        // Only g (mu=11) and r (mu=18) used
        let mu_g = 11.0_f64;
        let mu_r = 18.0_f64;
        let mean_mu = (mu_g + mu_r) / 2.0;
        let expected = (((mu_g - mean_mu).powi(2) + (mu_r - mean_mu).powi(2)) / 2.0).sqrt();
        assert!((result[0] - expected).abs() < 1e-10);
    }

    #[test]
    fn color_spread_uniform_bands() {
        // All bands have the same weighted mean -> spread should be 0
        let eval = ColorSpread::new(make_passbands(&["g", "r"]));
        let t = vec![0.0_f64; 4];
        let m = vec![5.0, 5.0, 5.0, 5.0];
        let w = vec![1.0_f64; 4];
        let bands: Vec<StringPassband> = vec!["g", "g", "r", "r"]
            .into_iter()
            .map(StringPassband::from)
            .collect();
        let mut mcts = MultiColorTimeSeries::from_flat(t, m, w, bands);
        let result = eval.eval_multicolor(&mut mcts).unwrap();
        assert!(result[0].abs() < 1e-10);
    }

    #[test]
    fn color_spread_names() {
        let eval = ColorSpread::new(make_passbands(&["g", "r"]));
        assert_eq!(eval.get_names(), vec!["color_spread"]);
        assert_eq!(eval.size_hint(), 1);
    }

    #[test]
    fn color_spread_serde() {
        let eval = ColorSpread::new(make_passbands(&["g", "r"]));
        let json = serde_json::to_string(&eval).unwrap();
        let eval2: ColorSpread<StringPassband> = serde_json::from_str(&json).unwrap();
        assert_eq!(json, serde_json::to_string(&eval2).unwrap());
    }
}
