use crate::float_trait::Float;
use crate::periodogram::freq::FreqGrid;
use crate::time_series::TimeSeries;

use enum_dispatch::enum_dispatch;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PeriodogramPowerError {
    #[error("PeriodogramFft supports FreqGrid::ZeroBasedPow2 only")]
    PeriodogramFftWrongFreqGrid,
}

/// Periodogram power normalization strategy
///
/// Different normalization strategies for the Lomb-Scargle periodogram power values.
/// See [astropy documentation](https://docs.astropy.org/en/stable/timeseries/lombscargle.html#periodogram-normalization)
/// for more details on the different normalizations and their statistical properties.
///
/// The raw periodogram power computed by this library corresponds to the "psd" normalization
/// in astropy (unnormalized). Other normalizations are derived from this base power.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum PeriodogramNormalization {
    /// Standard normalization
    ///
    /// Power is normalized to lie in the range [0, 1], representing the fraction of variance
    /// explained by the sinusoidal model at each frequency. This is computed as:
    /// `P_standard = P_raw * 2 / (n - 1)`
    ///
    /// This is the default normalization in astropy's `LombScargle`.
    Standard,
    /// Model normalization
    ///
    /// Normalized by the residuals around the periodic model rather than the constant model.
    /// Computed as: `P_model = P_standard / (1 - P_standard)`
    ///
    /// Values range from 0 to infinity, with higher values indicating better fit.
    Model,
    /// Logarithmic normalization
    ///
    /// A logarithmic transformation of the standard normalization.
    /// Computed as: `P_log = -ln(1 - P_standard)`
    ///
    /// Values range from 0 to infinity.
    Log,
    /// Power spectral density (unnormalized)
    ///
    /// The raw periodogram power without additional normalization.
    /// This is equivalent to scipy's `lombscargle` with `normalize=False` after dividing
    /// by the variance of the data.
    ///
    /// Values can exceed 1 for strong periodic signals.
    ///
    /// This is the default to maintain backward compatibility.
    #[default]
    Psd,
}

impl PeriodogramNormalization {
    /// Apply normalization to raw periodogram power values
    ///
    /// # Arguments
    /// * `power` - Raw periodogram power values (psd normalization)
    /// * `n` - Number of data points in the time series
    ///
    /// # Returns
    /// Normalized power values according to the selected normalization strategy
    pub fn normalize<T: Float>(&self, power: Vec<T>, n: usize) -> Vec<T> {
        match self {
            PeriodogramNormalization::Psd => power,
            PeriodogramNormalization::Standard => {
                let factor = T::two() / (T::approx_from(n).unwrap() - T::one());
                power.into_iter().map(|p| p * factor).collect()
            }
            PeriodogramNormalization::Model => {
                let factor = T::two() / (T::approx_from(n).unwrap() - T::one());
                power
                    .into_iter()
                    .map(|p| {
                        let p_std = p * factor;
                        // Avoid division by zero when p_std approaches 1
                        if p_std >= T::one() {
                            T::infinity()
                        } else {
                            p_std / (T::one() - p_std)
                        }
                    })
                    .collect()
            }
            PeriodogramNormalization::Log => {
                let factor = T::two() / (T::approx_from(n).unwrap() - T::one());
                power
                    .into_iter()
                    .map(|p| {
                        let p_std = p * factor;
                        // Avoid log(0) when p_std approaches 1
                        if p_std >= T::one() {
                            T::infinity()
                        } else {
                            -T::ln(T::one() - p_std)
                        }
                    })
                    .collect()
            }
        }
    }
}

/// Periodogram execution algorithm
#[enum_dispatch]
pub trait PeriodogramPowerTrait<T>: Debug + Clone + Send
where
    T: Float,
{
    fn power(
        &self,
        freq: &FreqGrid<T>,
        ts: &mut TimeSeries<T>,
    ) -> Result<Vec<T>, PeriodogramPowerError>;
}
