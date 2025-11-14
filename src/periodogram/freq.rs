use crate::float_trait::Float;
use crate::sorted_array::SortedArray;

use crate::RecurrentSinCos;
use crate::error::SortedArrayError;
use crate::periodogram::sin_cos_iterator::SinCosIterator;
use conv::{ConvAsUtil, ConvUtil, RoundToNearest};
use enum_dispatch::enum_dispatch;
use itertools::Itertools;
use macro_const::macro_const;
use ndarray::{Array1, ArrayView1};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::fmt::Debug;

macro_const! {
    const NYQUIST_FREQ_DOC: &'static str = r"Derive Nyquist frequency from time series

Nyquist frequency for unevenly time series is not well-defined. Here we define it as
$\pi / \delta t$, where $\delta t$ is some typical interval between consequent observations
";
}

#[doc = NYQUIST_FREQ_DOC!()]
#[enum_dispatch]
trait NyquistFreqTrait: Send + Sync + Clone + Debug {
    fn nyquist_freq<T: Float>(&self, t: &[T]) -> T;
}

#[doc = NYQUIST_FREQ_DOC!()]
#[enum_dispatch(NyquistFreqTrait)]
#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum NyquistFreq {
    Average(AverageNyquistFreq),
    Median(MedianNyquistFreq),
    Quantile(QuantileNyquistFreq),
    Fixed(FixedNyquistFreq),
}

impl NyquistFreq {
    pub fn average() -> Self {
        Self::Average(AverageNyquistFreq)
    }

    pub fn median() -> Self {
        Self::Median(MedianNyquistFreq)
    }

    pub fn quantile(quantile: f32) -> Self {
        Self::Quantile(QuantileNyquistFreq { quantile })
    }

    pub fn fixed(freq: f32) -> Self {
        Self::Fixed(FixedNyquistFreq(freq))
    }
}

/// $\Delta t = \mathrm{duration} / (N - 1)$ is the mean time interval between observations
///
/// The denominator is $(N-1)$ for compatibility with Nyquist frequency for uniform grid. Note that
/// in literature definition of "average Nyquist" frequency usually differ and place $N$ to the
/// denominator
#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Average")]
pub struct AverageNyquistFreq;

impl NyquistFreqTrait for AverageNyquistFreq {
    fn nyquist_freq<T: Float>(&self, t: &[T]) -> T {
        let n = t.len();
        T::PI() * (n - 1).value_as().unwrap() / (t[n - 1] - t[0])
    }
}

fn diff<T: Float>(x: &[T]) -> Vec<T> {
    x.iter().tuple_windows().map(|(&a, &b)| b - a).collect()
}

/// $\Delta t$ is the median time interval between observations
#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Median")]
pub struct MedianNyquistFreq;

impl NyquistFreqTrait for MedianNyquistFreq {
    fn nyquist_freq<T: Float>(&self, t: &[T]) -> T {
        let sorted_dt: SortedArray<_> = diff(t).into();
        let dt = sorted_dt.median();
        T::PI() / dt
    }
}

/// $\Delta t$ is the $q$th quantile of time intervals between subsequent observations
#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Quantile")]
pub struct QuantileNyquistFreq {
    pub quantile: f32,
}

impl NyquistFreqTrait for QuantileNyquistFreq {
    fn nyquist_freq<T: Float>(&self, t: &[T]) -> T {
        let sorted_dt: SortedArray<_> = diff(t).into();
        let dt = sorted_dt.ppf(self.quantile);
        T::PI() / dt
    }
}

/// User-defined Nyquist frequency
///
/// Note, that the actual maximum periodogram frequency provided by `FreqGrid` differs from this
/// value because of `max_freq_factor` and maximum value to step ratio rounding
#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(rename = "Fixed")]
pub struct FixedNyquistFreq(pub f32);

impl FixedNyquistFreq {
    /// pi / dt
    pub fn from_dt<T: Float>(dt: T) -> Self {
        let dt: f32 = dt.approx().unwrap();
        assert!(dt > 0.0);
        Self(core::f32::consts::PI / dt)
    }
}

impl NyquistFreqTrait for FixedNyquistFreq {
    fn nyquist_freq<T: Float>(&self, _t: &[T]) -> T {
        self.0.value_as().unwrap()
    }
}

#[enum_dispatch]
pub trait FreqGridTrait<T>: Send + Sync + Clone + Debug {
    fn size(&self) -> usize;
    fn get(&self, i: usize) -> T;
    fn minimum(&self) -> T;
    fn maximum(&self) -> T;
    /// Iterator of (sin(freq * time), cos(freq * time)) over the freq values
    fn iter_sin_cos_mul(&self, time: T) -> SinCosIterator<'_, T>;
}

#[enum_dispatch(FreqGridTrait<T>)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound = "T: Float")]
#[non_exhaustive]
pub enum FreqGrid<T: Float> {
    Arbitrary(SortedArray<T>),
    ZeroBasedPow2(ZeroBasedPow2FreqGrid<T>),
    Linear(LinearFreqGrid<T>),
}

impl<T: Float> FreqGrid<T> {
    /// Construct from a sorted frequency array
    pub fn try_from_sorted_array(
        sorted_array: impl Into<Array1<T>>,
    ) -> Result<Self, SortedArrayError> {
        Ok(Self::Arbitrary(SortedArray::from_sorted(sorted_array)?))
    }

    /// Construct from an array reference, array will be copied and sorted
    pub fn from_array<'a>(array: impl Into<ArrayView1<'a, T>>) -> Self {
        let array_view = array.into();
        Self::Arbitrary(array_view.into())
    }

    /// Construct a linear grid starting at zero, and having 2^log2_size_m1 + 1 points
    pub fn zero_based_pow2(step: T, log2_size_m1: u32) -> Self {
        Self::ZeroBasedPow2(ZeroBasedPow2FreqGrid::new(step, log2_size_m1))
    }

    /// Construct a linear grid
    pub fn linear(start: T, step: T, size: usize) -> Self {
        if start.is_zero() {
            if let Some(zero_based_pow2) = ZeroBasedPow2FreqGrid::try_with_size(step, size) {
                return Self::ZeroBasedPow2(zero_based_pow2);
            }
        }
        Self::Linear(LinearFreqGrid::new(start, step, size))
    }

    /// Unwrap into ZeroBasedPow2FreqGrid or panic with a given message.
    pub fn to_zero_based_pow2(&self) -> Option<&ZeroBasedPow2FreqGrid<T>> {
        match self {
            FreqGrid::ZeroBasedPow2(x) => Some(x),
            _ => None,
        }
    }
}

impl<T: Float> From<FreqGrid<T>> for Cow<'static, FreqGrid<T>> {
    fn from(value: FreqGrid<T>) -> Self {
        Cow::Owned(value)
    }
}

impl<T: Float> FreqGridTrait<T> for SortedArray<T> {
    fn size(&self) -> usize {
        self.len()
    }

    fn get(&self, i: usize) -> T {
        self.0[i]
    }

    fn minimum(&self) -> T {
        self.0[0]
    }

    fn maximum(&self) -> T {
        self.0[self.0.len() - 1]
    }

    fn iter_sin_cos_mul(&self, time: T) -> SinCosIterator<'_, T> {
        let angle_iter = self.iter().copied().map(move |freq| freq * time);
        SinCosIterator::from_angles(angle_iter)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound = "T: Float")]
pub struct ZeroBasedPow2FreqGrid<T: Float> {
    /// Step between the points
    step: T,
    /// Number of points, guaranteed to be 2^k + 1
    size: usize,
    /// log2(size - 1)
    log2_size_m1: u32,
}

impl<T: Float> ZeroBasedPow2FreqGrid<T> {
    pub fn new(step: T, log2_size_m1: u32) -> Self {
        assert!(
            step.is_finite() && step.is_sign_positive(),
            "step should be positive finite"
        );
        Self {
            step,
            size: (1 << log2_size_m1) + 1,
            log2_size_m1,
        }
    }

    pub fn try_with_size(step: T, size: usize) -> Option<Self> {
        assert!(
            step.is_sign_positive() && step.is_finite(),
            "step should be finite and positive"
        );
        assert!(size > 0, "Size must not be zero");
        let size_m1 = size - 1;
        if size_m1.is_power_of_two() {
            Some(Self::new(step, size_m1.ilog2()))
        } else {
            None
        }
    }

    pub fn from_t(t: &[T], params: &DynamicFreqGridParams) -> Self {
        let (_duration, step, max_freq) = params.duration_step_max_freq(t);

        let log2_size: u32 = T::log2(max_freq / step)
            .approx_by::<RoundToNearest>()
            .unwrap();
        Self::new(step, log2_size)
    }

    pub fn step(&self) -> T {
        self.step
    }
}

impl<T: Float> FreqGridTrait<T> for ZeroBasedPow2FreqGrid<T> {
    fn size(&self) -> usize {
        self.size
    }
    fn get(&self, i: usize) -> T {
        self.step * i.approx().unwrap()
    }

    fn minimum(&self) -> T {
        T::zero()
    }

    fn maximum(&self) -> T {
        self.step * (self.size - 1).approx().unwrap()
    }

    fn iter_sin_cos_mul(&self, time: T) -> SinCosIterator<'static, T> {
        RecurrentSinCos::with_zero_first(self.step * time).into()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound = "T: Float")]
pub struct LinearFreqGrid<T: Float> {
    /// Grid start point
    start: T,
    /// Distance between points
    step: T,
    /// Number of points
    size: usize,
}

impl<T: Float> LinearFreqGrid<T> {
    pub fn new(start: T, step: T, size: usize) -> Self {
        assert!(start >= T::zero(), "start must not be negative");
        assert!(
            step.is_finite() && step.is_sign_positive(),
            "frequency step must be finite and positive"
        );
        assert!(size > 0, "Size must not be zero");
        Self { start, step, size }
    }

    pub fn from_t(t: &[T], params: &DynamicFreqGridParams) -> Self {
        let (duration, step, max_freq) = params.duration_step_max_freq(t);
        // Corresponds to the half-duration
        let min_freq = T::four() * T::PI() / duration;
        // At least 1
        let size: usize = {
            let sizef = (max_freq - min_freq) / step;
            if sizef >= T::one() {
                sizef.approx_by::<RoundToNearest>().unwrap()
            } else {
                1
            }
        };
        Self {
            start: min_freq,
            step,
            size,
        }
    }
}

impl<T: Float> FreqGridTrait<T> for LinearFreqGrid<T> {
    fn size(&self) -> usize {
        self.size
    }

    fn get(&self, i: usize) -> T {
        self.start + self.step * i.approx().unwrap()
    }

    fn minimum(&self) -> T {
        self.start
    }

    fn maximum(&self) -> T {
        self.start + self.step * (self.size - 1).approx().unwrap()
    }

    fn iter_sin_cos_mul(&self, time: T) -> SinCosIterator<'_, T> {
        RecurrentSinCos::new(self.start * time, self.step * time).into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_based_pow2_vs_linear() {
        // Test with different power of 2 sizes
        for log2_size in 1..=5 {
            let size = (1 << log2_size) + 1;
            let step = 0.1;

            // Create both grids
            let pow2_grid = ZeroBasedPow2FreqGrid::new(step, log2_size);
            let linear_grid = LinearFreqGrid::new(0.0, step, size);

            // Check sizes match
            assert_eq!(pow2_grid.size(), linear_grid.size());

            // Check all values match
            for i in 0..size {
                assert_eq!(pow2_grid.get(i), linear_grid.get(i));
            }

            // Check minimum and maximum values match
            assert_eq!(pow2_grid.minimum(), linear_grid.minimum());
            assert_eq!(pow2_grid.maximum(), linear_grid.maximum());

            // Check step values match
            assert_eq!(pow2_grid.step(), step);
            assert_eq!(linear_grid.step, step);
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize, JsonSchema)]
pub struct DynamicFreqGridParams {
    pub resolution: f32,
    pub max_freq_factor: f32,
    pub nyquist: NyquistFreq,
}

impl DynamicFreqGridParams {
    pub fn new(resolution: f32, max_freq_factor: f32, nyquist: impl Into<NyquistFreq>) -> Self {
        assert!(resolution > 0.0, "Resolution must be positive");
        assert!(max_freq_factor > 0.0, "Max frequency must be positive");
        Self {
            resolution,
            max_freq_factor,
            nyquist: nyquist.into(),
        }
    }

    /// Helper function for from_t implementations
    #[inline(always)]
    fn duration_step_max_freq<T: Float>(&self, t: &[T]) -> (T, T, T) {
        let sizef: T = t.len().approx().unwrap();
        let duration = t[t.len() - 1] - t[0];
        let step = T::two() * T::PI() * (sizef - T::one())
            / (sizef * self.resolution.value_as::<T>().unwrap() * duration);
        let max_freq = self.nyquist.nyquist_freq(t) * self.max_freq_factor.value_as::<T>().unwrap();

        (duration, step, max_freq)
    }
}

/// Defines a strategy of FreqGrid selection.
///
/// It is either a fixed grid, or a grid defined dynamically for each input time series.
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(bound = "T: Float")]
pub enum FreqGridStrategy<T: Float> {
    Fixed(FreqGrid<T>),
    Dynamic(DynamicFreqGridParams),
}

impl<T: Float> FreqGridStrategy<T> {
    pub fn fixed(freq_grid: FreqGrid<T>) -> Self {
        Self::Fixed(freq_grid)
    }

    pub fn dynamic(resolution: f32, max_freq_factor: f32, nyquist: impl Into<NyquistFreq>) -> Self {
        Self::Dynamic(DynamicFreqGridParams::new(
            resolution,
            max_freq_factor,
            nyquist,
        ))
    }

    pub fn freq_grid(&self, t: &[T], zero_base: bool) -> Cow<'_, FreqGrid<T>> {
        match self {
            Self::Fixed(freq_grid) => Cow::Borrowed(freq_grid),
            Self::Dynamic(params) => {
                let freq_grid = if zero_base {
                    let zero_based_grid = ZeroBasedPow2FreqGrid::from_t(t, params);
                    FreqGrid::ZeroBasedPow2(zero_based_grid)
                } else {
                    let linear_grid = LinearFreqGrid::from_t(t, params);
                    FreqGrid::Linear(linear_grid)
                };
                Cow::Owned(freq_grid)
            }
        }
    }
}

impl<T: Float> From<FreqGrid<T>> for FreqGridStrategy<T> {
    fn from(freq_grid: FreqGrid<T>) -> Self {
        Self::Fixed(freq_grid)
    }
}

impl<T: Float> From<DynamicFreqGridParams> for FreqGridStrategy<T> {
    fn from(params: DynamicFreqGridParams) -> Self {
        Self::Dynamic(params)
    }
}

impl<T: Float> From<ZeroBasedPow2FreqGrid<T>> for FreqGridStrategy<T> {
    fn from(zero_based_pow2freq_grid: ZeroBasedPow2FreqGrid<T>) -> Self {
        let freq_grid: FreqGrid<T> = zero_based_pow2freq_grid.into();
        freq_grid.into()
    }
}

impl<T: Float> From<LinearFreqGrid<T>> for FreqGridStrategy<T> {
    fn from(linear_grid: LinearFreqGrid<T>) -> Self {
        let freq_grid: FreqGrid<T> = linear_grid.into();
        freq_grid.into()
    }
}
