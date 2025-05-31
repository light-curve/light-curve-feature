use crate::float_trait::Float;
use crate::sorted_array::SortedArray;

use crate::RecurrentSinCos;
use conv::{ConvAsUtil, ConvUtil, RoundToNearest};
use enum_dispatch::enum_dispatch;
use itertools::Itertools;
use macro_const::macro_const;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::fmt::{Debug, Display};
use std::ops::Mul;

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
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
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
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
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
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
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
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
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
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
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
    fn iter_sin_cos(&self) -> RecurrentSinCos<T>;
}

#[enum_dispatch(FreqGridTrait<T>)]
#[derive(Clone, Debug)]
#[non_exhaustive]
pub enum FreqGrid<T: Float> {
    ZeroBasedPow2(ZeroBasedPow2FreqGrid<T>),
    Linear(LinearFreqGrid<T>),
    // Arbitrary(SortedArray<T>),
}

impl<T: Float> FreqGrid<T> {
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
    pub fn expect_zero_based_pow2(&self, message: impl Display) -> &ZeroBasedPow2FreqGrid<T> {
        match self {
            FreqGrid::ZeroBasedPow2(x) => x,
            _ => panic!("FreqGrid is not ZeroBasedPow2: {message}"),
        }
    }
}

impl<T: Float> Mul<T> for &FreqGrid<T> {
    type Output = FreqGrid<T>;

    fn mul(self, rhs: T) -> Self::Output {
        match self {
            FreqGrid::ZeroBasedPow2(grid) => FreqGrid::ZeroBasedPow2(grid * rhs),
            FreqGrid::Linear(grid) => FreqGrid::Linear(grid * rhs),
        }
    }
}

impl<T> FreqGrid<T>
where
    T: Float,
{
    pub fn from_t(t: &[T], resolution: f32, max_freq_factor: f32, nyquist: NyquistFreq) -> Self {
        assert!(resolution.is_sign_positive() && resolution.is_finite());

        let sizef: T = t.len().approx().unwrap();
        let duration = t[t.len() - 1] - t[0];
        let step = T::two() * T::PI() * (sizef - T::one())
            / (sizef * resolution.value_as::<T>().unwrap() * duration);
        let max_freq = nyquist.nyquist_freq(t) * max_freq_factor.value_as::<T>().unwrap();
        let log2_size: u32 = T::log2(max_freq / step)
            .approx_by::<RoundToNearest>()
            .unwrap();
        Self::ZeroBasedPow2(ZeroBasedPow2FreqGrid::new(step, log2_size))
    }
}

#[derive(Clone, Debug)]
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
        assert!(size > 0, "Size must not be zero");
        let size_m1 = size - 1;
        if size_m1.is_power_of_two() {
            Some(Self::new(step, size_m1.ilog2()))
        } else {
            None
        }
    }

    pub fn step(&self) -> T {
        self.step
    }
}

impl<T: Float> Mul<T> for ZeroBasedPow2FreqGrid<T> {
    type Output = ZeroBasedPow2FreqGrid<T>;
    fn mul(mut self, rhs: T) -> Self::Output {
        self.step *= rhs;
        self
    }
}

impl<T: Float> Mul<T> for &ZeroBasedPow2FreqGrid<T> {
    type Output = ZeroBasedPow2FreqGrid<T>;
    fn mul(self, rhs: T) -> Self::Output {
        ZeroBasedPow2FreqGrid {
            step: self.step * rhs,
            size: self.size,
            log2_size_m1: self.log2_size_m1,
        }
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

    fn iter_sin_cos(&self) -> RecurrentSinCos<T> {
        RecurrentSinCos::with_zero_first(self.step)
    }
}

#[derive(Clone, Debug)]
pub struct LinearFreqGrid<T: Float> {
    /// Grid start point
    start: T,
    /// Distance between points
    step: T,
    /// Number of points
    size: usize,
}

impl<T: Float> Mul<T> for &LinearFreqGrid<T> {
    type Output = LinearFreqGrid<T>;

    fn mul(self, rhs: T) -> Self::Output {
        LinearFreqGrid {
            start: self.start * rhs,
            step: self.step * rhs,
            size: self.size,
        }
    }
}

impl<T: Float> LinearFreqGrid<T> {
    pub fn new(start: T, step: T, size: usize) -> Self {
        assert!(step.is_finite(), "frequency step must be finite");
        assert!(size > 0, "Size must not be zero");
        Self { start, step, size }
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

    fn iter_sin_cos(&self) -> RecurrentSinCos<T> {
        RecurrentSinCos::new(self.start, self.step)
    }
}
