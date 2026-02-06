//! Periodogram-related stuff

use crate::float_trait::Float;
use crate::time_series::TimeSeries;

use enum_dispatch::enum_dispatch;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

pub mod fft_trait;
pub use fft_trait::{Fft, FftComplex, FftFloat, FftInputArray, FftOutputArray};

#[cfg(feature = "fftw")]
mod fft_fftw;
#[cfg(feature = "fftw")]
pub use fft_fftw::{FftwFft, FftwFloat};

mod fft_rustfft;
pub use fft_rustfft::{RustFft, RustFftFloat};

mod freq;
pub use freq::{
    AverageNyquistFreq, FixedNyquistFreq, FreqGrid, FreqGridStrategy, FreqGridTrait,
    MedianNyquistFreq, NyquistFreq, QuantileNyquistFreq,
};

mod power_fft;
pub use power_fft::PeriodogramPowerFft;

mod power_direct;
pub use power_direct::PeriodogramPowerDirect;

mod power_trait;
pub use power_trait::{PeriodogramNormalization, PeriodogramPowerError, PeriodogramPowerTrait};

pub mod sin_cos_iterator;

/// Default FFT-based periodogram power using RustFFT backend
pub type DefaultPeriodogramPowerFft<T> = PeriodogramPowerFft<T, RustFft<T>>;

/// Periodogram execution algorithm
#[enum_dispatch(PeriodogramPowerTrait<T>)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[serde(bound = "T: Float")]
#[non_exhaustive]
pub enum PeriodogramPower<T>
where
    T: Float,
{
    /// FFT-based periodogram using the RustFFT backend (default)
    Fft(PeriodogramPowerFft<T, RustFft<T>>),
    /// FFT-based periodogram using the FFTW backend (only available when FFTW is enabled)
    #[cfg(feature = "fftw")]
    FftFftw(PeriodogramPowerFft<T, FftwFft<T>>),
    /// Direct periodogram computation (slower but more precise)
    Direct(PeriodogramPowerDirect),
}

/// Lamb-Scargle periodogram calculator on uniform frequency grid
///
/// Frequencies are given by $\omega = \{\min\omega..\max\omega\}$: $N_\omega$ nodes with step
/// $\Delta\omega$: $\min\omega = \Delta\omega$, $\max\omega = N_\omega \Delta\omega$.
///
/// Parameters of the grid can be derived from time series properties: typical time interval
/// $\delta t$ and duration of observation. The idea is to set maximum frequency to Nyquist
/// value $\pi / \Delta t$ and minimum frequency to $2\pi / \mathrm{duration}$, while `nyquist` and
/// `resolution` factors are used to widen this interval:
/// $$
/// \max\omega = N_\omega \Delta\omega = \frac{\pi}{\Delta t},
/// $$
/// $$
/// \min\omega = \Delta\omega = \frac{2\pi}{\mathrm{resolution} \times \mathrm{duration}}.
/// $$
pub struct Periodogram<'a, T>
where
    T: Float,
{
    freq_grid: Cow<'a, FreqGrid<T>>,
    periodogram_power: PeriodogramPower<T>,
    normalization: PeriodogramNormalization,
}

impl<'a, T> Periodogram<'a, T>
where
    T: Float,
{
    pub fn new(
        periodogram_power: PeriodogramPower<T>,
        freq_grid: Cow<'a, FreqGrid<T>>,
    ) -> Result<Self, PeriodogramPowerError> {
        Self::with_normalization(
            periodogram_power,
            freq_grid,
            PeriodogramNormalization::default(),
        )
    }

    pub fn with_normalization(
        periodogram_power: PeriodogramPower<T>,
        freq_grid: Cow<'a, FreqGrid<T>>,
        normalization: PeriodogramNormalization,
    ) -> Result<Self, PeriodogramPowerError> {
        let is_fft = matches!(periodogram_power, PeriodogramPower::Fft(_));
        #[cfg(feature = "fftw")]
        let is_fft = is_fft || matches!(periodogram_power, PeriodogramPower::FftFftw(_));
        if is_fft && !matches!(freq_grid.as_ref(), FreqGrid::ZeroBasedPow2(_)) {
            return Err(PeriodogramPowerError::PeriodogramFftWrongFreqGrid);
        }
        Ok(Self {
            freq_grid,
            periodogram_power,
            normalization,
        })
    }

    pub fn from_t(
        periodogram_power: PeriodogramPower<T>,
        t: &[T],
        freq_grid_strategy: &'a FreqGridStrategy<T>,
    ) -> Result<Self, PeriodogramPowerError> {
        let zero_base = !matches!(periodogram_power, PeriodogramPower::Direct(_));
        Self::new(
            periodogram_power,
            freq_grid_strategy.freq_grid(t, zero_base),
        )
    }

    /// Set the power normalization strategy
    pub fn set_normalization(&mut self, normalization: PeriodogramNormalization) -> &mut Self {
        self.normalization = normalization;
        self
    }

    /// Get the current power normalization strategy
    pub fn normalization(&self) -> PeriodogramNormalization {
        self.normalization
    }

    pub fn freq(&self, i: usize) -> T {
        self.freq_grid.get(i)
    }

    /// Compute the periodogram power with the configured normalization
    ///
    /// Returns power values normalized according to the `normalization` setting.
    /// See [PeriodogramNormalization] for details on available normalizations.
    pub fn power(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        let raw_power = self
            .periodogram_power
            .power(&self.freq_grid, ts)
            .expect("Unexpected error from PeriodogramPowerTrait::power");
        self.normalization.normalize(raw_power, ts.lenu())
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    use crate::peak_indices::peak_indices_reverse_sorted;
    use crate::periodogram::freq::{DynamicFreqGridParams, ZeroBasedPow2FreqGrid};
    use crate::sorted_array::SortedArray;

    use approx::assert_relative_eq;
    use light_curve_common::{all_close, linspace};
    use ndarray::Array1;
    use rand::prelude::*;

    #[test]
    fn compr_direct_with_scipy() {
        const OMEGA_SIN: f64 = 0.07;
        const N: usize = 100;
        let t = linspace(0.0, 99.0, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA_SIN * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let periodogram = Periodogram::new(
            PeriodogramPowerDirect.into(),
            FreqGrid::zero_based_pow2(OMEGA_SIN, 0).into(),
        )
        .unwrap();
        assert_relative_eq!(
            periodogram.power(&mut ts)[1] * 2.0 / (N as f64 - 1.0),
            1.0,
            max_relative = 1.0 / (N as f64),
        );

        // import numpy as np
        // from scipy.signal import lombscargle
        //
        // t = np.arange(100)
        // m = np.sin(0.07 * t)
        // y = (m - m.mean()) / m.std(ddof=1)
        // freq = np.linspace(0.0, 0.04, 5)
        // print(lombscargle(t, y, freq, precenter=True, normalize=False))

        let freq_grid = ZeroBasedPow2FreqGrid::new(0.01, 2);
        let periodogram = Periodogram::new(
            PeriodogramPowerDirect.into(),
            FreqGrid::ZeroBasedPow2(freq_grid.clone()).into(),
        )
        .unwrap();
        assert_relative_eq!(
            &Array1::linspace(
                0.0,
                freq_grid.step() * (freq_grid.size() as f64 - 1.0),
                freq_grid.size(),
            ),
            &(0..freq_grid.size())
                .map(|i| periodogram.freq(i))
                .collect::<Array1<_>>(),
            max_relative = 1e-12,
        );
        let desired = [
            3.76158192e-33,
            1.69901802e+01,
            1.85772252e+01,
            2.19604974e+01,
            2.81505681e+01,
        ];
        let actual = periodogram.power(&mut ts);
        assert_relative_eq!(
            &actual[..],
            &desired[..],
            max_relative = 1e-6,
            epsilon = f64::EPSILON
        );
    }

    #[test]
    fn direct_vs_fft_one_to_one() {
        const OMEGA: f64 = 0.472;
        const N: usize = 64;
        const RESOLUTION: f32 = 1.0;
        const MAX_FREQ_FACTOR: f32 = 1.0;

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let params = DynamicFreqGridParams::new(RESOLUTION, MAX_FREQ_FACTOR, AverageNyquistFreq);
        let freq_grid_strategy = ZeroBasedPow2FreqGrid::from_t(&t, &params).into();

        let direct = Periodogram::from_t(PeriodogramPowerDirect.into(), &t, &freq_grid_strategy)
            .unwrap()
            .power(&mut ts);
        let fft = Periodogram::from_t(
            DefaultPeriodogramPowerFft::new().into(),
            &t,
            &freq_grid_strategy,
        )
        .unwrap()
        .power(&mut ts);
        all_close(&fft[..direct.len() - 1], &direct[..direct.len() - 1], 1e-8);
    }

    #[test]
    fn direct_vs_fft_uniform_sin_cos() {
        const OMEGA1: f64 = 0.472;
        const OMEGA2: f64 = 1.222;
        const AMPLITUDE2: f64 = 2.0;
        const N: usize = 100;
        const RESOLUTION: f32 = 4.0;
        const MAX_FREQ_FACTOR: f32 = 1.0;

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<_> = t
            .iter()
            .map(|&x| f64::sin(OMEGA1 * x) + AMPLITUDE2 * f64::cos(OMEGA2 * x))
            .collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let params = DynamicFreqGridParams::new(RESOLUTION, MAX_FREQ_FACTOR, AverageNyquistFreq);
        let freq_grid_strategy = ZeroBasedPow2FreqGrid::from_t(&t, &params).into();

        let direct = Periodogram::from_t(PeriodogramPowerDirect.into(), &t, &freq_grid_strategy)
            .unwrap()
            .power(&mut ts);
        let fft = Periodogram::from_t(
            DefaultPeriodogramPowerFft::new().into(),
            &t,
            &freq_grid_strategy,
        )
        .unwrap()
        .power(&mut ts);

        let fft_arr = ndarray::Array1::from_vec(fft);
        let direct_arr = ndarray::Array1::from_vec(direct);
        assert_eq!(
            peak_indices_reverse_sorted(&fft_arr)[..2],
            peak_indices_reverse_sorted(&direct_arr)[..2]
        );
    }

    #[test]
    fn direct_vs_fft_unevenly_sin_cos() {
        const OMEGA1: f64 = 0.222;
        const OMEGA2: f64 = 1.222;
        const AMPLITUDE2: f64 = 2.0;
        const NOISE_AMPLITUDE: f64 = 1.0;
        const N: usize = 100;

        let mut rng = StdRng::seed_from_u64(0);
        let t: SortedArray<_> = (0..N)
            .map(|_| rng.random::<f64>() * (N - 1) as f64)
            .collect::<Vec<_>>()
            .into();
        let m: Vec<_> = t
            .iter()
            .map(|&x| {
                f64::sin(OMEGA1 * x)
                    + AMPLITUDE2 * f64::cos(OMEGA2 * x)
                    + NOISE_AMPLITUDE * rng.random::<f64>()
            })
            .collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let freq_grid_strategy = ZeroBasedPow2FreqGrid::try_with_size(0.01, 257)
            .unwrap()
            .into();

        let direct = Periodogram::from_t(PeriodogramPowerDirect.into(), &t, &freq_grid_strategy)
            .unwrap()
            .power(&mut ts);
        let fft = Periodogram::from_t(
            DefaultPeriodogramPowerFft::new().into(),
            &t,
            &freq_grid_strategy,
        )
        .unwrap()
        .power(&mut ts);

        let fft_arr = ndarray::Array1::from_vec(fft);
        let direct_arr = ndarray::Array1::from_vec(direct);
        assert_eq!(
            peak_indices_reverse_sorted(&fft_arr)[..2],
            peak_indices_reverse_sorted(&direct_arr)[..2]
        );
    }

    #[test]
    fn arbitrary_vs_zero_based_pow2_for_direct() {
        let step = 0.1;
        let log2_size_m1 = 7;
        let size = (1 << log2_size_m1) + 1;

        let zero_based_grid = FreqGrid::zero_based_pow2(step, log2_size_m1);

        let freqs: ndarray::Array1<_> = (0..size).map(|i| step * i as f64).collect();
        let arbitrary_grid = FreqGrid::from_array(&freqs);

        let n_points = 100;
        let t: Vec<f64> = linspace(0.0, 10.0, n_points);
        let m: Vec<_> = t
            .iter()
            .map(|&x| (x * 2.3).sin() + (x * 0.7).cos())
            .collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);

        let direct = PeriodogramPowerDirect;
        let power_zero_based = direct.power(&zero_based_grid, &mut ts).unwrap();
        let power_arbitrary = direct.power(&arbitrary_grid, &mut ts).unwrap();

        assert_relative_eq!(
            &power_zero_based[..],
            &power_arbitrary[..],
            max_relative = 1e-10
        );
    }

    // Same as the previous test, but with non-zero step and LinearGrid
    #[test]
    fn arbitrary_vs_linear_for_direct() {
        let start = 0.33;
        let step = 0.1;
        let size = 1000;

        let linear_grid = FreqGrid::linear(start, step, size);

        let freqs: Vec<_> = (0..size).map(|i| start + step * i as f64).collect();
        let freqs_from_linear_grid = (0..size).map(|i| linear_grid.get(i)).collect::<Vec<_>>();
        assert_relative_eq!(
            &freqs_from_linear_grid[..],
            &freqs[..],
            max_relative = 1e-10
        );

        let arbitrary_grid = FreqGrid::try_from_sorted_array(freqs).unwrap();

        let n_points = 100;
        let t: Vec<f64> = linspace(0.0, 10.0, n_points);
        let m: Vec<_> = t
            .iter()
            .map(|&x| (x * 1.77).sin() + (x * 1756.55).cos())
            .collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);

        let direct = PeriodogramPowerDirect;
        let power_linear = direct.power(&linear_grid, &mut ts).unwrap();
        let power_arbitrary = direct.power(&arbitrary_grid, &mut ts).unwrap();

        assert_relative_eq!(
            &power_linear[..],
            &power_arbitrary[..],
            max_relative = 1e-10
        );
    }

    #[test]
    fn standard_normalization_bounds() {
        // For a pure sinusoid, standard normalization should give peak power close to 1.0
        const OMEGA_SIN: f64 = 0.07;
        const N: usize = 100;
        let t = linspace(0.0, 99.0, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA_SIN * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);

        let mut periodogram = Periodogram::new(
            PeriodogramPowerDirect.into(),
            FreqGrid::zero_based_pow2(OMEGA_SIN, 0).into(),
        )
        .unwrap();
        periodogram.set_normalization(PeriodogramNormalization::Standard);

        let power = periodogram.power(&mut ts);
        // Peak power at frequency index 1 should be close to 1.0
        assert_relative_eq!(power[1], 1.0, max_relative = 1.0 / (N as f64));
        // All values should be bounded [0, 1]
        for &p in &power {
            assert!(
                (0.0..=1.0 + 1e-10).contains(&p),
                "Power {} out of bounds",
                p
            );
        }
    }

    #[test]
    fn psd_normalization_matches_original() {
        // Psd normalization should match the original behavior (manual factor application)
        const OMEGA_SIN: f64 = 0.07;
        const N: usize = 100;
        let t = linspace(0.0, 99.0, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA_SIN * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);

        let periodogram = Periodogram::new(
            PeriodogramPowerDirect.into(),
            FreqGrid::zero_based_pow2(OMEGA_SIN, 0).into(),
        )
        .unwrap();

        // Default is Psd
        assert_eq!(periodogram.normalization(), PeriodogramNormalization::Psd);

        let power = periodogram.power(&mut ts);
        // Manually applying the factor should give ~1.0
        assert_relative_eq!(
            power[1] * 2.0 / (N as f64 - 1.0),
            1.0,
            max_relative = 1.0 / (N as f64),
        );
    }

    #[test]
    fn model_normalization() {
        const OMEGA_SIN: f64 = 0.07;
        const N: usize = 100;
        let t = linspace(0.0, 99.0, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA_SIN * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);

        let mut periodogram = Periodogram::new(
            PeriodogramPowerDirect.into(),
            FreqGrid::zero_based_pow2(OMEGA_SIN, 0).into(),
        )
        .unwrap();
        periodogram.set_normalization(PeriodogramNormalization::Model);

        let power = periodogram.power(&mut ts);
        // Model normalization: P_model = P_std / (1 - P_std)
        // For P_std close to 1, P_model should be very large
        assert!(power[1] > 10.0, "Model power at peak should be large");
        // All values should be non-negative
        for &p in &power {
            assert!(p >= 0.0, "Power {} should be non-negative", p);
        }
    }

    #[test]
    fn log_normalization() {
        const OMEGA_SIN: f64 = 0.07;
        const N: usize = 100;
        let t = linspace(0.0, 99.0, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA_SIN * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);

        let mut periodogram = Periodogram::new(
            PeriodogramPowerDirect.into(),
            FreqGrid::zero_based_pow2(OMEGA_SIN, 0).into(),
        )
        .unwrap();
        periodogram.set_normalization(PeriodogramNormalization::Log);

        let power = periodogram.power(&mut ts);
        // Log normalization: P_log = -ln(1 - P_std)
        // For P_std close to 1, P_log should be large
        assert!(power[1] > 3.0, "Log power at peak should be large");
        // All values should be non-negative
        for &p in &power {
            assert!(p >= 0.0, "Power {} should be non-negative", p);
        }
    }

    #[test]
    fn normalization_consistency_across_methods() {
        // Standard normalization should give same results for Direct and FFT methods
        const OMEGA: f64 = 0.472;
        const N: usize = 64;
        const RESOLUTION: f32 = 1.0;
        const MAX_FREQ_FACTOR: f32 = 1.0;

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let params = DynamicFreqGridParams::new(RESOLUTION, MAX_FREQ_FACTOR, AverageNyquistFreq);
        let freq_grid_strategy = ZeroBasedPow2FreqGrid::from_t(&t, &params).into();

        let mut direct =
            Periodogram::from_t(PeriodogramPowerDirect.into(), &t, &freq_grid_strategy).unwrap();
        direct.set_normalization(PeriodogramNormalization::Standard);

        let mut fft = Periodogram::from_t(
            DefaultPeriodogramPowerFft::new().into(),
            &t,
            &freq_grid_strategy,
        )
        .unwrap();
        fft.set_normalization(PeriodogramNormalization::Standard);

        let direct_power = direct.power(&mut ts);
        let fft_power = fft.power(&mut ts);

        // Exclude last element as FFT and Direct can differ slightly there
        all_close(
            &fft_power[..direct_power.len() - 1],
            &direct_power[..direct_power.len() - 1],
            1e-8,
        );
    }

    #[cfg(feature = "fftw")]
    #[test]
    fn fft_fftw_variant_works() {
        // Test the FftFftw variant of PeriodogramPower
        const OMEGA: f64 = 0.472;
        const N: usize = 128;

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);

        // Use a fixed frequency grid to ensure enough points
        let freq_grid_strategy = ZeroBasedPow2FreqGrid::try_with_size(0.01, 65)
            .unwrap()
            .into();

        // Use the explicit FftFftw variant
        let fftw_power: PeriodogramPower<f64> =
            PeriodogramPower::FftFftw(PeriodogramPowerFft::new());
        let fftw_periodogram = Periodogram::from_t(fftw_power, &t, &freq_grid_strategy).unwrap();
        let fftw_result = fftw_periodogram.power(&mut ts);

        // Compare with direct method
        let direct_periodogram =
            Periodogram::from_t(PeriodogramPowerDirect.into(), &t, &freq_grid_strategy).unwrap();
        let direct_result = direct_periodogram.power(&mut ts);

        // Results should be close
        let fft_arr = ndarray::Array1::from_vec(fftw_result);
        let direct_arr = ndarray::Array1::from_vec(direct_result);
        assert_eq!(
            peak_indices_reverse_sorted(&fft_arr)[..2],
            peak_indices_reverse_sorted(&direct_arr)[..2]
        );
    }

    #[cfg(feature = "fftw")]
    #[test]
    fn fft_fftw_vs_default_fft() {
        // Test that FftFftw variant gives similar results as default Fft variant (RustFFT)
        const OMEGA: f64 = 0.472;
        const N: usize = 128;

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);

        // Use a fixed frequency grid to ensure enough points
        let freq_grid_strategy = ZeroBasedPow2FreqGrid::try_with_size(0.01, 65)
            .unwrap()
            .into();

        // Use explicit FftFftw variant
        let fftw_power: PeriodogramPower<f64> =
            PeriodogramPower::FftFftw(PeriodogramPowerFft::new());
        let fftw_periodogram = Periodogram::from_t(fftw_power, &t, &freq_grid_strategy).unwrap();
        let fftw_result = fftw_periodogram.power(&mut ts);

        // Use default FFT (RustFFT)
        let default_fft_periodogram = Periodogram::from_t(
            DefaultPeriodogramPowerFft::new().into(),
            &t,
            &freq_grid_strategy,
        )
        .unwrap();
        let default_result = default_fft_periodogram.power(&mut ts);

        // Peak positions should match
        let fftw_arr = ndarray::Array1::from_vec(fftw_result);
        let default_arr = ndarray::Array1::from_vec(default_result);
        assert_eq!(
            peak_indices_reverse_sorted(&fftw_arr)[..2],
            peak_indices_reverse_sorted(&default_arr)[..2]
        );
    }

    #[cfg(feature = "fftw")]
    #[test]
    fn fft_fftw_different_sizes() {
        // Test FFTW with different array sizes
        let sizes = [16, 32, 64, 128, 256, 512];
        const OMEGA: f64 = 0.3;

        for &n in &sizes {
            let t = linspace(0.0, (n - 1) as f64, n);
            let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA * x)).collect();
            let mut ts = TimeSeries::new_without_weight(&t, &m);

            let freq_grid_strategy = ZeroBasedPow2FreqGrid::try_with_size(0.01, n / 2 + 1)
                .unwrap()
                .into();

            let fftw_power: PeriodogramPower<f64> =
                PeriodogramPower::FftFftw(PeriodogramPowerFft::new());
            let periodogram = Periodogram::from_t(fftw_power, &t, &freq_grid_strategy).unwrap();
            let power = periodogram.power(&mut ts);

            // Verify we get a valid result
            assert_eq!(power.len(), n / 2 + 1);
            assert!(power.iter().all(|&p| p.is_finite()));
            assert!(power.iter().all(|&p| p >= 0.0));
        }
    }

    #[cfg(feature = "fftw")]
    #[test]
    fn fft_fftw_f32() {
        // Test FFTW with f32
        const N: usize = 64;
        const OMEGA: f32 = 0.3;
        const RESOLUTION: f32 = 1.0;
        const MAX_FREQ_FACTOR: f32 = 1.0;

        let t = linspace(0.0_f32, (N - 1) as f32, N);
        let m: Vec<_> = t.iter().map(|&x| f32::sin(OMEGA * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let params = DynamicFreqGridParams::new(RESOLUTION, MAX_FREQ_FACTOR, AverageNyquistFreq);
        let freq_grid_strategy = ZeroBasedPow2FreqGrid::from_t(&t, &params).into();

        let fftw_power: PeriodogramPower<f32> =
            PeriodogramPower::FftFftw(PeriodogramPowerFft::new());
        let periodogram = Periodogram::from_t(fftw_power, &t, &freq_grid_strategy).unwrap();
        let power = periodogram.power(&mut ts);

        // Verify we get valid results
        assert!(power.iter().all(|&p| p.is_finite()));
        assert!(power.iter().all(|&p| p >= 0.0));
    }

    #[cfg(feature = "fftw")]
    #[test]
    fn fftw_vs_rustfft_consistency() {
        // Test that FFTW and RustFFT backends give consistent results
        const OMEGA: f64 = 0.472;
        const N: usize = 64;
        const RESOLUTION: f32 = 1.0;
        const MAX_FREQ_FACTOR: f32 = 1.0;

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA * x)).collect();
        let mut ts = TimeSeries::new_without_weight(&t, &m);
        let params = DynamicFreqGridParams::new(RESOLUTION, MAX_FREQ_FACTOR, AverageNyquistFreq);
        let freq_grid_strategy = ZeroBasedPow2FreqGrid::from_t(&t, &params).into();

        // RustFFT (default)
        let rustfft_periodogram = Periodogram::from_t(
            DefaultPeriodogramPowerFft::new().into(),
            &t,
            &freq_grid_strategy,
        )
        .unwrap();
        let rustfft_result = rustfft_periodogram.power(&mut ts);

        // FFTW (explicit)
        let fftw_power: PeriodogramPower<f64> =
            PeriodogramPower::FftFftw(PeriodogramPowerFft::new());
        let fftw_periodogram = Periodogram::from_t(fftw_power, &t, &freq_grid_strategy).unwrap();
        let fftw_result = fftw_periodogram.power(&mut ts);

        // Results should be very close
        all_close(
            &rustfft_result[..rustfft_result.len() - 1],
            &fftw_result[..fftw_result.len() - 1],
            1e-8,
        );
    }

    #[cfg(feature = "fftw")]
    #[test]
    fn fft_fftw_with_all_normalizations() {
        // Test FftFftw variant with all normalization types
        const OMEGA: f64 = 0.3;
        const N: usize = 64;

        let t = linspace(0.0, (N - 1) as f64, N);
        let m: Vec<_> = t.iter().map(|&x| f64::sin(OMEGA * x)).collect();

        let freq_grid_strategy = ZeroBasedPow2FreqGrid::try_with_size(0.01, N / 2 + 1)
            .unwrap()
            .into();

        for normalization in [
            PeriodogramNormalization::Psd,
            PeriodogramNormalization::Standard,
            PeriodogramNormalization::Model,
            PeriodogramNormalization::Log,
        ] {
            let mut ts = TimeSeries::new_without_weight(&t, &m);

            let fftw_power: PeriodogramPower<f64> =
                PeriodogramPower::FftFftw(PeriodogramPowerFft::new());
            let mut periodogram = Periodogram::from_t(fftw_power, &t, &freq_grid_strategy).unwrap();
            periodogram.set_normalization(normalization);

            let power = periodogram.power(&mut ts);

            // Verify we get valid results
            assert!(
                power.iter().all(|&p| p.is_finite()),
                "Non-finite values with {:?}",
                normalization
            );
            assert!(
                power.iter().all(|&p| p >= 0.0),
                "Negative values with {:?}",
                normalization
            );
        }
    }
}
