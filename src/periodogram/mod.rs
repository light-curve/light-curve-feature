//! Periodogram-related stuff

use crate::float_trait::Float;
use crate::time_series::TimeSeries;

use enum_dispatch::enum_dispatch;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::borrow::Cow;

mod fft;
pub use fft::{FftwComplex, FftwFloat};

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
pub use power_trait::{PeriodogramPowerError, PeriodogramPowerTrait};

pub mod sin_cos_iterator;

/// Periodogram execution algorithm
#[enum_dispatch(PeriodogramPowerTrait<T>)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[serde(bound = "T: Float")]
#[non_exhaustive]
pub enum PeriodogramPower<T>
where
    T: Float,
{
    Fft(PeriodogramPowerFft<T>),
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
}

impl<'a, T> Periodogram<'a, T>
where
    T: Float,
{
    pub fn new(
        periodogram_power: PeriodogramPower<T>,
        freq_grid: Cow<'a, FreqGrid<T>>,
    ) -> Result<Self, PeriodogramPowerError> {
        if matches!(periodogram_power, PeriodogramPower::Fft(_))
            && !matches!(freq_grid.as_ref(), FreqGrid::ZeroBasedPow2(_))
        {
            return Err(PeriodogramPowerError::PeriodogramFftWrongFreqGrid);
        }
        Ok(Self {
            freq_grid,
            periodogram_power,
        })
    }

    pub fn from_t(
        periodogram_power: PeriodogramPower<T>,
        t: &[T],
        freq_grid_strategy: &'a FreqGridStrategy<T>,
    ) -> Result<Self, PeriodogramPowerError> {
        let zero_base = match periodogram_power {
            PeriodogramPower::Direct(_) => false,
            PeriodogramPower::Fft(_) => true,
        };
        Self::new(
            periodogram_power,
            freq_grid_strategy.freq_grid(t, zero_base),
        )
    }

    pub fn freq(&self, i: usize) -> T {
        self.freq_grid.get(i)
    }

    pub fn power(&self, ts: &mut TimeSeries<T>) -> Vec<T> {
        self.periodogram_power
            .power(&self.freq_grid, ts)
            .expect("Unexpected error from PeriodogrmPowerTrait::power")
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
        let fft = Periodogram::from_t(PeriodogramPowerFft::new().into(), &t, &freq_grid_strategy)
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
        let fft = Periodogram::from_t(PeriodogramPowerFft::new().into(), &t, &freq_grid_strategy)
            .unwrap()
            .power(&mut ts);

        assert_eq!(
            peak_indices_reverse_sorted(&fft)[..2],
            peak_indices_reverse_sorted(&direct)[..2]
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
        let fft = Periodogram::from_t(PeriodogramPowerFft::new().into(), &t, &freq_grid_strategy)
            .unwrap()
            .power(&mut ts);

        assert_eq!(
            peak_indices_reverse_sorted(&fft)[..2],
            peak_indices_reverse_sorted(&direct)[..2]
        );
    }

    #[test]
    fn arbitrary_vs_zero_based_pow2_for_direct() {
        let step = 0.1;
        let log2_size_m1 = 7;
        let size = (1 << log2_size_m1) + 1;

        let zero_based_grid = FreqGrid::zero_based_pow2(step, log2_size_m1);

        let freqs: Vec<_> = (0..size).map(|i| step * i as f64).collect();
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
}
