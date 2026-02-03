//! FFTW-based FFT implementation for periodogram computation

use crate::periodogram::fft_trait::{Fft, FftFloat, FftInputArray, FftOutputArray};

pub use fftw::array::{AlignedAllocable, AlignedVec};
use fftw::error::Result;
pub use fftw::plan::{Plan, Plan32, Plan64, R2CPlan};
use fftw::types::Flag;
use num_complex::Complex;
use std::collections::HashMap;
use std::fmt;

/// Floating number trait for `fftw` crate
///
/// This trait extends `FftFloat` with FFTW-specific requirements for aligned memory allocation.
pub trait FftwFloat: FftFloat<Complex: AlignedAllocable + Send> + AlignedAllocable + Send {
    type FftwPlan: R2CPlan<Real = Self, Complex = Self::Complex> + Send;
}

impl FftwFloat for f32 {
    type FftwPlan = Plan<f32, Complex<f32>, Plan32>;
}

impl FftwFloat for f64 {
    type FftwPlan = Plan<f64, Complex<f64>, Plan64>;
}

/// Input array wrapper for FFTW (AlignedVec)
pub struct FftwInputArray<T: FftwFloat>(pub AlignedVec<T>);

impl<T: FftwFloat> FftInputArray<T> for FftwInputArray<T> {
    fn new_with_size(n: usize) -> Self {
        Self(AlignedVec::new(n))
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T: FftwFloat> AsMut<[T]> for FftwInputArray<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T: FftwFloat> fmt::Debug for FftwInputArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FftwInputArray(len = {})", self.0.len())
    }
}

/// Output array wrapper for FFTW (AlignedVec of Complex)
pub struct FftwOutputArray<T: FftwFloat>(pub AlignedVec<T::Complex>);

impl<T: FftwFloat> FftOutputArray<T> for FftwOutputArray<T> {
    fn new_with_size(n: usize) -> Self {
        Self(AlignedVec::new(n))
    }

    fn iter(&self) -> impl Iterator<Item = &T::Complex> {
        self.0.iter()
    }
}

impl<T: FftwFloat> AsMut<[T::Complex]> for FftwOutputArray<T> {
    fn as_mut(&mut self) -> &mut [T::Complex] {
        &mut self.0
    }
}

impl<T: FftwFloat> fmt::Debug for FftwOutputArray<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FftwOutputArray(len = {})", self.0.len())
    }
}

/// FFTW-based real-to-complex FFT implementation
pub struct FftwFft<T>
where
    T: FftwFloat,
    T::Complex: AlignedAllocable,
{
    plans: HashMap<usize, T::FftwPlan>,
}

impl<T> fmt::Debug for FftwFft<T>
where
    T: FftwFloat,
    T::Complex: AlignedAllocable,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}<{}>",
            std::any::type_name::<Self>(),
            std::any::type_name::<T>()
        )
    }
}

impl<T> Clone for FftwFft<T>
where
    T: FftwFloat,
    T::Complex: AlignedAllocable,
{
    fn clone(&self) -> Self {
        // Create a fresh instance - FFT plans will be regenerated as needed.
        // This is fine since the FFT instances are stored in ThreadLocal storage
        // and each thread maintains its own independent FFT state.
        Self::new()
    }
}

impl<T> FftwFft<T>
where
    T: FftwFloat,
    T::Complex: AlignedAllocable,
{
    fn flags(n: usize) -> Flag {
        const MAX_N_TO_MEASURE: usize = 1 << 12; // It takes ~3s to measure
        let mut flag = Flag::DESTROYINPUT;
        if n <= MAX_N_TO_MEASURE {
            flag.insert(Flag::MEASURE);
        } else {
            flag.insert(Flag::ESTIMATE);
        }
        flag
    }

    fn get_plan(&mut self, n: usize) -> &mut T::FftwPlan {
        self.plans
            .entry(n)
            .or_insert_with(|| R2CPlan::aligned(&[n], Self::flags(n)).unwrap())
    }

    /// Perform real-to-complex FFT using raw AlignedVec (for backwards compatibility)
    pub fn fft_raw(&mut self, x: &mut AlignedVec<T>, y: &mut AlignedVec<T::Complex>) -> Result<()> {
        let n = x.len();
        self.get_plan(n).r2c(x, y)?;
        Ok(())
    }
}

impl<T: FftwFloat> Fft<T> for FftwFft<T>
where
    T::Complex: AlignedAllocable,
{
    type InputArray = FftwInputArray<T>;
    type OutputArray = FftwOutputArray<T>;

    fn new() -> Self {
        Self {
            plans: HashMap::new(),
        }
    }

    fn fft(&mut self, x: &mut Self::InputArray, y: &mut Self::OutputArray) {
        let n = x.len();
        self.get_plan(n).r2c(&mut x.0, &mut y.0).unwrap();
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
mod tests {
    use super::*;
    use crate::periodogram::fft_trait::FftComplex;
    use light_curve_common::all_close;
    use std::f64::consts::PI;

    #[test]
    #[allow(clippy::float_cmp)]
    fn unity() {
        const N: usize = 1024;

        let mut x = FftwInputArray::new_with_size(N);
        for a in x.as_mut().iter_mut() {
            *a = 1.0;
        }

        let mut fft = FftwFft::new();

        let mut y: FftwOutputArray<f64> = FftwOutputArray::new_with_size(N / 2 + 1);
        fft.fft(&mut x, &mut y);

        assert_eq!(y.0[0].get_re(), 1024.0_f64);
        assert_eq!(y.0[0].get_im(), 0.0_f64);

        let (re, im): (Vec<_>, Vec<_>) = y.iter().map(|c| (c.get_re(), c.get_im())).unzip();
        all_close(&re[1..], &[0.0; 512], 1e-12);
        all_close(&im[1..], &[0.0; 512], 1e-12);
    }

    #[test]
    fn numpy_compr() {
        const N: usize = 32;

        let mut x = FftwInputArray::new_with_size(N);
        for i in 0..N {
            x.as_mut()[i] = f64::sin(2.0 * PI * 0.27 * (i as f64));
        }

        let mut fft = FftwFft::new();
        let mut y = FftwOutputArray::new_with_size(N / 2 + 1);
        fft.fft(&mut x, &mut y);

        let (actual_re, actual_im): (Vec<_>, Vec<_>) =
            y.iter().map(|c| (c.get_re(), c.get_im())).unzip();

        // np.fft.fft(np.sin(2 * np.pi * 0.27 * np.arange(32)))[:17]
        let desired_re = [
            1.10704834,
            1.1195868,
            1.15941438,
            1.23418408,
            1.36101006,
            1.57816611,
            1.98413371,
            2.92020198,
            6.86602938,
            -11.2588103,
            -2.77097256,
            -1.50267075,
            -1.01091584,
            -0.76502572,
            -0.63191196,
            -0.56424862,
            -0.54338985,
        ];
        let desired_im = [
            0.,
            0.06794917,
            0.14051614,
            0.22370033,
            0.32725189,
            0.47044727,
            0.70062801,
            1.17923298,
            3.07385847,
            -5.4167115,
            -1.38305977,
            -0.7445412,
            -0.46825362,
            -0.30311016,
            -0.18462464,
            -0.08785979,
            0.,
        ];

        all_close(&actual_re, &desired_re, 1e-8);
        all_close(&actual_im, &desired_im, 1e-8);
    }
}
