//! RustFFT-based FFT implementation for periodogram computation
//!
//! This module provides a pure Rust FFT backend using the `realfft` crate.

use crate::periodogram::fft_trait::{Fft, FftComplex, FftFloat, FftInputArray, FftOutputArray};

use num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// Marker trait for types that can be used with RustFFT.
///
/// This trait is implemented only for f32 and f64.
pub trait RustFftFloat: FftFloat {}
impl RustFftFloat for f32 {}
impl RustFftFloat for f64 {}

/// Input array wrapper for RustFFT (plain Vec)
pub struct RustFftInputArray<T>(pub(crate) Vec<T>);

impl<T: FftFloat> FftInputArray<T> for RustFftInputArray<T> {
    fn new_with_size(n: usize) -> Self {
        Self(vec![T::zero(); n])
    }

    fn len(&self) -> usize {
        self.0.len()
    }
}

impl<T> AsMut<[T]> for RustFftInputArray<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

/// Output array wrapper for RustFFT (plain Vec of Complex)
pub struct RustFftOutputArray<T: FftFloat>(pub(crate) Vec<T::Complex>);

impl<T: FftFloat> FftOutputArray<T> for RustFftOutputArray<T>
where
    T::Complex: Clone,
{
    fn new_with_size(n: usize) -> Self {
        Self(vec![<T::Complex as FftComplex<T>>::zero(); n])
    }

    fn iter(&self) -> impl Iterator<Item = &T::Complex> {
        self.0.iter()
    }
}

impl<T: FftFloat> AsMut<[T::Complex]> for RustFftOutputArray<T> {
    fn as_mut(&mut self) -> &mut [T::Complex] {
        &mut self.0
    }
}

macro_rules! impl_rustfft {
    ($name:ident, $float:ty) => {
        #[doc = concat!("RustFFT-based real-to-complex FFT implementation for ", stringify!($float))]
        pub struct $name {
            plans: HashMap<usize, Arc<dyn RealToComplex<$float>>>,
            scratch: HashMap<usize, Vec<Complex<$float>>>,
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                write!(f, stringify!($name))
            }
        }

        impl $name {
            fn get_plan(&mut self, n: usize) -> &Arc<dyn RealToComplex<$float>> {
                self.plans.entry(n).or_insert_with(|| {
                    let mut planner = RealFftPlanner::new();
                    let plan = planner.plan_fft_forward(n);
                    let scratch_len = plan.get_scratch_len();
                    self.scratch
                        .insert(n, vec![Complex::new(0.0, 0.0); scratch_len]);
                    plan
                })
            }
        }

        impl Fft<$float> for $name {
            type InputArray = RustFftInputArray<$float>;
            type OutputArray = RustFftOutputArray<$float>;

            fn new() -> Self {
                Self {
                    plans: HashMap::new(),
                    scratch: HashMap::new(),
                }
            }

            fn fft(&mut self, x: &mut Self::InputArray, y: &mut Self::OutputArray) {
                let n = x.len();
                let plan = self.get_plan(n).clone();
                let scratch = self.scratch.get_mut(&n).unwrap();
                plan.process_with_scratch(&mut x.0, &mut y.0, scratch)
                    .expect("RustFFT error");
            }
        }
    };
}

impl_rustfft!(RustFft32, f32);
impl_rustfft!(RustFft64, f64);

/// Trait that maps Float types to their concrete RustFFT implementations.
///
/// This trait is implemented for f32 and f64 only, providing the mapping
/// to the concrete FFT implementation types.
pub trait RustFftImpl: FftFloat {
    /// The concrete FFT implementation type for this float type
    type Impl: Fft<Self> + Send;
}

impl RustFftImpl for f32 {
    type Impl = RustFft32;
}

impl RustFftImpl for f64 {
    type Impl = RustFft64;
}

/// RustFFT-based real-to-complex FFT implementation.
///
/// This struct wraps the concrete FFT implementations (RustFft32 and RustFft64)
/// and provides a unified generic interface.
pub struct RustFft<T: RustFftImpl> {
    inner: T::Impl,
}

impl<T: RustFftImpl> fmt::Debug for RustFft<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "RustFft<{}>", std::any::type_name::<T>())
    }
}

impl<T: RustFftImpl> Clone for RustFft<T> {
    fn clone(&self) -> Self {
        // Create a fresh instance - FFT plans will be regenerated as needed.
        // This is fine since the FFT instances are stored in ThreadLocal storage
        // and each thread maintains its own independent FFT state.
        Self::new()
    }
}

impl<T: RustFftImpl> Fft<T> for RustFft<T>
where
    <T::Impl as Fft<T>>::InputArray: Send,
    <T::Impl as Fft<T>>::OutputArray: Send,
{
    type InputArray = <T::Impl as Fft<T>>::InputArray;
    type OutputArray = <T::Impl as Fft<T>>::OutputArray;

    fn new() -> Self {
        Self {
            inner: T::Impl::new(),
        }
    }

    fn fft(&mut self, x: &mut Self::InputArray, y: &mut Self::OutputArray) {
        self.inner.fft(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::periodogram::fft_trait::FftComplex;
    use light_curve_common::all_close;
    use std::f64::consts::PI;

    #[test]
    fn unity_f64() {
        const N: usize = 1024;

        let mut x = RustFftInputArray::new_with_size(N);
        for a in x.as_mut().iter_mut() {
            *a = 1.0_f64;
        }

        let mut fft: RustFft<f64> = RustFft::new();

        let mut y: RustFftOutputArray<f64> = RustFftOutputArray::new_with_size(N / 2 + 1);
        fft.fft(&mut x, &mut y);

        assert!((y.0[0].get_re() - 1024.0_f64).abs() < 1e-10);
        assert!(y.0[0].get_im().abs() < 1e-10);

        let (re, im): (Vec<_>, Vec<_>) = y.iter().map(|c| (c.get_re(), c.get_im())).unzip();
        all_close(&re[1..], &[0.0; 512], 1e-10);
        all_close(&im[1..], &[0.0; 512], 1e-10);
    }

    #[test]
    fn unity_f32() {
        const N: usize = 1024;

        let mut x = RustFftInputArray::new_with_size(N);
        for a in x.as_mut().iter_mut() {
            *a = 1.0_f32;
        }

        let mut fft: RustFft<f32> = RustFft::new();

        let mut y: RustFftOutputArray<f32> = RustFftOutputArray::new_with_size(N / 2 + 1);
        fft.fft(&mut x, &mut y);

        assert!((y.0[0].get_re() - 1024.0_f32).abs() < 1e-4);
        assert!(y.0[0].get_im().abs() < 1e-4);
    }

    #[test]
    fn numpy_compr() {
        const N: usize = 32;

        let mut x = RustFftInputArray::new_with_size(N);
        for i in 0..N {
            x.as_mut()[i] = f64::sin(2.0 * PI * 0.27 * (i as f64));
        }

        let mut fft: RustFft<f64> = RustFft::new();
        let mut y = RustFftOutputArray::new_with_size(N / 2 + 1);
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

        all_close(&actual_re, &desired_re, 1e-6);
        all_close(&actual_im, &desired_im, 1e-6);
    }

    #[test]
    fn rustfft_clone() {
        // Test that cloning RustFft creates a new instance
        let fft: RustFft<f64> = RustFft::new();
        let fft2 = fft.clone();

        // Both should be valid and able to perform FFT
        let mut fft2 = fft2;
        let mut x = RustFftInputArray::new_with_size(64);
        for i in 0..64 {
            x.as_mut()[i] = (i as f64).sin();
        }
        let mut y = RustFftOutputArray::new_with_size(33);
        fft2.fft(&mut x, &mut y);

        // Result should be valid
        assert!(
            y.iter()
                .all(|c| c.get_re().is_finite() && c.get_im().is_finite())
        );
    }

    #[test]
    fn rustfft_debug() {
        let fft: RustFft<f64> = RustFft::new();
        let debug_str = format!("{:?}", fft);
        assert!(debug_str.contains("RustFft"));
        assert!(debug_str.contains("f64"));
    }

    #[test]
    fn rustfft32_debug() {
        let fft = RustFft32::new();
        let debug_str = format!("{:?}", fft);
        assert!(debug_str.contains("RustFft32"));
    }

    #[test]
    fn rustfft64_debug() {
        let fft = RustFft64::new();
        let debug_str = format!("{:?}", fft);
        assert!(debug_str.contains("RustFft64"));
    }

    #[test]
    fn input_array_len() {
        let x: RustFftInputArray<f64> = RustFftInputArray::new_with_size(128);
        assert_eq!(x.len(), 128);
    }

    #[test]
    fn input_array_as_mut() {
        let mut x: RustFftInputArray<f64> = RustFftInputArray::new_with_size(10);
        let slice = x.as_mut();
        for (i, val) in slice.iter_mut().enumerate() {
            *val = i as f64;
        }
        assert_eq!(x.0[5], 5.0);
    }

    #[test]
    fn output_array_iter() {
        let y: RustFftOutputArray<f64> = RustFftOutputArray::new_with_size(5);
        assert_eq!(y.iter().count(), 5);
        // All zeros initially
        assert!(y.iter().all(|c| c.get_re() == 0.0 && c.get_im() == 0.0));
    }

    #[test]
    fn output_array_as_mut() {
        let mut y: RustFftOutputArray<f64> = RustFftOutputArray::new_with_size(5);
        let slice = y.as_mut();
        slice[2] = Complex::new(1.0, 2.0);
        assert_eq!(y.0[2].get_re(), 1.0);
        assert_eq!(y.0[2].get_im(), 2.0);
    }

    #[test]
    fn fft_plan_caching() {
        // Test that FFT plans are cached and reused
        let mut fft: RustFft<f64> = RustFft::new();

        // First FFT
        let mut x1 = RustFftInputArray::new_with_size(64);
        x1.as_mut()[0] = 1.0;
        let mut y1 = RustFftOutputArray::new_with_size(33);
        fft.fft(&mut x1, &mut y1);

        // Second FFT with same size (should reuse plan)
        let mut x2 = RustFftInputArray::new_with_size(64);
        x2.as_mut()[0] = 1.0;
        let mut y2 = RustFftOutputArray::new_with_size(33);
        fft.fft(&mut x2, &mut y2);

        // Results should be the same
        for (c1, c2) in y1.iter().zip(y2.iter()) {
            assert!((c1.get_re() - c2.get_re()).abs() < 1e-10);
            assert!((c1.get_im() - c2.get_im()).abs() < 1e-10);
        }
    }

    #[test]
    fn fft_different_sizes() {
        let mut fft: RustFft<f64> = RustFft::new();

        // Test various sizes
        for &n in &[8, 16, 32, 64, 128, 256] {
            let mut x = RustFftInputArray::new_with_size(n);
            for i in 0..n {
                x.as_mut()[i] = (i as f64 * 0.1).sin();
            }
            let mut y = RustFftOutputArray::new_with_size(n / 2 + 1);
            fft.fft(&mut x, &mut y);

            // Verify result is valid
            assert!(
                y.iter()
                    .all(|c| c.get_re().is_finite() && c.get_im().is_finite())
            );
        }
    }
}
