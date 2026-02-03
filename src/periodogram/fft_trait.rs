//! Abstract FFT trait and associated types for periodogram computation
//!
//! This module provides a unified interface for FFT backends (FFTW and RustFFT).

use num_complex::Complex;
use std::collections::HashMap;
use std::fmt;

/// Trait for complex numbers produced by FFT
pub trait FftComplex<T>: Send {
    /// Real part
    fn get_re(&self) -> T;

    /// Imaginary part
    fn get_im(&self) -> T;

    /// Create a zero complex number
    fn zero() -> Self
    where
        Self: Sized;
}

impl<T: Copy + Send + num_traits::Zero> FftComplex<T> for Complex<T> {
    #[inline]
    fn get_re(&self) -> T {
        self.re
    }

    #[inline]
    fn get_im(&self) -> T {
        self.im
    }

    #[inline]
    fn zero() -> Self {
        Complex::new(T::zero(), T::zero())
    }
}

/// Trait for real-to-complex FFT floating point types.
///
/// This trait is implemented for f32 and f64. It provides the basic bounds needed
/// for FFT operations, with additional backend-specific bounds checked where needed.
pub trait FftFloat: Copy + Clone + Send + Sync + 'static + num_traits::Zero {
    /// Complex number type for FFT output
    type Complex: FftComplex<Self>;
}

impl FftFloat for f32 {
    type Complex = Complex<f32>;
}

impl FftFloat for f64 {
    type Complex = Complex<f64>;
}

/// Trait for input arrays to FFT
pub trait FftInputArray<T>: AsMut<[T]> {
    /// Create a new array with the given size
    fn new_with_size(n: usize) -> Self;

    /// Get the length of the array
    fn len(&self) -> usize;
}

/// Trait for output arrays from FFT (complex values)
pub trait FftOutputArray<T: FftFloat>: AsMut<[T::Complex]> {
    /// Create a new array with the given size
    fn new_with_size(n: usize) -> Self;

    /// Get an iterator over the complex values
    fn iter(&self) -> impl Iterator<Item = &T::Complex>;
}

/// Abstract trait for real-to-complex FFT implementations
pub trait Fft<T: FftFloat>: fmt::Debug + Send {
    /// Input array type
    type InputArray: FftInputArray<T> + Send;
    /// Output array type
    type OutputArray: FftOutputArray<T> + Send;

    /// Create a new FFT instance
    fn new() -> Self;

    /// Perform real-to-complex FFT
    ///
    /// # Arguments
    /// * `x` - Input real array (may be modified)
    /// * `y` - Output complex array
    fn fft(&mut self, x: &mut Self::InputArray, y: &mut Self::OutputArray);
}

/// Cache for FFT arrays to avoid repeated allocations
pub struct FftArraysMap<T, F>
where
    T: FftFloat,
    F: Fft<T>,
{
    arrays: HashMap<usize, FftArrays<T, F>>,
}

/// A set of arrays for FFT computation in periodogram
pub struct FftArrays<T, F>
where
    T: FftFloat,
    F: Fft<T>,
{
    pub x_sch: F::InputArray,
    pub y_sch: F::OutputArray,
    pub x_sc2: F::InputArray,
    pub y_sc2: F::OutputArray,
}

impl<T, F> FftArrays<T, F>
where
    T: FftFloat,
    F: Fft<T>,
{
    fn new(n: usize) -> Self {
        let c_n = n / 2 + 1;
        Self {
            x_sch: F::InputArray::new_with_size(n),
            y_sch: F::OutputArray::new_with_size(c_n),
            x_sc2: F::InputArray::new_with_size(n),
            y_sc2: F::OutputArray::new_with_size(c_n),
        }
    }
}

impl<T, F> fmt::Debug for FftArrays<T, F>
where
    T: FftFloat,
    F: Fft<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FftArrays(n = {})", self.x_sch.len())
    }
}

impl<T, F> FftArraysMap<T, F>
where
    T: FftFloat,
    F: Fft<T>,
{
    pub fn new() -> Self {
        Self {
            arrays: HashMap::new(),
        }
    }

    pub fn get(&mut self, n: usize) -> &mut FftArrays<T, F> {
        self.arrays.entry(n).or_insert_with(|| FftArrays::new(n))
    }
}

impl<T, F> fmt::Debug for FftArraysMap<T, F>
where
    T: FftFloat,
    F: Fft<T>,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "FftArraysMap<{}, {}>",
            std::any::type_name::<T>(),
            std::any::type_name::<F>()
        )
    }
}
