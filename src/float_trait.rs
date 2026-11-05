use crate::periodogram::FftFloat;
use num_complex::Complex;

#[cfg(feature = "fftw")]
use crate::periodogram::FftwFloat;

use conv::prelude::*;
use lazy_static::lazy_static;
use ndarray::{Array0, ScalarOperand};
use num_traits::{FromPrimitive, float::Float as NumFloat, float::FloatConst};
use schemars::JsonSchema;
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::cmp::PartialOrd;
use std::fmt::{Debug, Display, LowerExp};
use std::hash::{Hash, Hasher};
use std::iter::Sum;
use std::ops::{AddAssign, DivAssign, MulAssign};

lazy_static! {
    static ref ARRAY0_UNITY_F32: Array0<f32> = Array0::from_elem((), 1.0);
}

lazy_static! {
    static ref ARRAY0_UNITY_F64: Array0<f64> = Array0::from_elem((), 1.0);
}

// Note: Two trait definitions are needed because Rust doesn't support conditional trait bounds.
// The only difference is the `+ FftwFloat` bound when the fftw feature is enabled.

/// Floating number trait, it is implemented for [f32] and [f64] only
#[cfg(feature = "fftw")]
#[cfg_attr(feature = "fast-math", reassoc::algebraic_float)]
pub trait Float:
    'static
    + Sized
    + NumFloat
    + FloatConst
    + FromPrimitive
    + PartialOrd
    + Sum
    + ValueFrom<u32>
    + ValueFrom<usize>
    + ValueFrom<f32>
    + ValueInto<f64>
    + ApproxFrom<usize>
    + ApproxFrom<f64>
    + ApproxInto<u32, RoundToNearest>
    + ApproxInto<usize, RoundToNearest>
    + ApproxInto<f32>
    + ApproxInto<f64>
    + Clone
    + Copy
    + Send
    + Sync
    + AddAssign
    + MulAssign
    + DivAssign
    + ScalarOperand
    + Display
    + Debug
    + LowerExp
    + FftFloat<Complex = Complex<Self>>
    + FftwFloat
    + DeserializeOwned
    + Serialize
    + JsonSchema
{
    fn half() -> Self;
    fn two() -> Self;
    fn three() -> Self;
    fn four() -> Self;
    fn five() -> Self;
    fn ten() -> Self;
    fn hundred() -> Self;
    fn array0_unity() -> &'static Array0<Self>;
}

/// Floating number trait, it is implemented for [f32] and [f64] only
#[cfg_attr(feature = "fast-math", reassoc::algebraic_float)]
#[cfg(not(feature = "fftw"))]
pub trait Float:
    'static
    + Sized
    + NumFloat
    + FloatConst
    + FromPrimitive
    + PartialOrd
    + Sum
    + ValueFrom<u32>
    + ValueFrom<usize>
    + ValueFrom<f32>
    + ValueInto<f64>
    + ApproxFrom<usize>
    + ApproxFrom<f64>
    + ApproxInto<u32, RoundToNearest>
    + ApproxInto<usize, RoundToNearest>
    + ApproxInto<f32>
    + ApproxInto<f64>
    + Clone
    + Copy
    + Send
    + Sync
    + AddAssign
    + MulAssign
    + DivAssign
    + ScalarOperand
    + Display
    + Debug
    + LowerExp
    + FftFloat<Complex = Complex<Self>>
    + DeserializeOwned
    + Serialize
    + JsonSchema
{
    fn half() -> Self;
    fn two() -> Self;
    fn three() -> Self;
    fn four() -> Self;
    fn five() -> Self;
    fn ten() -> Self;
    fn hundred() -> Self;
    fn array0_unity() -> &'static Array0<Self>;
}

impl Float for f32 {
    #[inline]
    fn half() -> Self {
        0.5
    }

    #[inline]
    fn two() -> Self {
        2.0
    }

    #[inline]
    fn three() -> Self {
        3.0
    }

    #[inline]
    fn four() -> Self {
        4.0
    }

    #[inline]
    fn five() -> Self {
        5.0
    }

    #[inline]
    fn ten() -> Self {
        10.0
    }

    #[inline]
    fn hundred() -> Self {
        100.0
    }

    fn array0_unity() -> &'static Array0<Self> {
        &ARRAY0_UNITY_F32
    }
}

impl Float for f64 {
    #[inline]
    fn half() -> Self {
        0.5
    }

    #[inline]
    fn two() -> Self {
        2.0
    }

    #[inline]
    fn three() -> Self {
        3.0
    }

    #[inline]
    fn four() -> Self {
        4.0
    }

    #[inline]
    fn five() -> Self {
        5.0
    }

    #[inline]
    fn ten() -> Self {
        10.0
    }

    #[inline]
    fn hundred() -> Self {
        100.0
    }

    fn array0_unity() -> &'static Array0<Self> {
        &ARRAY0_UNITY_F64
    }
}

/// Float equality treating all NaNs as equal to each other, so it is an equivalence relation
///
/// Use it together with [hash_float] to implement [Eq] and [Hash] for types holding floats.
pub(crate) fn float_total_eq<T: Float>(a: T, b: T) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// Element-wise [float_total_eq] for slices
pub(crate) fn float_slice_total_eq<T: Float>(a: &[T], b: &[T]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| float_total_eq(x, y))
}

/// Hash a float consistently with [float_total_eq]
pub(crate) fn hash_float<T: Float, H: Hasher>(x: T, state: &mut H) {
    // All NaNs are equal to each other, and so are both zeros, so canonicalize them
    let x = if x.is_nan() {
        T::nan()
    } else if x == T::zero() {
        T::zero()
    } else {
        x
    };
    x.integer_decode().hash(state);
}

/// Hash a float slice consistently with [float_slice_total_eq]
pub(crate) fn hash_float_slice<T: Float, H: Hasher>(a: &[T], state: &mut H) {
    a.len().hash(state);
    for &x in a {
        hash_float(x, state);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::DefaultHasher;

    fn hash_of(x: f64) -> u64 {
        let mut hasher = DefaultHasher::new();
        hash_float(x, &mut hasher);
        hasher.finish()
    }

    #[test]
    fn total_eq_and_hash_agree() {
        let values = [
            0.0,
            -0.0,
            1.0,
            -1.0,
            f64::NAN,
            -f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
        ];
        for &a in &values {
            assert!(float_total_eq(a, a));
            for &b in &values {
                if float_total_eq(a, b) {
                    assert_eq!(hash_of(a), hash_of(b), "{a} and {b}");
                }
            }
        }
        assert!(float_total_eq(0.0, -0.0));
        assert!(float_total_eq(f64::NAN, -f64::NAN));
        assert!(!float_total_eq(1.0, f64::NAN));
        assert_ne!(hash_of(1.0), hash_of(2.0));
    }
}
