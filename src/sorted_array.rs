use crate::error::SortedArrayError;
use crate::float_trait::Float;
use conv::prelude::*;
use ndarray::{Array1, ArrayView1};
use schemars::{JsonSchema, Schema, SchemaGenerator};
use serde::{Deserialize, Serialize};
use std::borrow::Cow;
use std::ops::Deref;

// Underlying array is guaranteed to be sorted and contiguous
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SortedArray<T>(pub Array1<T>);

impl<T> SortedArray<T>
where
    T: Float,
{
    pub fn from_sorted(sorted_array: impl Into<Array1<T>>) -> Result<Self, SortedArrayError> {
        let sorted_array = sorted_array.into();
        if sorted_array
            .as_slice()
            .ok_or(SortedArrayError::NonContiguous)?
            .is_sorted()
        {
            Ok(Self(sorted_array))
        } else {
            Err(SortedArrayError::Unsorted)
        }
    }

    pub fn maximum(&self) -> T {
        self[self.len() - 1]
    }

    pub fn minimum(&self) -> T {
        *self.first().unwrap()
    }

    pub fn median(&self) -> T {
        assert_ne!(self.len(), 0);
        let i = (self.len() - 1) / 2;
        if self.len() % 2 == 0 {
            T::half() * (self[i] + self[i + 1])
        } else {
            self[i]
        }
    }

    // R-5 from https://en.wikipedia.org/wiki/Quantile
    pub fn ppf(&self, q: f32) -> T {
        assert_ne!(self.len(), 0);
        assert!(
            (0.0..=1.0).contains(&q),
            "quantile should be between zero and unity"
        );
        let h = (self.len() as f32) * q - 0.5;
        let h_floor = h.floor();
        if h_floor < 0.0 {
            self.minimum()
        } else {
            #[allow(clippy::cast_sign_loss)]
            let i = h_floor as usize;
            if i >= self.len() - 1 {
                self.maximum()
            } else {
                self[i] + (h - h_floor).value_as::<T>().unwrap() * (self[i + 1] - self[i])
            }
        }
    }
}

impl<T> From<Vec<T>> for SortedArray<T>
where
    T: Float,
{
    fn from(mut v: Vec<T>) -> Self {
        v[..].sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        Self(Array1::from_vec(v))
    }
}

impl<T> From<&[T]> for SortedArray<T>
where
    T: Float,
{
    fn from(s: &[T]) -> Self {
        s.to_vec().into()
    }
}

impl<T> From<ArrayView1<'_, T>> for SortedArray<T>
where
    T: Float,
{
    fn from(v: ArrayView1<'_, T>) -> Self {
        v.to_vec().into()
    }
}

impl<T> Deref for SortedArray<T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.0.as_slice().unwrap()
    }
}

impl<T> AsRef<[T]> for SortedArray<T> {
    fn as_ref(&self) -> &[T] {
        self
    }
}

impl<T> JsonSchema for SortedArray<T>
where
    T: JsonSchema,
{
    fn inline_schema() -> bool {
        true
    }

    fn schema_name() -> Cow<'static, str> {
        Cow::Borrowed("SortedArray")
    }

    fn json_schema(generator: &mut SchemaGenerator) -> Schema {
        <[T] as JsonSchema>::json_schema(generator)
    }
}

#[allow(clippy::float_cmp)]
#[cfg(test)]
mod tests {
    use super::*;

    use light_curve_common::all_close;

    #[test]
    fn median_is_ppf_half() {
        for i in 0..10 {
            let a: SortedArray<f64> = (0..100 + i)
                .map(|_| rand::random())
                .collect::<Vec<_>>()
                .into();
            assert_eq!(a.median(), a.ppf(0.5));
        }
    }

    #[test]
    fn minimum_is_ppf_zero() {
        for i in 0..10 {
            let a: SortedArray<f64> = (0..100 + i)
                .map(|_| rand::random())
                .collect::<Vec<_>>()
                .into();
            assert_eq!(a.minimum(), a.ppf(0.0));
        }
    }

    #[test]
    fn maximum_is_ppf_unity() {
        for i in 0..10 {
            let a: SortedArray<f32> = (0..100 + i)
                .map(|_| rand::random())
                .collect::<Vec<_>>()
                .into();
            assert_eq!(a.maximum(), a.ppf(1.0));
        }
    }

    #[test]
    fn ppf_tenths() {
        let a = SortedArray::from_sorted(Array1::linspace(0.0, 1.0, 11)).unwrap();
        let q = Array1::linspace(0.0, 1.0, 11);
        let actual: Vec<_> = q.iter().map(|&q| a.ppf(q)).collect();
        // from scipy.stats.mstats import mquantiles
        // mquantiles(np.linspace(0, 1, 11), prob=np.linspace(0, 1, 11), alphap=0.5, betap=0.5)
        let desired = [0., 0.06, 0.17, 0.28, 0.39, 0.5, 0.61, 0.72, 0.83, 0.94, 1.];
        all_close(&actual, &desired, 1e-7);
    }
}
