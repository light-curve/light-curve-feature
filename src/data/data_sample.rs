use crate::data::sorted_array::SortedArray;
use crate::float_trait::Float;
use crate::types::CowArray1;

use conv::prelude::*;
use ndarray::{s, Array1, ArrayView1, Zip};

/// A [`TimeSeries`] component
#[derive(Clone, Debug)]
pub struct DataSample<'a, T>
where
    T: Float,
{
    pub sample: CowArray1<'a, T>,
    sorted: Option<SortedArray<T>>,
    min: Option<T>,
    max: Option<T>,
    mean: Option<T>,
    median: Option<T>,
    std: Option<T>,
    std2: Option<T>,
}

macro_rules! data_sample_getter {
    ($attr: ident, $getter: ident, $func: expr, $method_sorted: ident) => {
        pub fn $getter(&mut self) -> T {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some(match self.sorted.as_ref() {
                        Some(sorted) => sorted.$method_sorted(),
                        None => $func(self),
                    });
                    self.$attr.unwrap()
                }
            }
        }
    };
    ($attr: ident, $getter: ident, $func: expr) => {
        pub fn $getter(&mut self) -> T {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some($func(self));
                    self.$attr.unwrap()
                }
            }
        }
    };
}

impl<'a, T> DataSample<'a, T>
where
    T: Float,
{
    pub fn new(sample: CowArray1<'a, T>) -> Self {
        Self {
            sample,
            sorted: None,
            min: None,
            max: None,
            mean: None,
            median: None,
            std: None,
            std2: None,
        }
    }

    pub fn as_slice(&mut self) -> &[T] {
        if !self.sample.is_standard_layout() {
            let owned: Array1<_> = self.sample.iter().copied().collect::<Vec<_>>().into();
            self.sample = owned.into();
        }
        self.sample.as_slice().unwrap()
    }

    pub fn get_sorted(&mut self) -> &SortedArray<T> {
        if self.sorted.is_none() {
            self.sorted = Some(self.sample.to_vec().into());
        }
        self.sorted.as_ref().unwrap()
    }

    fn set_min_max(&mut self) {
        let (min, max) =
            self.sample
                .slice(s![1..])
                .fold((self.sample[0], self.sample[0]), |(min, max), &x| {
                    if x > max {
                        (min, x)
                    } else if x < min {
                        (x, max)
                    } else {
                        (min, max)
                    }
                });
        self.min = Some(min);
        self.max = Some(max);
    }

    data_sample_getter!(
        min,
        get_min,
        |ds: &mut DataSample<'a, T>| {
            ds.set_min_max();
            ds.min.unwrap()
        },
        minimum
    );
    data_sample_getter!(
        max,
        get_max,
        |ds: &mut DataSample<'a, T>| {
            ds.set_min_max();
            ds.max.unwrap()
        },
        maximum
    );
    data_sample_getter!(mean, get_mean, |ds: &mut DataSample<'a, T>| {
        ds.sample.mean().expect("time series must be non-empty")
    });
    data_sample_getter!(median, get_median, |ds: &mut DataSample<'a, T>| {
        ds.get_sorted().median()
    });
    data_sample_getter!(std, get_std, |ds: &mut DataSample<'a, T>| {
        ds.get_std2().sqrt()
    });
    data_sample_getter!(std2, get_std2, |ds: &mut DataSample<'a, T>| {
        // Benchmarks show that it is faster than `ndarray::ArrayBase::var(T::one)`
        let mean = ds.get_mean();
        ds.sample
            .fold(T::zero(), |sum, &x| sum + (x - mean).powi(2))
            / (ds.sample.len() - 1).approx().unwrap()
    });

    pub fn signal_to_noise(&mut self, value: T) -> T {
        if self.get_std().is_zero() {
            T::zero()
        } else {
            (value - self.get_mean()) / self.get_std()
        }
    }

    /// Returns true if all values are equal. Always true for zero- or one- length
    pub fn is_all_same(&self) -> bool {
        if self.sample.is_empty() {
            return true;
        }
        if self.max.is_some() && self.max == self.min {
            return true;
        }
        if self.std2 == Some(T::zero()) {
            return true;
        }
        if let Some(sorted) = &self.sorted {
            return sorted[0] == sorted[sorted.len() - 1];
        }
        let x0 = self.sample[0];
        // all() returns true for the empty slice, i.e. single-point time series
        Zip::from(self.sample.slice(s![1..])).all(|&x| x == x0)
    }
}

impl<'a, T> From<SortedArray<T>> for DataSample<'a, T>
where
    T: Float,
{
    fn from(sorted: SortedArray<T>) -> Self {
        let sample = sorted.0.clone().into();
        Self {
            sample,
            sorted: Some(sorted),
            min: None,
            max: None,
            median: None,
            mean: None,
            std: None,
            std2: None,
        }
    }
}

impl<'a, T, Slice: ?Sized> From<&'a Slice> for DataSample<'a, T>
where
    T: Float,
    Slice: AsRef<[T]>,
{
    fn from(s: &'a Slice) -> Self {
        ArrayView1::from(s).into()
    }
}

impl<'a, T> From<Vec<T>> for DataSample<'a, T>
where
    T: Float,
{
    fn from(v: Vec<T>) -> Self {
        Array1::from(v).into()
    }
}

impl<'a, T> From<ArrayView1<'a, T>> for DataSample<'a, T>
where
    T: Float,
{
    fn from(a: ArrayView1<'a, T>) -> Self {
        Self::new(a.into())
    }
}

impl<'a, T> From<Array1<T>> for DataSample<'a, T>
where
    T: Float,
{
    fn from(a: Array1<T>) -> Self {
        Self::new(a.into())
    }
}

impl<'a, T> From<CowArray1<'a, T>> for DataSample<'a, T>
where
    T: Float,
{
    fn from(a: CowArray1<'a, T>) -> Self {
        Self::new(a)
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    macro_rules! data_sample_test {
        ($name: ident, $method: ident, $desired: literal, $x: tt $(,)?) => {
            #[test]
            fn $name() {
                let x = $x;
                let desired = $desired;

                let mut ds: DataSample<_> = DataSample::from(&x);
                assert_relative_eq!(ds.$method(), desired, epsilon = 1e-6);
                assert_relative_eq!(ds.$method(), desired, epsilon = 1e-6);

                let mut ds: DataSample<_> = DataSample::from(&x);
                ds.get_sorted();
                assert_relative_eq!(ds.$method(), desired, epsilon = 1e-6);
                assert_relative_eq!(ds.$method(), desired, epsilon = 1e-6);
            }
        };
    }

    data_sample_test!(
        data_sample_min,
        get_min,
        -7.79420906,
        [3.92948846, 3.28436964, 6.73375373, -7.79420906, -7.23407407],
    );

    data_sample_test!(
        data_sample_max,
        get_max,
        6.73375373,
        [3.92948846, 3.28436964, 6.73375373, -7.79420906, -7.23407407],
    );

    data_sample_test!(
        data_sample_mean,
        get_mean,
        -0.21613426,
        [3.92948846, 3.28436964, 6.73375373, -7.79420906, -7.23407407],
    );

    data_sample_test!(
        data_sample_median_odd,
        get_median,
        3.28436964,
        [3.92948846, 3.28436964, 6.73375373, -7.79420906, -7.23407407],
    );

    data_sample_test!(
        data_sample_median_even,
        get_median,
        5.655794743124782,
        [9.47981408, 3.86815751, 9.90299294, -2.986894, 7.44343197, 1.52751816],
    );

    data_sample_test!(
        data_sample_std,
        get_std,
        6.7900544035968435,
        [3.92948846, 3.28436964, 6.73375373, -7.79420906, -7.23407407],
    );

    /// https://github.com/light-curve/light-curve-feature/issues/95
    #[test]
    fn std2_overflow() {
        const N: usize = (1 << 24) + 2;
        // Such a large integer cannot be represented as a float32
        let x = Array1::linspace(0.0_f32, 1.0, N);
        let mut ds = DataSample::new(x.into());
        // This should not panic
        let _std2 = ds.get_std2();
    }
}
