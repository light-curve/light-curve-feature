//! Sine-cosine iterator implementations

use crate::float_trait::Float;

pub enum SinCosIterator<'a, T> {
    LinearGrid(RecurrentSinCos<T>),
    Values(ValueSinCosIterator<'a, T>),
}

impl<'a, T: Float> SinCosIterator<'a, T> {
    pub fn from_angles(iter: impl Iterator<Item = T> + 'a) -> Self {
        Self::Values(ValueSinCosIterator::new(iter))
    }

    pub fn from_linear_grid(first: T, step: T) -> Self {
        let rec_sin_cos = RecurrentSinCos::new(first, step);
        Self::LinearGrid(rec_sin_cos)
    }
}

impl<'a, T> Iterator for SinCosIterator<'a, T>
where
    T: Float,
{
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            SinCosIterator::LinearGrid(iter) => iter.next(),
            SinCosIterator::Values(iter) => iter.next(),
        }
    }
}

pub struct ValueSinCosIterator<'a, T> {
    value_iter: Box<dyn Iterator<Item = T> + 'a>,
}

impl<'a, T> ValueSinCosIterator<'a, T> {
    pub fn new(value_iter: impl Iterator<Item = T> + 'a) -> Self {
        Self {
            value_iter: Box::new(value_iter),
        }
    }
}

impl<'a, T> Iterator for ValueSinCosIterator<'a, T>
where
    T: Float,
{
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        self.value_iter.next().map(T::sin_cos)
    }
}

impl<T> From<RecurrentSinCos<T>> for SinCosIterator<'static, T> {
    fn from(sin_cos: RecurrentSinCos<T>) -> Self {
        Self::LinearGrid(sin_cos)
    }
}

impl<'a, T: Float> From<ValueSinCosIterator<'a, T>> for SinCosIterator<'a, T> {
    fn from(sin_cos: ValueSinCosIterator<'a, T>) -> Self {
        Self::Values(sin_cos)
    }
}

/// Iterator over sin(start + k*step), cos(start + k*step) pairs
///
/// It yields sin(start), cos(start); and then sin/cos for angles start+step, start+2*step, ...
pub struct RecurrentSinCos<T> {
    increment: (T, T),
    current: (T, T),
}

impl<T: Float> RecurrentSinCos<T> {
    /// Construct [RecurrentSinCos] from angle x
    pub fn new(first: T, step: T) -> Self {
        Self {
            increment: (T::sin(step), T::cos(step)),
            current: (T::sin(first), T::cos(first)),
        }
    }

    pub fn with_zero_first(step: T) -> Self {
        Self {
            increment: (T::sin(step), T::cos(step)),
            current: (T::zero(), T::one()),
        }
    }
}

impl<T: Float> Iterator for RecurrentSinCos<T> {
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        let previous = self.current;
        self.current = (
            self.increment.0 * self.current.1 + self.increment.1 * self.current.0,
            self.increment.1 * self.current.1 - self.increment.0 * self.current.0,
        );
        Some(previous)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn recurrent_sin_cos() {
        const N: usize = 100;
        let first = std::f32::consts::PI / 6.0;
        let step = std::f32::consts::PI / (N as f32);
        let (desired_sin, desired_cos): (Vec<_>, Vec<_>) = (0..N)
            .map(|i| f32::sin_cos(first + step * (i as f32)))
            .unzip();
        let (actual_sin, actual_cos): (Vec<_>, Vec<_>) =
            RecurrentSinCos::new(first, step).take(N).unzip();
        assert_relative_eq!(&actual_sin[..], &desired_sin[..], max_relative = 1e-4);
        assert_relative_eq!(&actual_cos[..], &desired_cos[..], max_relative = 1e-4);
    }

    #[test]
    fn recurrent_sin_cos_with_zero_first() {
        let x = 0.01;
        const N: usize = 1000;
        let (desired_sin, desired_cos): (Vec<_>, Vec<_>) =
            (0..N).map(|i| f64::sin_cos(x * (i as f64))).unzip();
        let (actual_sin, actual_cos): (Vec<_>, Vec<_>) =
            RecurrentSinCos::with_zero_first(x).take(N).unzip();
        assert_relative_eq!(&actual_sin[..], &desired_sin[..], max_relative = 1e-12);
        assert_relative_eq!(&actual_cos[..], &desired_cos[..], max_relative = 1e-12);
    }
}
