use crate::data::TimeSeries;
use crate::float_trait::Float;
use crate::periodogram::freq::FreqGrid;

use enum_dispatch::enum_dispatch;
use ndarray::Array1;
use std::fmt::Debug;

/// Periodogram execution algorithm
#[enum_dispatch]
pub trait PeriodogramPowerTrait<T>: Debug + Clone + Send
where
    T: Float,
{
    fn power(&self, freq: &FreqGrid<T>, ts: &mut TimeSeries<T>) -> Array1<T>;
}
