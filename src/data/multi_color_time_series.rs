use crate::data::TimeSeries;
use crate::float_trait::Float;
use crate::multicolor::PassbandTrait;

use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};

pub struct MultiColorTimeSeries<'a, P: PassbandTrait, T: Float>(BTreeMap<P, TimeSeries<'a, T>>);

impl<'a, P: PassbandTrait, T: Float> Deref for MultiColorTimeSeries<'a, P, T> {
    type Target = BTreeMap<P, TimeSeries<'a, T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, P: PassbandTrait, T: Float> DerefMut for MultiColorTimeSeries<'a, P, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<'a, P: PassbandTrait, T: Float> FromIterator<(P, TimeSeries<'a, T>)>
    for MultiColorTimeSeries<'a, P, T>
{
    fn from_iter<I: IntoIterator<Item = (P, TimeSeries<'a, T>)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}
