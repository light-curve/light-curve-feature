use crate::data::TimeSeries;
use crate::float_trait::Float;
use crate::multicolor::PassbandTrait;
use crate::PassbandSet;

use itertools::Either;
use itertools::EitherOrBoth;
use itertools::Itertools;
use std::collections::BTreeMap;
use std::ops::{Deref, DerefMut};

pub struct MultiColorTimeSeries<'a, P: PassbandTrait, T: Float>(BTreeMap<P, TimeSeries<'a, T>>);

impl<'a, 'p, P, T> MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait + 'p,
    T: Float,
{
    pub fn new(map: impl Into<BTreeMap<P, TimeSeries<'a, T>>>) -> Self {
        Self(map.into())
    }

    pub fn iter_passband_set<'slf, 'ps>(
        &'slf self,
        passband_set: &'ps PassbandSet<P>,
    ) -> impl Iterator<Item = (&P, Option<&TimeSeries<'a, T>>)> + 'ps
    where
        'a: 'ps,
        'slf: 'ps,
        'ps: 'slf,
    {
        match passband_set {
            PassbandSet::AllAvailable => Either::Left(self.0.iter().map(|(p, ts)| (p, Some(ts)))),
            PassbandSet::FixedSet(set) => Either::Right(self.iter_matched_passbands(set.iter())),
        }
    }

    pub fn iter_passband_set_mut<'slf, 'ps>(
        &'slf mut self,
        passband_set: &'ps PassbandSet<P>,
    ) -> impl Iterator<Item = (&P, Option<&mut TimeSeries<'a, T>>)> + 'ps
    where
        'a: 'ps,
        'slf: 'ps,
        'ps: 'slf,
    {
        match passband_set {
            PassbandSet::AllAvailable => {
                Either::Left(self.0.iter_mut().map(|(p, ts)| (p, Some(ts))))
            }
            PassbandSet::FixedSet(set) => {
                Either::Right(self.iter_matched_passbands_mut(set.iter()))
            }
        }
    }

    pub fn iter_matched_passbands(
        &self,
        passband_it: impl Iterator<Item = &'p P>,
    ) -> impl Iterator<Item = (&'p P, Option<&TimeSeries<'a, T>>)> {
        passband_it.map(|p| (p, self.0.get(p)))
    }

    pub fn iter_matched_passbands_mut(
        &mut self,
        passband_it: impl Iterator<Item = &'p P>,
    ) -> impl Iterator<Item = (&'p P, Option<&mut TimeSeries<'a, T>>)> {
        passband_it
            .merge_join_by(self.0.iter_mut(), |p1, (p2, _ts)| p1.cmp(p2))
            .filter_map(|either_or_both| match either_or_both {
                // mcts misses required passband
                EitherOrBoth::Left(p) => Some((p, None)),
                // mcts has some passban passband_set doesn't require
                EitherOrBoth::Right(_) => None,
                // passbands match
                EitherOrBoth::Both(p, (_, ts)) => Some((p, Some(ts))),
            })
    }
}

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
