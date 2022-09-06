use crate::data::TimeSeries;
use crate::float_trait::Float;
use crate::multicolor::PassbandTrait;
use crate::{DataSample, PassbandSet};

use itertools::Either;
use itertools::EitherOrBoth;
use itertools::Itertools;
use std::collections::BTreeMap;

pub struct MultiColorTimeSeries<'a, P: PassbandTrait, T: Float> {
    mapping: BTreeMap<P, TimeSeries<'a, T>>,
    flat: Option<FlatMultiColorTimeSeries<'static, P, T>>,
}

impl<'a, 'p, P, T> MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait + 'p,
    T: Float,
{
    pub fn new(map: impl Into<BTreeMap<P, TimeSeries<'a, T>>>) -> Self {
        Self {
            mapping: map.into(),
            flat: None,
        }
    }

    pub fn flatten(&mut self) -> &FlatMultiColorTimeSeries<'static, P, T> {
        self.flat
            .get_or_insert_with(|| FlatMultiColorTimeSeries::from_mapping(&mut self.mapping))
    }

    pub fn passbands<'slf>(
        &'slf self,
    ) -> std::collections::btree_map::Keys<'slf, P, TimeSeries<'a, T>>
    where
        'a: 'slf,
    {
        self.mapping.keys()
    }

    pub fn iter_passband_set<'slf, 'ps>(
        &'slf self,
        passband_set: &'ps PassbandSet<P>,
    ) -> impl Iterator<Item = (&P, Option<&TimeSeries<'a, T>>)> + 'slf
    where
        'a: 'slf,
        'ps: 'a,
    {
        match passband_set {
            PassbandSet::AllAvailable => {
                Either::Left(self.mapping.iter().map(|(p, ts)| (p, Some(ts))))
            }
            PassbandSet::FixedSet(set) => Either::Right(self.iter_matched_passbands(set.iter())),
        }
    }

    pub fn iter_passband_set_mut<'slf, 'ps>(
        &'slf mut self,
        passband_set: &'ps PassbandSet<P>,
    ) -> impl Iterator<Item = (&P, Option<&mut TimeSeries<'a, T>>)> + 'slf
    where
        'a: 'slf,
        'ps: 'a,
    {
        match passband_set {
            PassbandSet::AllAvailable => {
                Either::Left(self.mapping.iter_mut().map(|(p, ts)| (p, Some(ts))))
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
        passband_it.map(|p| (p, self.mapping.get(p)))
    }

    pub fn iter_matched_passbands_mut(
        &mut self,
        passband_it: impl Iterator<Item = &'p P>,
    ) -> impl Iterator<Item = (&'p P, Option<&mut TimeSeries<'a, T>>)> {
        passband_it
            .merge_join_by(self.mapping.iter_mut(), |p1, (p2, _ts)| p1.cmp(p2))
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

impl<'a, P: PassbandTrait, T: Float> FromIterator<(P, TimeSeries<'a, T>)>
    for MultiColorTimeSeries<'a, P, T>
{
    fn from_iter<I: IntoIterator<Item = (P, TimeSeries<'a, T>)>>(iter: I) -> Self {
        Self {
            mapping: iter.into_iter().collect(),
            flat: None,
        }
    }
}

pub struct FlatMultiColorTimeSeries<'a, P: PassbandTrait, T: Float> {
    pub t: DataSample<'a, T>,
    pub m: DataSample<'a, T>,
    pub w: DataSample<'a, T>,
    pub passbands: Vec<P>,
}

impl<P, T> FlatMultiColorTimeSeries<'static, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    pub fn from_mapping(mapping: &mut BTreeMap<P, TimeSeries<T>>) -> Self {
        let (t, m, w, passbands): (Vec<_>, Vec<_>, Vec<_>, _) = mapping
            .iter_mut()
            .map(|(p, ts)| {
                itertools::multizip((
                    ts.t.as_slice().iter().copied(),
                    ts.m.as_slice().iter().copied(),
                    ts.w.as_slice().iter().copied(),
                    std::iter::repeat(p.clone()),
                ))
            })
            .kmerge_by(|(t1, _, _, _), (t2, _, _, _)| t1 <= t2)
            .multiunzip();

        Self {
            t: t.into(),
            m: m.into(),
            w: w.into(),
            passbands,
        }
    }
}
