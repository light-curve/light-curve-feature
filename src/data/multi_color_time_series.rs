use crate::data::TimeSeries;
use crate::float_trait::Float;
use crate::multicolor::PassbandTrait;
use crate::{DataSample, PassbandSet};

use conv::prelude::*;
use itertools::Either;
use itertools::EitherOrBoth;
use itertools::Itertools;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Deref, DerefMut};

#[derive(Clone, Debug)]
pub enum MultiColorTimeSeries<'a, P: PassbandTrait, T: Float> {
    Mapping(MappedMultiColorTimeSeries<'a, P, T>),
    Flat(FlatMultiColorTimeSeries<'a, P, T>),
    MappingFlat {
        mapping: MappedMultiColorTimeSeries<'a, P, T>,
        flat: FlatMultiColorTimeSeries<'a, P, T>,
    },
}

impl<'a, 'p, P, T> MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait + 'p,
    T: Float,
{
    pub fn total_lenu(&self) -> usize {
        match self {
            Self::Mapping(mapping) => mapping.total_lenu(),
            Self::Flat(flat) => flat.total_lenu(),
            Self::MappingFlat { flat, .. } => flat.total_lenu(),
        }
    }

    pub fn total_lenf(&self) -> T {
        match self {
            Self::Mapping(mapping) => mapping.total_lenf(),
            Self::Flat(flat) => flat.total_lenf(),
            Self::MappingFlat { flat, .. } => flat.total_lenf(),
        }
    }

    pub fn passband_count(&self) -> usize {
        match self {
            Self::Mapping(mapping) => mapping.passband_count(),
            Self::Flat(flat) => flat.passband_count(),
            // Both flat and mapping have the same number of passbands and should be equally fast
            Self::MappingFlat { flat, .. } => flat.passband_count(),
        }
    }

    pub fn from_map(map: impl Into<BTreeMap<P, TimeSeries<'a, T>>>) -> Self {
        Self::Mapping(MappedMultiColorTimeSeries::new(map))
    }

    pub fn from_flat(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
        w: impl Into<DataSample<'a, T>>,
        passbands: impl Into<Vec<P>>,
    ) -> Self {
        Self::Flat(FlatMultiColorTimeSeries::new(t, m, w, passbands))
    }

    fn ensure_mapping(&mut self) -> &mut Self {
        if matches!(self, MultiColorTimeSeries::Flat(_)) {
            take_mut::take(self, |slf| match slf {
                Self::Flat(mut flat) => {
                    let mapping = MappedMultiColorTimeSeries::from_flat(&mut flat);
                    Self::MappingFlat { mapping, flat }
                }
                _ => unreachable!("We just checked that we are in ::Flat variant"),
            });
        }
        self
    }

    fn enforce_mapping(&mut self) -> &mut Self {
        match self {
            Self::Mapping(_) => {}
            Self::Flat(_flat) => take_mut::take(self, |slf| match slf {
                Self::Flat(flat) => Self::Mapping(flat.into()),
                _ => unreachable!("We just checked that we are in ::Flat variant"),
            }),
            Self::MappingFlat { .. } => {
                take_mut::take(self, |slf| match slf {
                    Self::MappingFlat { mapping, .. } => Self::Mapping(mapping),
                    _ => unreachable!("We just checked that we are in ::MappingFlat variant"),
                });
            }
        }
        self
    }

    pub fn mapping_mut(&mut self) -> &mut MappedMultiColorTimeSeries<'a, P, T> {
        match self.ensure_mapping() {
            Self::Mapping(mapping) => mapping,
            Self::Flat(_flat) => {
                unreachable!(
                    "::Flat variant is already transformed to ::MappingFlat in ensure_mapping"
                )
            }
            Self::MappingFlat { mapping, .. } => mapping,
        }
    }

    pub fn mapping(&self) -> Option<&MappedMultiColorTimeSeries<'a, P, T>> {
        match self {
            Self::Mapping(mapping) => Some(mapping),
            Self::Flat(_flat) => None,
            Self::MappingFlat { mapping, .. } => Some(mapping),
        }
    }

    fn ensure_flat(&mut self) -> &mut Self {
        if matches!(self, MultiColorTimeSeries::Mapping(_)) {
            take_mut::take(self, |slf| match slf {
                Self::Mapping(mut mapping) => {
                    let flat = FlatMultiColorTimeSeries::from_mapping(&mut mapping);
                    Self::MappingFlat { mapping, flat }
                }
                _ => unreachable!("We just checked that we are in ::Mapping variant"),
            });
        }
        self
    }

    pub fn flat_mut(&mut self) -> &mut FlatMultiColorTimeSeries<'a, P, T> {
        match self.ensure_flat() {
            Self::Mapping(_mapping) => {
                unreachable!(
                    "::Mapping variant is already transformed to ::MappingFlat in ensure_flat"
                )
            }
            Self::Flat(flat) => flat,
            Self::MappingFlat { flat, .. } => flat,
        }
    }

    pub fn flat(&self) -> Option<&FlatMultiColorTimeSeries<'a, P, T>> {
        match self {
            Self::Mapping(_mapping) => None,
            Self::Flat(flat) => Some(flat),
            Self::MappingFlat { flat, .. } => Some(flat),
        }
    }

    pub fn passbands<'slf>(
        &'slf self,
    ) -> Either<
        std::collections::btree_map::Keys<'slf, P, TimeSeries<'a, T>>,
        std::collections::btree_set::Iter<P>,
    >
    where
        'a: 'slf,
    {
        match self {
            Self::Mapping(mapping) => Either::Left(mapping.passbands()),
            Self::Flat(flat) => Either::Right(flat.passband_set.iter()),
            Self::MappingFlat { mapping, .. } => Either::Left(mapping.passbands()),
        }
    }

    /// Inserts new pair of passband and time series into the multicolor time series.
    ///
    /// It always converts [MultiColorTimeSeries] to [MultiColorTimeSeries::Mapping] variant.
    /// Also it replaces existing time series if passband is already present, and returns old time
    /// series.
    pub fn insert(&mut self, passband: P, ts: TimeSeries<'a, T>) -> Option<TimeSeries<'a, T>> {
        match self.enforce_mapping() {
            Self::Mapping(mapping) => mapping.0.insert(passband, ts),
            _ => unreachable!("We just converted self to ::Mapping variant"),
        }
    }
}

impl<'a, P, T> Default for MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn default() -> Self {
        Self::Mapping(MappedMultiColorTimeSeries::new(BTreeMap::new()))
    }
}

#[derive(Debug, Clone)]
pub struct MappedMultiColorTimeSeries<'a, P: PassbandTrait, T: Float>(
    BTreeMap<P, TimeSeries<'a, T>>,
);

impl<'a, P, T> PartialEq for MappedMultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<'a, 'p, P, T> MappedMultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait + 'p,
    T: Float,
{
    pub fn new(map: impl Into<BTreeMap<P, TimeSeries<'a, T>>>) -> Self {
        Self(map.into())
    }

    pub fn from_flat(flat: &mut FlatMultiColorTimeSeries<P, T>) -> Self {
        let mut map = BTreeMap::new();
        let groups = itertools::multizip((
            flat.t.as_slice().iter(),
            flat.m.as_slice().iter(),
            flat.w.as_slice().iter(),
            flat.passbands.iter(),
        ))
        .group_by(|(_t, _m, _w, p)| (*p).clone());
        for (p, group) in &groups {
            let (t_vec, m_vec, w_vec) = map
                .entry(p.clone())
                .or_insert_with(|| (vec![], vec![], vec![]));
            for (&t, &m, &w, _p) in group {
                t_vec.push(t);
                m_vec.push(m);
                w_vec.push(w);
            }
        }
        Self(
            map.into_iter()
                .map(|(p, (t, m, w))| (p, TimeSeries::new(t, m, w)))
                .collect(),
        )
    }

    pub fn total_lenu(&self) -> usize {
        self.0.values().map(|ts| ts.lenu()).sum()
    }

    pub fn total_lenf(&self) -> T {
        self.total_lenu().value_as::<T>().unwrap()
    }

    pub fn passband_count(&self) -> usize {
        self.0.len()
    }

    pub fn passbands<'slf>(
        &'slf self,
    ) -> std::collections::btree_map::Keys<'slf, P, TimeSeries<'a, T>>
    where
        'a: 'slf,
    {
        self.0.keys()
    }

    pub fn iter_ts<'slf>(
        &'slf self,
    ) -> std::collections::btree_map::Values<'slf, P, TimeSeries<'a, T>>
    where
        'a: 'slf,
    {
        self.0.values()
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
            PassbandSet::AllAvailable => Either::Left(self.iter().map(|(p, ts)| (p, Some(ts)))),
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
            PassbandSet::AllAvailable => Either::Left(self.iter_mut().map(|(p, ts)| (p, Some(ts)))),
            PassbandSet::FixedSet(set) => {
                Either::Right(self.iter_matched_passbands_mut(set.iter()))
            }
        }
    }

    pub fn iter_matched_passbands(
        &self,
        passband_it: impl Iterator<Item = &'p P>,
    ) -> impl Iterator<Item = (&'p P, Option<&TimeSeries<'a, T>>)> {
        passband_it.map(|p| (p, self.get(p)))
    }

    pub fn iter_matched_passbands_mut(
        &mut self,
        passband_it: impl Iterator<Item = &'p P>,
    ) -> impl Iterator<Item = (&'p P, Option<&mut TimeSeries<'a, T>>)> {
        passband_it
            .merge_join_by(self.iter_mut(), |p1, (p2, _ts)| p1.cmp(p2))
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
    for MappedMultiColorTimeSeries<'a, P, T>
{
    fn from_iter<I: IntoIterator<Item = (P, TimeSeries<'a, T>)>>(iter: I) -> Self {
        Self(iter.into_iter().collect())
    }
}

impl<'a, P: PassbandTrait, T: Float> Deref for MappedMultiColorTimeSeries<'a, P, T> {
    type Target = BTreeMap<P, TimeSeries<'a, T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, P: PassbandTrait, T: Float> DerefMut for MappedMultiColorTimeSeries<'a, P, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug, Clone)]
pub struct FlatMultiColorTimeSeries<'a, P: PassbandTrait, T: Float> {
    pub t: DataSample<'a, T>,
    pub m: DataSample<'a, T>,
    pub w: DataSample<'a, T>,
    pub passbands: Vec<P>,
    passband_set: BTreeSet<P>,
}

impl<'a, P, T> PartialEq for FlatMultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.t == other.t
            && self.m == other.m
            && self.w == other.w
            && self.passbands == other.passbands
    }
}

impl<'a, P, T> FlatMultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    pub fn new(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
        w: impl Into<DataSample<'a, T>>,
        passbands: impl Into<Vec<P>>,
    ) -> Self {
        let t = t.into();
        let m = m.into();
        let w = w.into();
        let passbands = passbands.into();
        let passband_set = passbands.iter().cloned().collect();

        assert_eq!(
            t.sample.len(),
            m.sample.len(),
            "t and m should have the same size"
        );
        assert_eq!(
            m.sample.len(),
            w.sample.len(),
            "m and err should have the same size"
        );
        assert_eq!(
            t.sample.len(),
            passbands.len(),
            "t and passbands should have the same size"
        );

        Self {
            t,
            m,
            w,
            passbands,
            passband_set,
        }
    }

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
            .kmerge_by(|(t1, _m1, _w1, _p1), (t2, _m2, _w2, _p2)| t1 <= t2)
            .multiunzip();

        Self {
            t: t.into(),
            m: m.into(),
            w: w.into(),
            passbands,
            passband_set: mapping.keys().cloned().collect(),
        }
    }

    pub fn total_lenu(&self) -> usize {
        self.t.sample.len()
    }

    pub fn total_lenf(&self) -> T {
        self.t.sample.len().value_as::<T>().unwrap()
    }

    pub fn passband_count(&self) -> usize {
        self.passband_set.len()
    }
}

impl<'a, P, T> From<FlatMultiColorTimeSeries<'a, P, T>> for MappedMultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(mut flat: FlatMultiColorTimeSeries<'a, P, T>) -> Self {
        Self::from_flat(&mut flat)
    }
}

impl<'a, P, T> From<MappedMultiColorTimeSeries<'a, P, T>> for FlatMultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(mut mapped: MappedMultiColorTimeSeries<'a, P, T>) -> Self {
        Self::from_mapping(&mut mapped.0)
    }
}

impl<'a, P, T> From<FlatMultiColorTimeSeries<'a, P, T>> for MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(flat: FlatMultiColorTimeSeries<'a, P, T>) -> Self {
        Self::Flat(flat)
    }
}

impl<'a, P, T> From<MappedMultiColorTimeSeries<'a, P, T>> for MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(mapped: MappedMultiColorTimeSeries<'a, P, T>) -> Self {
        Self::Mapping(mapped)
    }
}

impl<'a, P, T> From<MultiColorTimeSeries<'a, P, T>> for FlatMultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(mcts: MultiColorTimeSeries<'a, P, T>) -> Self {
        match mcts {
            MultiColorTimeSeries::Flat(flat) => flat,
            MultiColorTimeSeries::Mapping(mapped) => mapped.into(),
            MultiColorTimeSeries::MappingFlat { flat, .. } => flat,
        }
    }
}

impl<'a, P, T> From<MultiColorTimeSeries<'a, P, T>> for MappedMultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(mcts: MultiColorTimeSeries<'a, P, T>) -> Self {
        match mcts {
            MultiColorTimeSeries::Flat(flat) => flat.into(),
            MultiColorTimeSeries::Mapping(mapping) => mapping,
            MultiColorTimeSeries::MappingFlat { mapping, .. } => mapping,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::MonochromePassband;

    use ndarray::Array1;

    #[test]
    fn multi_color_ts_insert() {
        let mut mcts = MultiColorTimeSeries::default();
        mcts.insert(
            MonochromePassband::new(4700.0, "g"),
            TimeSeries::new_without_weight(Array1::linspace(0.0, 1.0, 11), Array1::zeros(11)),
        );
        assert_eq!(mcts.passband_count(), 1);
        assert_eq!(mcts.total_lenu(), 11);
        mcts.insert(
            MonochromePassband::new(6200.0, "r"),
            TimeSeries::new_without_weight(Array1::linspace(0.0, 1.0, 6), Array1::zeros(6)),
        );
        assert_eq!(mcts.passband_count(), 2);
        assert_eq!(mcts.total_lenu(), 17);
    }

    fn compare_variants<P: PassbandTrait, T: Float>(mcts: MultiColorTimeSeries<P, T>) {
        let flat: FlatMultiColorTimeSeries<_, _> = mcts.clone().into();
        let mapped: MappedMultiColorTimeSeries<_, _> = mcts.clone().into();
        let mapped_from_flat: MappedMultiColorTimeSeries<_, _> = flat.clone().into();
        let flat_from_mapped: FlatMultiColorTimeSeries<_, _> = mapped.clone().into();
        assert_eq!(mapped, mapped_from_flat);
        assert_eq!(flat, flat_from_mapped);
    }

    #[test]
    fn convert_between_variants() {
        let mut mcts = MultiColorTimeSeries::default();
        compare_variants(mcts.clone());
        mcts.insert(
            MonochromePassband::new(4700.0, "g"),
            TimeSeries::new_without_weight(Array1::linspace(0.0, 1.0, 11), Array1::zeros(11)),
        );
        compare_variants(mcts.clone());
        mcts.insert(
            MonochromePassband::new(6200.0, "r"),
            TimeSeries::new_without_weight(Array1::linspace(0.0, 1.0, 6), Array1::zeros(6)),
        );
        compare_variants(mcts.clone());
    }
}
