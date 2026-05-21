use crate::data::TimeSeries;
use crate::float_trait::Float;
use crate::multicolor::PassbandTrait;
use crate::{DataSample, PassbandSet};

use conv::prelude::*;
use itertools::EitherOrBoth;
use itertools::Itertools;
use std::collections::{BTreeMap, BTreeSet};
use std::ops::{Deref, DerefMut};

// Inner enum holds the actual variant data; all passband references borrow from
// the outer `passband_vec` owned by `MultiColorTimeSeries`.
enum MultiColorTimeSeriesInner<'pb, 'a, P: PassbandTrait, T: Float> {
    Mapping(MappedMultiColorTimeSeries<'pb, 'a, P, T>),
    Flat(FlatMultiColorTimeSeries<'pb, 'a, P, T>),
    MappingFlat {
        mapping: MappedMultiColorTimeSeries<'pb, 'a, P, T>,
        flat: FlatMultiColorTimeSeries<'pb, 'a, P, T>,
    },
}

impl<'pb, 'a, P: PassbandTrait, T: Float> std::fmt::Debug
    for MultiColorTimeSeriesInner<'pb, 'a, P, T>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mapping(m) => f.debug_tuple("Mapping").field(m).finish(),
            Self::Flat(fl) => f.debug_tuple("Flat").field(fl).finish(),
            Self::MappingFlat { mapping, flat } => f
                .debug_struct("MappingFlat")
                .field("mapping", mapping)
                .field("flat", flat)
                .finish(),
        }
    }
}

#[ouroboros::self_referencing]
#[derive(Debug)]
pub struct MultiColorTimeSeries<'a, P: PassbandTrait + 'a, T: Float> {
    passband_vec: Vec<P>,
    #[borrows(passband_vec)]
    #[covariant]
    inner: MultiColorTimeSeriesInner<'this, 'a, P, T>,
}

impl<'a, P, T> Clone for MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn clone(&self) -> Self {
        // Build owned BTreeMap from current state, then reconstruct.
        let owned: BTreeMap<P, TimeSeries<'a, T>> = self.with_inner(|inner| match inner {
            MultiColorTimeSeriesInner::Mapping(m)
            | MultiColorTimeSeriesInner::MappingFlat { mapping: m, .. } => {
                m.0.iter()
                    .map(|(p, ts)| ((*p).clone(), ts.clone()))
                    .collect()
            }
            MultiColorTimeSeriesInner::Flat(flat) => {
                let mut map = BTreeMap::new();
                for (&t, &m, &w, p) in itertools::multizip((
                    flat.t.sample.iter(),
                    flat.m.sample.iter(),
                    flat.w.sample.iter(),
                    flat.passbands.iter().copied(),
                )) {
                    let entry = map
                        .entry((*p).clone())
                        .or_insert_with(|| (vec![], vec![], vec![]));
                    entry.0.push(t);
                    entry.1.push(m);
                    entry.2.push(w);
                }
                map.into_iter()
                    .map(|(p, (t, m, w))| (p, TimeSeries::new(t, m, w)))
                    .collect()
            }
        });
        Self::from_map(owned)
    }
}

impl<'a, P, T> MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    pub fn total_lenu(&self) -> usize {
        self.with_inner(|inner| match inner {
            MultiColorTimeSeriesInner::Mapping(m) => m.total_lenu(),
            MultiColorTimeSeriesInner::Flat(f) => f.total_lenu(),
            MultiColorTimeSeriesInner::MappingFlat { flat, .. } => flat.total_lenu(),
        })
    }

    pub fn total_lenf(&self) -> T {
        self.with_inner(|inner| match inner {
            MultiColorTimeSeriesInner::Mapping(m) => m.total_lenf(),
            MultiColorTimeSeriesInner::Flat(f) => f.total_lenf(),
            MultiColorTimeSeriesInner::MappingFlat { flat, .. } => flat.total_lenf(),
        })
    }

    pub fn passband_count(&self) -> usize {
        self.borrow_passband_vec().len()
    }

    /// Build from an owned BTreeMap. Clones K passband keys into `passband_vec`.
    pub fn from_map(map: impl Into<BTreeMap<P, TimeSeries<'a, T>>>) -> Self {
        let map: BTreeMap<P, TimeSeries<'a, T>> = map.into();
        let passband_vec: Vec<P> = map.keys().cloned().collect();
        MultiColorTimeSeriesBuilder {
            passband_vec,
            inner_builder: |vec: &Vec<P>| {
                let mapping = MappedMultiColorTimeSeries(
                    map.into_iter()
                        .map(|(p, ts)| {
                            let idx = vec.binary_search(&p).expect("key must be in passband_vec");
                            (&vec[idx], ts)
                        })
                        .collect(),
                );
                MultiColorTimeSeriesInner::Mapping(mapping)
            },
        }
        .build()
    }

    /// Build from flat arrays, deduplicating passbands automatically.
    pub fn from_flat(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
        w: impl Into<DataSample<'a, T>>,
        passbands: impl AsRef<[P]>,
    ) -> Self {
        let passbands = passbands.as_ref();
        // Collect references first, then clone only the K unique ones
        let uniq_passbands: Vec<P> = passbands
            .iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .cloned()
            .collect();
        Self::from_flat_with_passband_vec(t, m, w, passbands, uniq_passbands)
    }

    /// Build from flat arrays with a pre-built sorted unique passband vec.
    /// Each item in `passbands` is matched into `uniq_passbands` by binary search
    /// with no extra clones for the N-length array.
    pub fn from_flat_with_passband_vec(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
        w: impl Into<DataSample<'a, T>>,
        passbands: impl AsRef<[P]>,
        uniq_passbands: Vec<P>,
    ) -> Self {
        let t = t.into();
        let m = m.into();
        let w = w.into();
        let passbands = passbands.as_ref();
        let indices: Vec<usize> = passbands
            .iter()
            .map(|p| {
                uniq_passbands
                    .binary_search(p)
                    .expect("passband must be present in uniq_passbands")
            })
            .collect();
        let passband_vec = uniq_passbands;

        assert_eq!(
            t.sample.len(),
            m.sample.len(),
            "t and m should have the same size"
        );
        assert_eq!(
            m.sample.len(),
            w.sample.len(),
            "m and w should have the same size"
        );
        assert_eq!(
            t.sample.len(),
            indices.len(),
            "t and passbands should have the same size"
        );

        MultiColorTimeSeriesBuilder {
            passband_vec,
            inner_builder: |vec: &Vec<P>| {
                let passbands_refs: Vec<&P> = indices.iter().map(|&i| &vec[i]).collect();
                MultiColorTimeSeriesInner::Flat(FlatMultiColorTimeSeries {
                    t,
                    m,
                    w,
                    passbands: passbands_refs,
                })
            },
        }
        .build()
    }

    fn ensure_mapping(&mut self) {
        let needs_mapping =
            self.with_inner(|inner| matches!(inner, MultiColorTimeSeriesInner::Flat(_)));
        if needs_mapping {
            self.with_inner_mut(|inner| {
                take_mut::take(inner, |inner| match inner {
                    MultiColorTimeSeriesInner::Flat(mut flat) => {
                        let mapping = MappedMultiColorTimeSeries::from_flat(&mut flat);
                        MultiColorTimeSeriesInner::MappingFlat { mapping, flat }
                    }
                    _ => unreachable!("checked above"),
                });
            });
        }
    }

    /// Run a closure with mutable access to the mapping representation.
    /// Converts to `MappingFlat` or `Mapping` if currently `Flat`.
    pub fn with_mapping_mut<R>(
        &mut self,
        f: impl FnOnce(&mut MappedMultiColorTimeSeries<'_, 'a, P, T>) -> R,
    ) -> R {
        self.ensure_mapping();
        self.with_inner_mut(|inner| match inner {
            MultiColorTimeSeriesInner::Mapping(m) => f(m),
            MultiColorTimeSeriesInner::MappingFlat { mapping, .. } => f(mapping),
            MultiColorTimeSeriesInner::Flat(_) => unreachable!("ensure_mapping was called"),
        })
    }

    pub fn mapping(&self) -> Option<&MappedMultiColorTimeSeries<'_, 'a, P, T>> {
        match self.borrow_inner() {
            MultiColorTimeSeriesInner::Mapping(m) => Some(m),
            MultiColorTimeSeriesInner::MappingFlat { mapping, .. } => Some(mapping),
            MultiColorTimeSeriesInner::Flat(_) => None,
        }
    }

    fn ensure_flat(&mut self) {
        let needs_flat =
            self.with_inner(|inner| matches!(inner, MultiColorTimeSeriesInner::Mapping(_)));
        if needs_flat {
            self.with_inner_mut(|inner| {
                take_mut::take(inner, |inner| match inner {
                    MultiColorTimeSeriesInner::Mapping(mut mapping) => {
                        let flat = FlatMultiColorTimeSeries::from_mapping(&mut mapping);
                        MultiColorTimeSeriesInner::MappingFlat { mapping, flat }
                    }
                    _ => unreachable!("checked above"),
                });
            });
        }
    }

    pub fn with_flat_mut<R>(
        &mut self,
        f: impl FnOnce(&mut FlatMultiColorTimeSeries<'_, 'a, P, T>) -> R,
    ) -> R {
        self.ensure_flat();
        self.with_inner_mut(|inner| match inner {
            MultiColorTimeSeriesInner::Flat(fl) => f(fl),
            MultiColorTimeSeriesInner::MappingFlat { flat, .. } => f(flat),
            MultiColorTimeSeriesInner::Mapping(_) => unreachable!("ensure_flat was called"),
        })
    }

    pub fn flat(&self) -> Option<&FlatMultiColorTimeSeries<'_, 'a, P, T>> {
        match self.borrow_inner() {
            MultiColorTimeSeriesInner::Flat(f) => Some(f),
            MultiColorTimeSeriesInner::MappingFlat { flat, .. } => Some(flat),
            MultiColorTimeSeriesInner::Mapping(_) => None,
        }
    }

    pub fn passbands(&self) -> impl Iterator<Item = &P> {
        self.borrow_passband_vec().iter()
    }

    /// Inserts new pair of passband and time series into the multicolor time series.
    ///
    /// Always converts to `Mapping`. For an existing passband, does an in-place
    /// insert (O(log K)). For a new passband, rebuilds the entire struct.
    pub fn insert(&mut self, passband: P, ts: TimeSeries<'a, T>) -> Option<TimeSeries<'a, T>> {
        // Fast path: passband already in passband_vec — no rebuild needed.
        if self.borrow_passband_vec().binary_search(&passband).is_ok() {
            let mut old = None;
            self.ensure_mapping();
            self.with_inner_mut(|inner| {
                let mapping = match inner {
                    MultiColorTimeSeriesInner::Mapping(m) => m,
                    MultiColorTimeSeriesInner::MappingFlat { mapping, .. } => mapping,
                    _ => unreachable!(),
                };
                // BTreeMap<&'pb P, V> can look up by &P because &P: Borrow<P> and
                // &'pb P: Borrow<P>, so get_mut(&passband as &P) resolves correctly.
                let value = mapping
                    .0
                    .get_mut(&passband as &P)
                    .expect("passband was found in passband_vec, so must be in mapping");
                old = Some(std::mem::replace(value, ts));
            });
            return old;
        }

        // Slow path: new passband — rebuild the struct. Move TimeSeries values where
        // possible (Mapping/MappingFlat); for Flat, reconstruct from individual elements.
        let mut old_ts = None;
        take_mut::take(self, |mut slf| {
            let mut owned_map: BTreeMap<P, TimeSeries<'a, T>> = BTreeMap::new();
            slf.with_inner_mut(|inner| {
                take_mut::take(inner, |inner_val| {
                    match inner_val {
                        MultiColorTimeSeriesInner::Mapping(m)
                        | MultiColorTimeSeriesInner::MappingFlat { mapping: m, .. } => {
                            for (p, ts) in m.0 {
                                owned_map.insert((*p).clone(), ts);
                            }
                        }
                        MultiColorTimeSeriesInner::Flat(flat) => {
                            let mut raw = BTreeMap::new();
                            for (&t, &m, &w, p) in itertools::multizip((
                                flat.t.sample.iter(),
                                flat.m.sample.iter(),
                                flat.w.sample.iter(),
                                flat.passbands.iter().copied(),
                            )) {
                                let entry = raw
                                    .entry((*p).clone())
                                    .or_insert_with(|| (vec![], vec![], vec![]));
                                entry.0.push(t);
                                entry.1.push(m);
                                entry.2.push(w);
                            }
                            owned_map = raw
                                .into_iter()
                                .map(|(p, (t, m, w))| (p, TimeSeries::new(t, m, w)))
                                .collect();
                        }
                    }
                    MultiColorTimeSeriesInner::Mapping(MappedMultiColorTimeSeries(BTreeMap::new()))
                });
            });
            old_ts = owned_map.insert(passband, ts);
            Self::from_map(owned_map)
        });
        old_ts
    }
}

impl<'a, P, T> Default for MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn default() -> Self {
        MultiColorTimeSeriesBuilder {
            passband_vec: Vec::new(),
            inner_builder: |_| {
                MultiColorTimeSeriesInner::Mapping(MappedMultiColorTimeSeries(BTreeMap::new()))
            },
        }
        .build()
    }
}

#[derive(Debug)]
pub struct MappedMultiColorTimeSeries<'pb, 'a, P: PassbandTrait, T: Float>(
    BTreeMap<&'pb P, TimeSeries<'a, T>>,
);

impl<'pb, 'a, P, T> Clone for MappedMultiColorTimeSeries<'pb, 'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<'pb, 'a, P, T> PartialEq for MappedMultiColorTimeSeries<'pb, 'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|((k1, v1), (k2, v2))| k1 == k2 && v1 == v2)
    }
}

impl<'pb, 'a, P, T> MappedMultiColorTimeSeries<'pb, 'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    pub fn from_flat(flat: &mut FlatMultiColorTimeSeries<'pb, 'a, P, T>) -> Self {
        let mut map = BTreeMap::new();
        let groups = itertools::multizip((
            flat.t.as_slice().iter(),
            flat.m.as_slice().iter(),
            flat.w.as_slice().iter(),
            flat.passbands.iter().copied(),
        ))
        .chunk_by(|(_t, _m, _w, p)| *p);
        for (p, group) in &groups {
            let entry = map.entry(p).or_insert_with(|| (vec![], vec![], vec![]));
            for (&t, &m, &w, _p) in group {
                entry.0.push(t);
                entry.1.push(m);
                entry.2.push(w);
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

    pub fn passbands(&self) -> impl Iterator<Item = &P> {
        self.0.keys().copied()
    }

    pub fn iter_ts(&self) -> std::collections::btree_map::Values<'_, &'pb P, TimeSeries<'a, T>> {
        self.0.values()
    }

    pub fn iter_passband_set<'slf, 'ps>(
        &'slf self,
        passband_set: &'ps PassbandSet<P>,
    ) -> impl Iterator<Item = (&'ps P, Option<&'slf TimeSeries<'a, T>>)> + 'slf
    where
        'a: 'slf,
        'ps: 'slf,
    {
        let PassbandSet(set) = passband_set;
        self.iter_matched_passbands(set.iter())
    }

    pub fn iter_passband_set_mut<'slf, 'ps>(
        &'slf mut self,
        passband_set: &'ps PassbandSet<P>,
    ) -> impl Iterator<Item = (&'ps P, Option<&'slf mut TimeSeries<'a, T>>)> + 'slf
    where
        'a: 'slf,
        'ps: 'slf,
    {
        let PassbandSet(set) = passband_set;
        self.iter_matched_passbands_mut(set.iter())
    }

    pub fn iter_matched_passbands<'slf, 'ps>(
        &'slf self,
        passband_it: impl Iterator<Item = &'ps P>,
    ) -> impl Iterator<Item = (&'ps P, Option<&'slf TimeSeries<'a, T>>)>
    where
        'ps: 'slf,
        P: 'ps,
    {
        passband_it.map(|p| (p, self.get(p)))
    }

    pub fn iter_matched_passbands_mut<'slf, 'ps>(
        &'slf mut self,
        passband_it: impl Iterator<Item = &'ps P>,
    ) -> impl Iterator<Item = (&'ps P, Option<&'slf mut TimeSeries<'a, T>>)>
    where
        'ps: 'slf,
        P: 'ps,
    {
        passband_it
            .merge_join_by(self.iter_mut(), |p1, (p2, _ts)| p1.cmp(*p2))
            .filter_map(|either_or_both| match either_or_both {
                EitherOrBoth::Left(p) => Some((p, None)),
                EitherOrBoth::Right(_) => None,
                EitherOrBoth::Both(p, (_, ts)) => Some((p, Some(ts))),
            })
    }
}

impl<'pb, 'a, P: PassbandTrait, T: Float> Deref for MappedMultiColorTimeSeries<'pb, 'a, P, T> {
    type Target = BTreeMap<&'pb P, TimeSeries<'a, T>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'pb, 'a, P: PassbandTrait, T: Float> DerefMut for MappedMultiColorTimeSeries<'pb, 'a, P, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[derive(Debug)]
pub struct FlatMultiColorTimeSeries<'pb, 'a, P: PassbandTrait, T: Float> {
    pub t: DataSample<'a, T>,
    pub m: DataSample<'a, T>,
    pub w: DataSample<'a, T>,
    pub passbands: Vec<&'pb P>,
}

impl<'pb, 'a, P, T> Clone for FlatMultiColorTimeSeries<'pb, 'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn clone(&self) -> Self {
        Self {
            t: self.t.clone(),
            m: self.m.clone(),
            w: self.w.clone(),
            passbands: self.passbands.clone(),
        }
    }
}

impl<'pb, 'a, P, T> PartialEq for FlatMultiColorTimeSeries<'pb, 'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn eq(&self, other: &Self) -> bool {
        self.t == other.t
            && self.m == other.m
            && self.w == other.w
            && self
                .passbands
                .iter()
                .zip(other.passbands.iter())
                .all(|(a, b)| *a == *b)
            && self.passbands.len() == other.passbands.len()
    }
}

impl<'pb, 'a, P, T> FlatMultiColorTimeSeries<'pb, 'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    pub fn from_mapping(mapping: &mut MappedMultiColorTimeSeries<'pb, 'a, P, T>) -> Self {
        let (t, m, w, passbands): (Vec<_>, Vec<_>, Vec<_>, Vec<_>) = mapping
            .iter_mut()
            .map(|(&p, ts)| {
                itertools::multizip((
                    ts.t.as_slice().iter().copied(),
                    ts.m.as_slice().iter().copied(),
                    ts.w.as_slice().iter().copied(),
                    std::iter::repeat(p),
                ))
            })
            .kmerge_by(|(t1, _m1, _w1, _p1), (t2, _m2, _w2, _p2)| t1 <= t2)
            .multiunzip();

        Self {
            t: t.into(),
            m: m.into(),
            w: w.into(),
            passbands,
        }
    }

    pub fn total_lenu(&self) -> usize {
        self.t.sample.len()
    }

    pub fn total_lenf(&self) -> T {
        self.t.sample.len().value_as::<T>().unwrap()
    }
}

impl<'pb, 'a, P, T> From<FlatMultiColorTimeSeries<'pb, 'a, P, T>>
    for MappedMultiColorTimeSeries<'pb, 'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(mut flat: FlatMultiColorTimeSeries<'pb, 'a, P, T>) -> Self {
        Self::from_flat(&mut flat)
    }
}

impl<'a, P, T> From<BTreeMap<P, TimeSeries<'a, T>>> for MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn from(map: BTreeMap<P, TimeSeries<'a, T>>) -> Self {
        Self::from_map(map)
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
        // Collect (passband name, ts length) pairs from the mapping representation.
        let mapped_info: Vec<(String, usize)> = {
            let mut m = mcts.clone();
            let mut info = vec![];
            m.with_mapping_mut(|mapping| {
                for (p, ts) in mapping.iter() {
                    info.push((p.name().to_string(), ts.lenu()));
                }
            });
            info
        };
        // Collect the same from a freshly built mapping derived from the flat representation.
        let mapped_from_flat_info: Vec<(String, usize)> = {
            let mut m = mcts.clone();
            m.ensure_flat();
            let mut info = vec![];
            m.with_inner_mut(|inner| {
                if let MultiColorTimeSeriesInner::MappingFlat { flat, .. }
                | MultiColorTimeSeriesInner::Flat(flat) = inner
                {
                    let rebuilt = MappedMultiColorTimeSeries::from_flat(flat);
                    for (p, ts) in rebuilt.iter() {
                        info.push((p.name().to_string(), ts.lenu()));
                    }
                }
            });
            info
        };
        assert_eq!(mapped_info, mapped_from_flat_info);
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

    #[test]
    fn from_flat_with_passband_vec() {
        let passband_g = MonochromePassband::new(4700.0_f64, "g");
        let passband_r = MonochromePassband::new(6200.0_f64, "r");
        let passband_vec = vec![passband_g.clone(), passband_r.clone()];
        let passbands = vec![
            passband_g.clone(),
            passband_g.clone(),
            passband_r.clone(),
            passband_r.clone(),
        ];
        let t = vec![0.0_f64, 1.0, 0.0, 1.0];
        let m = vec![1.0_f64, 2.0, 3.0, 4.0];
        let w = vec![1.0_f64; 4];
        let mcts =
            MultiColorTimeSeries::from_flat_with_passband_vec(t, m, w, passbands, passband_vec);
        assert_eq!(mcts.passband_count(), 2);
        assert_eq!(mcts.total_lenu(), 4);
    }
}
