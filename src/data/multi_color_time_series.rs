use crate::data::TimeSeries;
use crate::float_trait::Float;
use crate::multicolor::PassbandTrait;
use crate::{DataSample, PassbandSet};

use conv::prelude::*;
use itertools::EitherOrBoth;
use itertools::Itertools;
use std::borrow::Cow;
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

/// Multi-color time series holding per-band light curves.
///
/// Two construction paths are available:
///
/// * **Owned** (`from_map`, `from_flat`, `from_flat_with_passband_vec`): the unique
///   passband values are cloned into the struct and per-observation references borrow
///   from that internal copy.
///
/// * **Borrowed** (`from_flat_borrowed`): the unique passband slice is borrowed from
///   the caller with lifetime `'a` — the same lifetime as the data arrays.  Zero
///   passband clones, zero passband allocations.
#[ouroboros::self_referencing]
#[derive(Debug)]
pub struct MultiColorTimeSeries<'a, P: PassbandTrait + 'a, T: Float> {
    passband_vec: Cow<'a, [P]>,
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
        match self.borrow_inner() {
            MultiColorTimeSeriesInner::Mapping(m)
            | MultiColorTimeSeriesInner::MappingFlat { mapping: m, .. } => {
                let ts_vec: Vec<TimeSeries<'a, T>> = m.0.values().cloned().collect();
                MultiColorTimeSeriesBuilder {
                    passband_vec: self.borrow_passband_vec().clone(),
                    inner_builder: |vec: &Cow<'_, [P]>| {
                        MultiColorTimeSeriesInner::Mapping(MappedMultiColorTimeSeries(
                            vec.iter().zip(ts_vec).collect(),
                        ))
                    },
                }
                .build()
            }
            MultiColorTimeSeriesInner::Flat(flat) => {
                let t = flat.t.clone();
                let m = flat.m.clone();
                let w = flat.w.clone();
                // Clone passband values to re-look them up in the new passband_vec inside
                // the builder — needed because Cow::Owned clones to a new address.
                let obs_passbands: Vec<P> = flat.passbands.iter().map(|p| (*p).clone()).collect();
                MultiColorTimeSeriesBuilder {
                    passband_vec: self.borrow_passband_vec().clone(),
                    inner_builder: |vec: &Cow<'_, [P]>| {
                        let passbands: Vec<&P> = obs_passbands
                            .iter()
                            .map(|p| {
                                vec.iter()
                                    .find(|q| *q == p)
                                    .expect("passband must be present in passband_vec")
                            })
                            .collect();
                        MultiColorTimeSeriesInner::Flat(FlatMultiColorTimeSeries {
                            t,
                            m,
                            w,
                            passbands,
                        })
                    },
                }
                .build()
            }
        }
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
            passband_vec: Cow::Owned(passband_vec),
            inner_builder: |vec: &Cow<'_, [P]>| {
                let mapping =
                    MappedMultiColorTimeSeries(vec.iter().zip(map.into_values()).collect());
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
        passband: impl AsRef<[P]>,
    ) -> Self {
        let passband = passband.as_ref();
        let uniq_passbands: Vec<P> = passband
            .iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .cloned()
            .collect();
        Self::from_flat_with_passband_vec(t, m, w, passband, uniq_passbands)
    }

    /// Build from flat arrays with a pre-built unique passband vec.
    /// Each item in `passband` must appear in `uniq_passbands`.
    pub fn from_flat_with_passband_vec(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
        w: impl Into<DataSample<'a, T>>,
        passband: impl AsRef<[P]>,
        uniq_passbands: Vec<P>,
    ) -> Self {
        let t = t.into();
        let m = m.into();
        let w = w.into();
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
            passband.as_ref().len(),
            "t and passband should have the same size"
        );

        MultiColorTimeSeriesBuilder {
            passband_vec: Cow::Owned(uniq_passbands),
            inner_builder: move |vec: &Cow<'_, [P]>| {
                let passbands_refs: Vec<&P> = passband
                    .as_ref()
                    .iter()
                    .map(|p| {
                        vec.iter()
                            .find(|q| *q == p)
                            .expect("passband must be present in passband_vec")
                    })
                    .collect();
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

    /// Build from flat arrays with a fully borrowed passband slice — zero passband clones.
    ///
    /// * `uniq_passbands` — the K unique passbands, borrowed with `'a`.
    /// * `passband` — per-observation references into `uniq_passbands`, also `'a`.
    ///
    /// Every element of `passband` must be a reference that points into `uniq_passbands`.
    /// The references are used directly with no per-observation search.
    pub fn from_flat_borrowed(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
        w: impl Into<DataSample<'a, T>>,
        passband: Vec<&'a P>,
        uniq_passbands: &'a [P],
    ) -> Self {
        let t = t.into();
        let m = m.into();
        let w = w.into();
        MultiColorTimeSeriesBuilder {
            passband_vec: Cow::Borrowed(uniq_passbands),
            inner_builder: |_vec: &Cow<'_, [P]>| {
                MultiColorTimeSeriesInner::Flat(FlatMultiColorTimeSeries {
                    t,
                    m,
                    w,
                    passbands: passband,
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

    /// Inserts a new passband / time-series pair, converting to mapping form.
    pub fn insert(&mut self, passband: P, ts: TimeSeries<'a, T>) -> Option<TimeSeries<'a, T>> {
        let exists = self.borrow_passband_vec().iter().any(|p| p == &passband);
        if exists {
            self.ensure_mapping();
            let mut old = None;
            self.with_inner_mut(|inner| {
                let mapping = match inner {
                    MultiColorTimeSeriesInner::Mapping(m) => m,
                    MultiColorTimeSeriesInner::MappingFlat { mapping, .. } => mapping,
                    _ => unreachable!(),
                };
                let value = mapping
                    .0
                    .get_mut(&passband as &P)
                    .expect("passband was found in sorted_passbands, so must be in mapping");
                old = Some(std::mem::replace(value, ts));
            });
            return old;
        }

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
                            type TmwVecs<T> = (Vec<T>, Vec<T>, Vec<T>);
                            let mut raw: BTreeMap<P, TmwVecs<T>> = BTreeMap::new();
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
            passband_vec: Cow::Owned(Vec::new()),
            inner_builder: |_| {
                MultiColorTimeSeriesInner::Mapping(MappedMultiColorTimeSeries(BTreeMap::new()))
            },
        }
        .build()
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
        for (&t, &m, &w, p) in itertools::multizip((
            flat.t.as_slice().iter(),
            flat.m.as_slice().iter(),
            flat.w.as_slice().iter(),
            flat.passbands.iter().copied(),
        )) {
            let entry = map.entry(p).or_insert_with(|| (vec![], vec![], vec![]));
            entry.0.push(t);
            entry.1.push(m);
            entry.2.push(w);
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

#[cfg(test)]
mod tests {
    use super::*;

    use crate::MonochromePassband;
    use approx::assert_relative_eq;
    use ndarray::Array1;

    type P = MonochromePassband<'static, f64>;
    type Mcts = MultiColorTimeSeries<'static, P, f64>;

    fn g() -> P {
        MonochromePassband::new(4700.0_f64, "g")
    }
    fn r() -> P {
        MonochromePassband::new(6200.0_f64, "r")
    }
    fn i() -> P {
        MonochromePassband::new(7500.0_f64, "i")
    }

    // g=3 obs, r=2 obs, interleaved timestamps
    fn make_mapping_mcts() -> Mcts {
        let mut map = BTreeMap::new();
        map.insert(
            g(),
            TimeSeries::new_without_weight(
                Array1::from(vec![0.0, 1.0, 2.0]),
                Array1::from(vec![1.0, 2.0, 3.0]),
            ),
        );
        map.insert(
            r(),
            TimeSeries::new_without_weight(
                Array1::from(vec![0.5, 1.5]),
                Array1::from(vec![10.0, 20.0]),
            ),
        );
        MultiColorTimeSeries::from_map(map)
    }

    // same data, built from flat interleaved arrays
    fn make_flat_mcts() -> Mcts {
        let passbands = vec![g(), g(), r(), g(), r()];
        MultiColorTimeSeries::from_flat(
            vec![0.0_f64, 1.0, 0.5, 2.0, 1.5],
            vec![1.0_f64, 2.0, 10.0, 3.0, 20.0],
            vec![1.0_f64; 5],
            passbands,
        )
    }

    // Verify Mapping and Flat variants agree on per-band lengths after round-tripping.
    fn assert_variants_agree(mcts: MultiColorTimeSeries<P, f64>) {
        let from_mapping: Vec<(String, usize)> = {
            let mut m = mcts.clone();
            let mut info = vec![];
            m.with_mapping_mut(|mapping| {
                for (p, ts) in mapping.iter() {
                    info.push((p.name().to_string(), ts.lenu()));
                }
            });
            info
        };
        let from_flat: Vec<(String, usize)> = {
            let mut m = mcts.clone();
            m.ensure_flat();
            let mut info = vec![];
            m.with_flat_mut(|flat| {
                let rebuilt = MappedMultiColorTimeSeries::from_flat(flat);
                for (p, ts) in rebuilt.iter() {
                    info.push((p.name().to_string(), ts.lenu()));
                }
            });
            info
        };
        assert_eq!(from_mapping, from_flat);
    }

    #[test]
    fn default_is_empty() {
        let mcts: Mcts = MultiColorTimeSeries::default();
        assert_eq!(mcts.passband_count(), 0);
        assert_eq!(mcts.total_lenu(), 0);
        assert!(mcts.passbands().next().is_none());
    }

    // from_flat deduplicates passbands and assigns observations correctly
    #[test]
    fn from_flat_deduplicates_and_assigns() {
        let mut mcts = make_flat_mcts();
        assert_eq!(mcts.passband_count(), 2);
        assert_eq!(mcts.total_lenu(), 5);
        mcts.with_mapping_mut(|mapping| {
            assert_eq!(mapping.get(&g() as &P).map(|ts| ts.lenu()), Some(3));
            assert_eq!(mapping.get(&r() as &P).map(|ts| ts.lenu()), Some(2));
        });
    }

    // from_flat_with_passband_vec preserves user-supplied passband order (no sorting)
    #[test]
    fn from_flat_with_passband_vec_preserves_order() {
        // r before g — reverse of alphabetical order
        let uniq = vec![r(), g()];
        let passbands = vec![g(), g(), r(), g(), r()];
        let mut mcts = MultiColorTimeSeries::from_flat_with_passband_vec(
            vec![0.0_f64, 1.0, 0.5, 2.0, 1.5],
            vec![1.0_f64, 2.0, 10.0, 3.0, 20.0],
            vec![1.0_f64; 5],
            passbands,
            uniq,
        );
        // passband order must be [r, g] as supplied
        let names: Vec<&str> = mcts.passbands().map(|p| p.name()).collect();
        assert_eq!(names, vec!["r", "g"]);
        // observations must still reach the correct bands
        mcts.with_mapping_mut(|mapping| {
            assert_eq!(mapping.get(&g() as &P).map(|ts| ts.lenu()), Some(3));
            assert_eq!(mapping.get(&r() as &P).map(|ts| ts.lenu()), Some(2));
        });
    }

    // from_flat_borrowed uses references directly — stays Flat, converts correctly
    #[test]
    fn from_flat_borrowed_stays_flat_and_converts() {
        // r before g — user-supplied order, not alphabetical
        let uniq = vec![r(), g()];
        let passband: Vec<&P> = vec![&uniq[1], &uniq[1], &uniq[0], &uniq[1], &uniq[0]];
        let mut mcts = MultiColorTimeSeries::from_flat_borrowed(
            vec![0.0_f64, 1.0, 0.5, 2.0, 1.5],
            vec![1.0_f64, 2.0, 10.0, 3.0, 20.0],
            vec![1.0_f64; 5],
            passband,
            &uniq,
        );
        assert!(mcts.flat().is_some());
        assert!(mcts.mapping().is_none());
        let names: Vec<&str> = mcts.passbands().map(|p| p.name()).collect();
        assert_eq!(names, vec!["r", "g"]);
        mcts.with_mapping_mut(|mapping| {
            assert_eq!(mapping.get(&g() as &P).map(|ts| ts.lenu()), Some(3));
            assert_eq!(mapping.get(&r() as &P).map(|ts| ts.lenu()), Some(2));
        });
    }

    // Flat↔Mapping conversions and round-trips agree on per-band data
    #[test]
    fn variant_conversion_round_trip() {
        assert_variants_agree(make_mapping_mcts());
        assert_variants_agree(make_flat_mcts());

        // Mapping → Flat produces time-sorted timestamps
        let mut mcts = make_mapping_mcts();
        mcts.with_flat_mut(|flat| {
            let t: Vec<f64> = flat.t.as_slice().to_vec();
            assert!(
                t.windows(2).all(|w| w[0] <= w[1]),
                "flat timestamps not sorted"
            );
        });
        assert!(mcts.flat().is_some());
        assert!(mcts.mapping().is_some());

        // Flat → Mapping produces correct per-band sizes
        let mut mcts = make_flat_mcts();
        mcts.with_mapping_mut(|mapping| {
            assert_eq!(mapping.get(&g() as &P).map(|ts| ts.lenu()), Some(3));
            assert_eq!(mapping.get(&r() as &P).map(|ts| ts.lenu()), Some(2));
        });
        assert!(mcts.flat().is_some());
        assert!(mcts.mapping().is_some());
    }

    // clone preserves passband order (including non-sorted) and all data
    #[test]
    fn clone_preserves_order_and_data() {
        // sorted order via from_map
        let cloned = make_mapping_mcts().clone();
        assert_eq!(
            cloned.passbands().map(|p| p.name()).collect::<Vec<_>>(),
            vec!["g", "r"]
        );
        assert_eq!(cloned.total_lenu(), 5);

        // non-sorted order via from_flat_with_passband_vec — order is preserved in clone
        let uniq = vec![r(), g()];
        let mcts = MultiColorTimeSeries::from_flat_with_passband_vec(
            vec![0.0_f64, 1.0, 0.5, 2.0, 1.5],
            vec![1.0_f64; 5],
            vec![1.0_f64; 5],
            vec![g(), g(), r(), g(), r()],
            uniq,
        );
        let cloned = mcts.clone();
        assert_eq!(
            cloned.passbands().map(|p| p.name()).collect::<Vec<_>>(),
            vec!["r", "g"]
        );
        assert_eq!(cloned.total_lenu(), 5);

        // MappingFlat variant
        let mut mcts = make_flat_mcts();
        mcts.with_mapping_mut(|_| {});
        let cloned = mcts.clone();
        assert_eq!(cloned.passband_count(), 2);
        assert_eq!(cloned.total_lenu(), 5);
    }

    // insert: existing passband replaces ts; new passband extends; works on Flat variant too
    #[test]
    fn insert_variants() {
        // replace existing on Mapping
        let mut mcts = make_mapping_mcts();
        let old = mcts.insert(
            g(),
            TimeSeries::new_without_weight(
                Array1::from(vec![5.0_f64, 6.0]),
                Array1::from(vec![99.0_f64, 100.0]),
            ),
        );
        assert_eq!(old.unwrap().lenu(), 3);
        assert_eq!(mcts.passband_count(), 2);
        assert_eq!(mcts.total_lenu(), 4);

        // add new passband on Mapping — preserves existing order, appends new
        let mut mcts = make_mapping_mcts();
        assert!(
            mcts.insert(
                i(),
                TimeSeries::new_without_weight(
                    Array1::from(vec![0.0_f64]),
                    Array1::from(vec![1.0_f64])
                )
            )
            .is_none()
        );
        assert_eq!(mcts.passband_count(), 3);
        assert_eq!(mcts.total_lenu(), 6);

        // add new passband on Flat
        let mut mcts = make_flat_mcts();
        assert!(mcts.flat().is_some());
        mcts.insert(
            i(),
            TimeSeries::new_without_weight(
                Array1::from(vec![0.0_f64]),
                Array1::from(vec![42.0_f64]),
            ),
        );
        assert_eq!(mcts.passband_count(), 3);
        assert_eq!(mcts.total_lenu(), 6);
    }

    #[test]
    fn iter_matched_passbands_missing_band_is_none() {
        let mut mcts = make_mapping_mcts();
        let query = [g(), i()];
        mcts.with_mapping_mut(|mapping| {
            let results: Vec<_> = mapping.iter_matched_passbands(query.iter()).collect();
            assert_eq!(results.len(), 2);
            assert!(results[0].1.is_some(), "g-band should be present");
            assert!(results[1].1.is_none(), "i-band should be missing");
        });
    }

    #[test]
    fn total_lenu_and_lenf_agree() {
        let mapping_mcts = make_mapping_mcts();
        assert_eq!(mapping_mcts.total_lenu(), 5);
        assert_relative_eq!(mapping_mcts.total_lenf(), 5.0_f64, epsilon = 1e-10);
        let flat_mcts = make_flat_mcts();
        assert_eq!(flat_mcts.total_lenu(), 5);
        assert_relative_eq!(flat_mcts.total_lenf(), 5.0_f64, epsilon = 1e-10);
    }
}
