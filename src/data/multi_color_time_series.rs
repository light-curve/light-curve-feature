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
// the outer `passband_vec` owned by `OwnedMultiColorTimeSeries`.
enum OwnedMCTSInner<'pb, 'a, P: PassbandTrait, T: Float> {
    Mapping(MappedMultiColorTimeSeries<'pb, 'a, P, T>),
    Flat(FlatMultiColorTimeSeries<'pb, 'a, P, T>),
    MappingFlat {
        mapping: MappedMultiColorTimeSeries<'pb, 'a, P, T>,
        flat: FlatMultiColorTimeSeries<'pb, 'a, P, T>,
    },
}

impl<'pb, 'a, P: PassbandTrait, T: Float> std::fmt::Debug for OwnedMCTSInner<'pb, 'a, P, T> {
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

// The ouroboros-based owned struct: owns passband_vec, inner borrows from it.
#[ouroboros::self_referencing]
#[derive(Debug)]
struct OwnedMultiColorTimeSeries<'a, P: PassbandTrait + 'a, T: Float> {
    passband_vec: Vec<P>,
    #[borrows(passband_vec)]
    #[covariant]
    inner: OwnedMCTSInner<'this, 'a, P, T>,
}

impl<'a, P, T> Clone for OwnedMultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn clone(&self) -> Self {
        match self.borrow_inner() {
            OwnedMCTSInner::Mapping(m) | OwnedMCTSInner::MappingFlat { mapping: m, .. } => {
                let idx_ts: Vec<(usize, TimeSeries<'a, T>)> =
                    m.0.values()
                        .enumerate()
                        .map(|(i, ts)| (i, ts.clone()))
                        .collect();
                let passband_vec = self.borrow_passband_vec().clone();
                OwnedMultiColorTimeSeriesBuilder {
                    passband_vec,
                    inner_builder: |vec: &Vec<P>| {
                        OwnedMCTSInner::Mapping(MappedMultiColorTimeSeries(
                            idx_ts
                                .into_iter()
                                .map(|(idx, ts)| (&vec[idx], ts))
                                .collect(),
                        ))
                    },
                }
                .build()
            }
            OwnedMCTSInner::Flat(flat) => {
                type Tmw<T> = (Vec<T>, Vec<T>, Vec<T>);
                let mut map: BTreeMap<P, Tmw<T>> = BTreeMap::new();
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
                Self::from_map(
                    map.into_iter()
                        .map(|(p, (t, m, w))| (p, TimeSeries::new(t, m, w)))
                        .collect::<BTreeMap<_, _>>(),
                )
            }
        }
    }
}

impl<'a, P, T> OwnedMultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn total_lenu(&self) -> usize {
        self.with_inner(|inner| match inner {
            OwnedMCTSInner::Mapping(m) => m.total_lenu(),
            OwnedMCTSInner::Flat(f) => f.total_lenu(),
            OwnedMCTSInner::MappingFlat { flat, .. } => flat.total_lenu(),
        })
    }

    fn total_lenf(&self) -> T {
        self.with_inner(|inner| match inner {
            OwnedMCTSInner::Mapping(m) => m.total_lenf(),
            OwnedMCTSInner::Flat(f) => f.total_lenf(),
            OwnedMCTSInner::MappingFlat { flat, .. } => flat.total_lenf(),
        })
    }

    fn passband_count(&self) -> usize {
        self.borrow_passband_vec().len()
    }

    fn from_map(map: impl Into<BTreeMap<P, TimeSeries<'a, T>>>) -> Self {
        let map: BTreeMap<P, TimeSeries<'a, T>> = map.into();
        let passband_vec: Vec<P> = map.keys().cloned().collect();
        OwnedMultiColorTimeSeriesBuilder {
            passband_vec,
            inner_builder: |vec: &Vec<P>| {
                let mapping = MappedMultiColorTimeSeries(
                    map.into_iter()
                        .enumerate()
                        .map(|(idx, (_, ts))| (&vec[idx], ts))
                        .collect(),
                );
                OwnedMCTSInner::Mapping(mapping)
            },
        }
        .build()
    }

    fn from_flat(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
        w: impl Into<DataSample<'a, T>>,
        passbands: impl AsRef<[P]>,
    ) -> Self {
        let passbands = passbands.as_ref();
        let uniq_passbands: Vec<P> = passbands
            .iter()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .cloned()
            .collect();
        Self::from_flat_with_passband_vec(t, m, w, passbands, uniq_passbands)
    }

    fn from_flat_with_passband_vec(
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

        OwnedMultiColorTimeSeriesBuilder {
            passband_vec,
            inner_builder: |vec: &Vec<P>| {
                let passbands_refs: Vec<&P> = indices.iter().map(|&i| &vec[i]).collect();
                OwnedMCTSInner::Flat(FlatMultiColorTimeSeries {
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
        let needs_mapping = self.with_inner(|inner| matches!(inner, OwnedMCTSInner::Flat(_)));
        if needs_mapping {
            self.with_inner_mut(|inner| {
                take_mut::take(inner, |inner| match inner {
                    OwnedMCTSInner::Flat(mut flat) => {
                        let mapping = MappedMultiColorTimeSeries::from_flat(&mut flat);
                        OwnedMCTSInner::MappingFlat { mapping, flat }
                    }
                    _ => unreachable!("checked above"),
                });
            });
        }
    }

    fn with_mapping_mut<R>(
        &mut self,
        f: impl FnOnce(&mut MappedMultiColorTimeSeries<'_, 'a, P, T>) -> R,
    ) -> R {
        self.ensure_mapping();
        self.with_inner_mut(|inner| match inner {
            OwnedMCTSInner::Mapping(m) => f(m),
            OwnedMCTSInner::MappingFlat { mapping, .. } => f(mapping),
            OwnedMCTSInner::Flat(_) => unreachable!("ensure_mapping was called"),
        })
    }

    fn mapping(&self) -> Option<&MappedMultiColorTimeSeries<'_, 'a, P, T>> {
        match self.borrow_inner() {
            OwnedMCTSInner::Mapping(m) => Some(m),
            OwnedMCTSInner::MappingFlat { mapping, .. } => Some(mapping),
            OwnedMCTSInner::Flat(_) => None,
        }
    }

    fn ensure_flat(&mut self) {
        let needs_flat = self.with_inner(|inner| matches!(inner, OwnedMCTSInner::Mapping(_)));
        if needs_flat {
            self.with_inner_mut(|inner| {
                take_mut::take(inner, |inner| match inner {
                    OwnedMCTSInner::Mapping(mut mapping) => {
                        let flat = FlatMultiColorTimeSeries::from_mapping(&mut mapping);
                        OwnedMCTSInner::MappingFlat { mapping, flat }
                    }
                    _ => unreachable!("checked above"),
                });
            });
        }
    }

    fn with_flat_mut<R>(
        &mut self,
        f: impl FnOnce(&mut FlatMultiColorTimeSeries<'_, 'a, P, T>) -> R,
    ) -> R {
        self.ensure_flat();
        self.with_inner_mut(|inner| match inner {
            OwnedMCTSInner::Flat(fl) => f(fl),
            OwnedMCTSInner::MappingFlat { flat, .. } => f(flat),
            OwnedMCTSInner::Mapping(_) => unreachable!("ensure_flat was called"),
        })
    }

    fn flat(&self) -> Option<&FlatMultiColorTimeSeries<'_, 'a, P, T>> {
        match self.borrow_inner() {
            OwnedMCTSInner::Flat(f) => Some(f),
            OwnedMCTSInner::MappingFlat { flat, .. } => Some(flat),
            OwnedMCTSInner::Mapping(_) => None,
        }
    }

    fn passbands(&self) -> impl Iterator<Item = &P> {
        self.borrow_passband_vec().iter()
    }

    fn insert(&mut self, passband: P, ts: TimeSeries<'a, T>) -> Option<TimeSeries<'a, T>> {
        if self.borrow_passband_vec().binary_search(&passband).is_ok() {
            let mut old = None;
            self.ensure_mapping();
            self.with_inner_mut(|inner| {
                let mapping = match inner {
                    OwnedMCTSInner::Mapping(m) => m,
                    OwnedMCTSInner::MappingFlat { mapping, .. } => mapping,
                    _ => unreachable!(),
                };
                let value = mapping
                    .0
                    .get_mut(&passband as &P)
                    .expect("passband was found in passband_vec, so must be in mapping");
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
                        OwnedMCTSInner::Mapping(m)
                        | OwnedMCTSInner::MappingFlat { mapping: m, .. } => {
                            for (p, ts) in m.0 {
                                owned_map.insert((*p).clone(), ts);
                            }
                        }
                        OwnedMCTSInner::Flat(flat) => {
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
                    OwnedMCTSInner::Mapping(MappedMultiColorTimeSeries(BTreeMap::new()))
                });
            });
            old_ts = owned_map.insert(passband, ts);
            Self::from_map(owned_map)
        });
        old_ts
    }
}

impl<'a, P, T> Default for OwnedMultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn default() -> Self {
        OwnedMultiColorTimeSeriesBuilder {
            passband_vec: Vec::new(),
            inner_builder: |_| OwnedMCTSInner::Mapping(MappedMultiColorTimeSeries(BTreeMap::new())),
        }
        .build()
    }
}

// Borrowed flat variant: passband_slice and per-obs refs all share lifetime 'a with the
// data arrays.  No self-reference, no ouroboros needed.
#[derive(Debug)]
struct BorrowedFlatMCTS<'a, P: PassbandTrait, T: Float> {
    passband_slice: &'a [P],
    flat: FlatMultiColorTimeSeries<'a, 'a, P, T>,
}

impl<'a, P, T> Clone for BorrowedFlatMCTS<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn clone(&self) -> Self {
        Self {
            passband_slice: self.passband_slice,
            flat: self.flat.clone(),
        }
    }
}

impl<'a, P, T> BorrowedFlatMCTS<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    /// Convert to an owned mapping MCTS.  Called when a feature requires the mapping
    /// representation.  Clones K unique passband keys (one per unique band).
    fn into_owned_mapping(self) -> OwnedMultiColorTimeSeries<'a, P, T> {
        type TmwVecs<T> = (Vec<T>, Vec<T>, Vec<T>);
        let mut map: BTreeMap<P, TmwVecs<T>> = BTreeMap::new();
        for (&t, &m, &w, &p) in itertools::multizip((
            self.flat.t.sample.iter(),
            self.flat.m.sample.iter(),
            self.flat.w.sample.iter(),
            self.flat.passbands.iter(),
        )) {
            let entry = map
                .entry((*p).clone())
                .or_insert_with(|| (vec![], vec![], vec![]));
            entry.0.push(t);
            entry.1.push(m);
            entry.2.push(w);
        }
        OwnedMultiColorTimeSeries::from_map(
            map.into_iter()
                .map(|(p, (t, m, w))| (p, TimeSeries::new(t, m, w)))
                .collect::<BTreeMap<_, _>>(),
        )
    }
}

enum MCTSRepr<'a, P: PassbandTrait, T: Float> {
    Owned(OwnedMultiColorTimeSeries<'a, P, T>),
    BorrowedFlat(BorrowedFlatMCTS<'a, P, T>),
}

impl<'a, P: PassbandTrait, T: Float> std::fmt::Debug for MCTSRepr<'a, P, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(o) => f.debug_tuple("Owned").field(o).finish(),
            Self::BorrowedFlat(b) => f.debug_tuple("BorrowedFlat").field(b).finish(),
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
/// * **Borrowed** (`from_flat_borrowed`): both the unique passband slice and the
///   per-observation references borrow from the caller with lifetime `'a` — the same
///   lifetime as the data arrays.  Zero clones, zero allocations for passbands.
#[derive(Debug)]
pub struct MultiColorTimeSeries<'a, P: PassbandTrait + 'a, T: Float> {
    repr: MCTSRepr<'a, P, T>,
}

impl<'a, P, T> Clone for MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn clone(&self) -> Self {
        Self {
            repr: match &self.repr {
                MCTSRepr::Owned(owned) => MCTSRepr::Owned(owned.clone()),
                MCTSRepr::BorrowedFlat(borrowed) => MCTSRepr::BorrowedFlat(borrowed.clone()),
            },
        }
    }
}

impl<'a, P, T> MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    pub fn total_lenu(&self) -> usize {
        match &self.repr {
            MCTSRepr::Owned(owned) => owned.total_lenu(),
            MCTSRepr::BorrowedFlat(borrowed) => borrowed.flat.total_lenu(),
        }
    }

    pub fn total_lenf(&self) -> T {
        match &self.repr {
            MCTSRepr::Owned(owned) => owned.total_lenf(),
            MCTSRepr::BorrowedFlat(borrowed) => borrowed.flat.total_lenf(),
        }
    }

    pub fn passband_count(&self) -> usize {
        match &self.repr {
            MCTSRepr::Owned(owned) => owned.passband_count(),
            MCTSRepr::BorrowedFlat(borrowed) => borrowed.passband_slice.len(),
        }
    }

    /// Build from an owned BTreeMap. Clones K passband keys into `passband_vec`.
    pub fn from_map(map: impl Into<BTreeMap<P, TimeSeries<'a, T>>>) -> Self {
        Self {
            repr: MCTSRepr::Owned(OwnedMultiColorTimeSeries::from_map(map)),
        }
    }

    /// Build from flat arrays, deduplicating passbands automatically.
    pub fn from_flat(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
        w: impl Into<DataSample<'a, T>>,
        passbands: impl AsRef<[P]>,
    ) -> Self {
        Self {
            repr: MCTSRepr::Owned(OwnedMultiColorTimeSeries::from_flat(t, m, w, passbands)),
        }
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
        Self {
            repr: MCTSRepr::Owned(OwnedMultiColorTimeSeries::from_flat_with_passband_vec(
                t,
                m,
                w,
                passbands,
                uniq_passbands,
            )),
        }
    }

    /// Build from flat arrays with fully borrowed passbands — zero clones.
    ///
    /// * `passband_slice` — the K unique passbands, borrowed with `'a`.
    /// * `passbands` — per-observation references into `passband_slice`, also `'a`.
    ///
    /// The caller must ensure every element of `passbands` points into `passband_slice`.
    /// Use this path when a long-lived passband slice already exists (e.g. stored in the
    /// evaluator) and you have pre-built per-observation references.
    pub fn from_flat_borrowed(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
        w: impl Into<DataSample<'a, T>>,
        passbands: Vec<&'a P>,
        passband_slice: &'a [P],
    ) -> Self {
        Self {
            repr: MCTSRepr::BorrowedFlat(BorrowedFlatMCTS {
                passband_slice,
                flat: FlatMultiColorTimeSeries {
                    t: t.into(),
                    m: m.into(),
                    w: w.into(),
                    passbands,
                },
            }),
        }
    }

    fn ensure_mapping(&mut self) {
        match self.repr {
            MCTSRepr::Owned(ref mut o) => o.ensure_mapping(),
            MCTSRepr::BorrowedFlat(_) => {
                take_mut::take(&mut self.repr, |repr| {
                    let MCTSRepr::BorrowedFlat(bf) = repr else {
                        unreachable!()
                    };
                    MCTSRepr::Owned(bf.into_owned_mapping())
                });
            }
        }
    }

    /// Run a closure with mutable access to the mapping representation.
    /// Converts to `MappingFlat` or `Mapping` if currently `Flat`.
    /// A `BorrowedFlat` MCTS is converted to `Owned` (K passband clones) on first call.
    pub fn with_mapping_mut<R>(
        &mut self,
        f: impl FnOnce(&mut MappedMultiColorTimeSeries<'_, 'a, P, T>) -> R,
    ) -> R {
        self.ensure_mapping();
        match &mut self.repr {
            MCTSRepr::Owned(owned) => owned.with_mapping_mut(f),
            MCTSRepr::BorrowedFlat(_) => unreachable!("ensure_mapping was called"),
        }
    }

    pub fn mapping(&self) -> Option<&MappedMultiColorTimeSeries<'_, 'a, P, T>> {
        match &self.repr {
            MCTSRepr::Owned(owned) => owned.mapping(),
            MCTSRepr::BorrowedFlat(_) => None,
        }
    }

    fn ensure_flat(&mut self) {
        match &mut self.repr {
            MCTSRepr::Owned(owned) => owned.ensure_flat(),
            MCTSRepr::BorrowedFlat(_) => {} // already flat
        }
    }

    pub fn with_flat_mut<R>(
        &mut self,
        f: impl FnOnce(&mut FlatMultiColorTimeSeries<'_, 'a, P, T>) -> R,
    ) -> R {
        self.ensure_flat();
        match &mut self.repr {
            MCTSRepr::Owned(owned) => owned.with_flat_mut(f),
            MCTSRepr::BorrowedFlat(borrowed) => f(&mut borrowed.flat),
        }
    }

    pub fn flat(&self) -> Option<&FlatMultiColorTimeSeries<'_, 'a, P, T>> {
        match &self.repr {
            MCTSRepr::Owned(owned) => owned.flat(),
            MCTSRepr::BorrowedFlat(borrowed) => Some(&borrowed.flat),
        }
    }

    pub fn passbands(&self) -> impl Iterator<Item = &P> {
        match &self.repr {
            MCTSRepr::Owned(owned) => itertools::Either::Left(owned.passbands()),
            MCTSRepr::BorrowedFlat(borrowed) => {
                itertools::Either::Right(borrowed.passband_slice.iter())
            }
        }
    }

    /// Inserts a new passband / time-series pair.
    ///
    /// Always converts to mapping form.  A `BorrowedFlat` MCTS is first converted to
    /// `Owned` (K passband clones) before insertion.
    pub fn insert(&mut self, passband: P, ts: TimeSeries<'a, T>) -> Option<TimeSeries<'a, T>> {
        if let MCTSRepr::BorrowedFlat(_) = &self.repr {
            take_mut::take(&mut self.repr, |repr| {
                let MCTSRepr::BorrowedFlat(bf) = repr else {
                    unreachable!()
                };
                MCTSRepr::Owned(bf.into_owned_mapping())
            });
        }
        match &mut self.repr {
            MCTSRepr::Owned(owned) => owned.insert(passband, ts),
            MCTSRepr::BorrowedFlat(_) => unreachable!("converted above"),
        }
    }
}

impl<'a, P, T> Default for MultiColorTimeSeries<'a, P, T>
where
    P: PassbandTrait,
    T: Float,
{
    fn default() -> Self {
        Self {
            repr: MCTSRepr::Owned(OwnedMultiColorTimeSeries::default()),
        }
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
    use approx::assert_relative_eq;
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
        let mapped_from_flat_info: Vec<(String, usize)> = {
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

    #[test]
    fn from_flat_borrowed_zero_clones() {
        let passband_g = MonochromePassband::new(4700.0_f64, "g");
        let passband_r = MonochromePassband::new(6200.0_f64, "r");
        let passband_slice = vec![passband_g.clone(), passband_r.clone()];
        // per-obs refs: g g r r — built by caller, no clones
        let passbands: Vec<&MonochromePassband<f64>> = vec![
            &passband_slice[0],
            &passband_slice[0],
            &passband_slice[1],
            &passband_slice[1],
        ];
        let t = vec![0.0_f64, 1.0, 0.0, 1.0];
        let m = vec![1.0_f64, 2.0, 3.0, 4.0];
        let w = vec![1.0_f64; 4];
        let mcts = MultiColorTimeSeries::from_flat_borrowed(t, m, w, passbands, &passband_slice);
        assert_eq!(mcts.passband_count(), 2);
        assert_eq!(mcts.total_lenu(), 4);
        assert!(mcts.flat().is_some());
        assert!(mcts.mapping().is_none());
    }

    #[test]
    fn borrowed_flat_with_mapping_mut_converts() {
        let passband_g = MonochromePassband::new(4700.0_f64, "g");
        let passband_r = MonochromePassband::new(6200.0_f64, "r");
        let passband_slice = vec![passband_g.clone(), passband_r.clone()];
        let passbands: Vec<&MonochromePassband<f64>> =
            vec![&passband_slice[0], &passband_slice[0], &passband_slice[1]];
        let t = vec![0.0_f64, 1.0, 2.0];
        let m = vec![1.0_f64, 2.0, 3.0];
        let w = vec![1.0_f64; 3];
        let mut mcts =
            MultiColorTimeSeries::from_flat_borrowed(t, m, w, passbands, &passband_slice);
        mcts.with_mapping_mut(|mapping| {
            let g_len = mapping
                .get(&passband_g as &MonochromePassband<f64>)
                .map(|ts| ts.lenu());
            let r_len = mapping
                .get(&passband_r as &MonochromePassband<f64>)
                .map(|ts| ts.lenu());
            assert_eq!(g_len, Some(2));
            assert_eq!(r_len, Some(1));
        });
    }

    fn make_mapping_mcts() -> MultiColorTimeSeries<'static, MonochromePassband<'static, f64>, f64> {
        let g = MonochromePassband::new(4700.0_f64, "g");
        let r = MonochromePassband::new(6200.0_f64, "r");
        let mut map = BTreeMap::new();
        map.insert(
            g,
            TimeSeries::new_without_weight(
                Array1::from(vec![0.0, 1.0, 2.0]),
                Array1::from(vec![1.0, 2.0, 3.0]),
            ),
        );
        map.insert(
            r,
            TimeSeries::new_without_weight(
                Array1::from(vec![0.5, 1.5]),
                Array1::from(vec![10.0, 20.0]),
            ),
        );
        MultiColorTimeSeries::from_map(map)
    }

    fn make_flat_mcts() -> MultiColorTimeSeries<'static, MonochromePassband<'static, f64>, f64> {
        let g = MonochromePassband::new(4700.0_f64, "g");
        let r = MonochromePassband::new(6200.0_f64, "r");
        let passbands = vec![g.clone(), g.clone(), r.clone(), g.clone(), r.clone()];
        let t = vec![0.0_f64, 1.0, 0.5, 2.0, 1.5];
        let m = vec![1.0_f64, 2.0, 10.0, 3.0, 20.0];
        let w = vec![1.0_f64; 5];
        MultiColorTimeSeries::from_flat(t, m, w, passbands)
    }

    #[test]
    fn from_flat_deduplicates_passbands() {
        let mcts = make_flat_mcts();
        assert_eq!(mcts.passband_count(), 2);
        assert_eq!(mcts.total_lenu(), 5);
    }

    #[test]
    fn passbands_returns_sorted_order() {
        let mcts = make_mapping_mcts();
        let names: Vec<&str> = mcts.passbands().map(|p| p.name()).collect();
        assert_eq!(names, vec!["g", "r"]);
    }

    #[test]
    fn mapping_returns_some_for_mapping_variant() {
        let mcts = make_mapping_mcts();
        assert!(mcts.mapping().is_some());
        assert!(mcts.flat().is_none());
    }

    #[test]
    fn flat_returns_some_for_flat_variant() {
        let mcts = make_flat_mcts();
        assert!(mcts.flat().is_some());
        assert!(mcts.mapping().is_none());
    }

    #[test]
    fn with_mapping_mut_converts_flat_to_mapping_flat() {
        let mut mcts = make_flat_mcts();
        assert!(mcts.flat().is_some());
        assert!(mcts.mapping().is_none());
        mcts.with_mapping_mut(|_| {});
        assert!(mcts.mapping().is_some());
        assert!(mcts.flat().is_some());
    }

    #[test]
    fn with_flat_mut_converts_mapping_to_mapping_flat() {
        let mut mcts = make_mapping_mcts();
        assert!(mcts.mapping().is_some());
        assert!(mcts.flat().is_none());
        mcts.with_flat_mut(|_| {});
        assert!(mcts.flat().is_some());
        assert!(mcts.mapping().is_some());
    }

    #[test]
    fn with_mapping_mut_gives_correct_per_band_lengths() {
        let mut mcts = make_flat_mcts();
        mcts.with_mapping_mut(|mapping| {
            let g = MonochromePassband::new(4700.0_f64, "g");
            let r = MonochromePassband::new(6200.0_f64, "r");
            let g_len = mapping
                .get(&g as &MonochromePassband<f64>)
                .map(|ts| ts.lenu());
            let r_len = mapping
                .get(&r as &MonochromePassband<f64>)
                .map(|ts| ts.lenu());
            assert_eq!(g_len, Some(3));
            assert_eq!(r_len, Some(2));
        });
    }

    #[test]
    fn with_flat_mut_produces_sorted_timestamps() {
        let mut mcts = make_mapping_mcts();
        mcts.with_flat_mut(|flat| {
            let t: Vec<f64> = flat.t.as_slice().to_vec();
            for i in 1..t.len() {
                assert!(t[i - 1] <= t[i], "flat timestamps not sorted at index {i}");
            }
        });
    }

    #[test]
    fn clone_from_mapping_preserves_data() {
        let mcts = make_mapping_mcts();
        let cloned = mcts.clone();
        assert_eq!(cloned.passband_count(), mcts.passband_count());
        assert_eq!(cloned.total_lenu(), mcts.total_lenu());
        let names: Vec<&str> = cloned.passbands().map(|p| p.name()).collect();
        assert_eq!(names, vec!["g", "r"]);
    }

    #[test]
    fn clone_from_flat_preserves_data() {
        let mcts = make_flat_mcts();
        assert!(mcts.flat().is_some());
        let cloned = mcts.clone();
        assert_eq!(cloned.passband_count(), 2);
        assert_eq!(cloned.total_lenu(), 5);
    }

    #[test]
    fn clone_from_mapping_flat_preserves_data() {
        let mut mcts = make_flat_mcts();
        mcts.with_mapping_mut(|_| {});
        assert!(mcts.mapping().is_some());
        assert!(mcts.flat().is_some());
        let cloned = mcts.clone();
        assert_eq!(cloned.passband_count(), 2);
        assert_eq!(cloned.total_lenu(), 5);
    }

    #[test]
    fn insert_existing_passband_returns_old_ts() {
        let mut mcts = make_mapping_mcts();
        let g = MonochromePassband::new(4700.0_f64, "g");
        let new_ts = TimeSeries::new_without_weight(
            Array1::from(vec![5.0_f64, 6.0]),
            Array1::from(vec![99.0_f64, 100.0]),
        );
        let old = mcts.insert(g, new_ts);
        assert!(old.is_some());
        assert_eq!(old.unwrap().lenu(), 3);
        assert_eq!(mcts.passband_count(), 2);
        assert_eq!(mcts.total_lenu(), 4);
    }

    #[test]
    fn insert_new_passband_increases_count() {
        let mut mcts = make_mapping_mcts();
        let i_band = MonochromePassband::new(7500.0_f64, "i");
        let new_ts = TimeSeries::new_without_weight(
            Array1::from(vec![0.0_f64, 1.0]),
            Array1::from(vec![5.0_f64, 6.0]),
        );
        let old = mcts.insert(i_band, new_ts);
        assert!(old.is_none());
        assert_eq!(mcts.passband_count(), 3);
        assert_eq!(mcts.total_lenu(), 7);
    }

    #[test]
    fn insert_on_flat_mcts() {
        let mut mcts = make_flat_mcts();
        assert!(mcts.flat().is_some());
        let i_band = MonochromePassband::new(7500.0_f64, "i");
        let new_ts = TimeSeries::new_without_weight(
            Array1::from(vec![0.0_f64]),
            Array1::from(vec![42.0_f64]),
        );
        mcts.insert(i_band, new_ts);
        assert_eq!(mcts.passband_count(), 3);
        assert_eq!(mcts.total_lenu(), 6);
    }

    #[test]
    fn iter_matched_passbands_returns_none_for_missing_band() {
        let mut mcts = make_mapping_mcts();
        let g = MonochromePassband::new(4700.0_f64, "g");
        let i_band = MonochromePassband::new(7500.0_f64, "i");
        let query = [g.clone(), i_band.clone()];
        mcts.with_mapping_mut(|mapping| {
            let results: Vec<_> = mapping.iter_matched_passbands(query.iter()).collect();
            assert_eq!(results.len(), 2);
            assert!(results[0].1.is_some(), "g-band should be present");
            assert!(results[1].1.is_none(), "i-band should be missing");
        });
    }

    #[test]
    fn total_lenu_and_total_lenf_agree() {
        let mapping_mcts = make_mapping_mcts();
        assert_eq!(mapping_mcts.total_lenu(), 5);
        assert_relative_eq!(mapping_mcts.total_lenf(), 5.0_f64, epsilon = 1e-10);

        let flat_mcts = make_flat_mcts();
        assert_eq!(flat_mcts.total_lenu(), 5);
        assert_relative_eq!(flat_mcts.total_lenf(), 5.0_f64, epsilon = 1e-10);
    }

    #[test]
    fn default_is_empty() {
        let mcts: MultiColorTimeSeries<MonochromePassband<f64>, f64> =
            MultiColorTimeSeries::default();
        assert_eq!(mcts.passband_count(), 0);
        assert_eq!(mcts.total_lenu(), 0);
        assert!(mcts.passbands().next().is_none());
    }
}
