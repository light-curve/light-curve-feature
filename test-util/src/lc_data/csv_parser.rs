use crate::lc_data::{Error, Record, TripleArray};

use conv::ConvUtil;
use itertools::{process_results, Itertools};
use light_curve_feature::Float;
use std::io::Read;

pub(super) fn arrays_from_reader<T, R, Ph, F>(reader: R, filter: F) -> Result<TripleArray<T>, Error>
where
    T: Float,
    R: Read,
    F: Fn(&Ph) -> bool,
    Ph: Record<T>,
{
    let mut csv_reader = csv::ReaderBuilder::new().from_reader(reader);
    let iter = csv_reader
        .deserialize()
        .filter(|record: &Result<Ph, _>| match record {
            Ok(record) => filter(record),
            Err(_) => true,
        })
        .map(|record| -> Result<_, csv::Error> {
            let (time, mag, magerr) = record?.into_triple();
            let w = magerr.powi(-2);
            Ok((
                time.approx_as::<T>().unwrap(),
                mag.approx_as::<T>().unwrap(),
                w.approx_as::<T>().unwrap(),
            ))
        });
    let (t, m, w): (Vec<_>, Vec<_>, Vec<_>) = process_results(iter, |iter| iter.multiunzip())?;
    Ok((t.into(), m.into(), w.into()))
}
