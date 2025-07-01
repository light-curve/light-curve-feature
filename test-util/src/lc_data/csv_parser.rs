use crate::lc_data::{Error, MultiColorLightCurve, Record};

use itertools::{Itertools, process_results};
use light_curve_feature::Float;
use std::io::Read;

pub(super) fn mclc_from_reader<T, B, R, Ph, F>(
    reader: R,
    filter: F,
) -> Result<MultiColorLightCurve<T>, Error>
where
    T: Float,
    R: Read,
    F: Fn(&Ph) -> bool,
    Ph: Record<T, B>,
    B: ToString,
{
    let mut csv_reader = csv::ReaderBuilder::new().from_reader(reader);
    let iter = csv_reader
        .deserialize()
        .filter(|record: &Result<Ph, _>| match record {
            Ok(record) => filter(record),
            Err(_) => true,
        })
        .map(|record| -> Result<_, csv::Error> {
            let (time, mag, magerr, band) = record?.into_quadruple();
            let band: String = band.to_string();
            let w = magerr.powi(-2);
            Ok((time, mag, w, band))
        });
    let (time, mag, w, band): (Vec<T>, Vec<T>, Vec<T>, Vec<String>) =
        process_results(iter, |iter| iter.multiunzip())?;
    Ok(MultiColorLightCurve::new(time, mag, w, band))
}
