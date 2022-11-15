use crate::lc_data::csv_parser::arrays_from_reader;
use crate::lc_data::{Error, FluxLightCurveRecord, MagLightCurveRecord, Record, TripleArray};

use include_dir::{include_dir, Dir};
use light_curve_feature::Float;
use std::path::Path;

const FROM_ISSUES_LC_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/../test-data/from-issues");

fn issue_light_curve<T, P, Ph>(path: P, band: Option<char>) -> Result<TripleArray<T>, Error>
where
    T: Float,
    P: AsRef<Path>,
    Ph: Record<T>,
{
    let data = FROM_ISSUES_LC_DIR.get_file(path).unwrap().contents();
    match band {
        Some(band) => arrays_from_reader::<T, _, Ph, _>(data, |record| record.band() == band),
        None => arrays_from_reader::<T, _, Ph, _>(data, |_| true),
    }
}

pub fn issue_light_curve_mag<T, P>(path: P, band: Option<char>) -> TripleArray<T>
where
    T: Float,
    P: AsRef<Path>,
{
    issue_light_curve::<T, P, MagLightCurveRecord>(path, band).unwrap()
}

pub fn issue_light_curve_flux<T, P>(path: P, band: Option<char>) -> TripleArray<T>
where
    T: Float,
    P: AsRef<Path>,
{
    issue_light_curve::<T, P, FluxLightCurveRecord>(path, band).unwrap()
}

fn iter_issue_light_curve<T, Ph>(
    band: Option<char>,
) -> impl Iterator<Item = (&'static str, TripleArray<T>)>
where
    T: Float,
    Ph: Record<T>,
{
    FROM_ISSUES_LC_DIR
        .find("**/*.csv")
        .unwrap()
        .filter_map(move |entry| {
            let path = entry.as_file()?.path();
            Some((
                path.to_str().unwrap(),
                issue_light_curve::<T, _, Ph>(path, band).ok()?,
            ))
        })
}

pub fn iter_issue_light_curve_mag<T>(
    band: Option<char>,
) -> impl Iterator<Item = (&'static str, TripleArray<T>)>
where
    T: Float,
{
    iter_issue_light_curve::<T, MagLightCurveRecord>(band)
}

pub fn iter_issue_light_curve_flux<T>(
    band: Option<char>,
) -> impl Iterator<Item = (&'static str, TripleArray<T>)>
where
    T: Float,
{
    iter_issue_light_curve::<T, FluxLightCurveRecord>(band)
}
