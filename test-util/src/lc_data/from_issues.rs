use crate::lc_data::csv_parser::mclc_from_reader;
use crate::lc_data::{
    Error, FluxLightCurveRecord, MagLightCurveRecord, MultiColorLightCurve, Record,
};

use include_dir::{include_dir, Dir};
use lazy_static::lazy_static;
use light_curve_feature::Float;
use light_curve_feature::TimeSeries;
use std::path::Path;

const FROM_ISSUES_LC_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/../test-data/from-issues");

fn issue_light_curve<T, B, P, Ph>(path: P) -> Result<MultiColorLightCurve<T>, Error>
where
    T: Float,
    P: AsRef<Path>,
    Ph: Record<T, B>,
    B: ToString,
{
    let data = FROM_ISSUES_LC_DIR.get_file(path).unwrap().contents();
    mclc_from_reader::<T, _, _, Ph, _>(data, |_| true)
}

pub fn issue_light_curve_mag<T, P>(path: P) -> MultiColorLightCurve<T>
where
    T: Float,
    P: AsRef<Path>,
{
    issue_light_curve::<T, String, P, MagLightCurveRecord>(path).unwrap()
}

pub fn issue_light_curve_flux<T, P>(path: P) -> MultiColorLightCurve<T>
where
    T: Float,
    P: AsRef<Path>,
{
    issue_light_curve::<T, String, P, FluxLightCurveRecord>(path).unwrap()
}

pub fn issue_ts_mag<T, P>(path: P, band: Option<&'static str>) -> TimeSeries<'static, T>
where
    T: Float,
    P: AsRef<Path>,
{
    issue_light_curve::<T, String, P, MagLightCurveRecord>(path)
        .unwrap()
        .into_time_series(band)
}

pub fn issue_ts_flux<T, P>(path: P, band: Option<&'static str>) -> TimeSeries<'static, T>
where
    T: Float,
    P: AsRef<Path>,
{
    issue_light_curve::<T, String, P, FluxLightCurveRecord>(path)
        .unwrap()
        .into_time_series(band)
}

fn iter_issues_light_curve<T, B, Ph>() -> impl Iterator<Item = (String, MultiColorLightCurve<T>)>
where
    T: Float,
    Ph: Record<T, B>,
    B: ToString,
{
    FROM_ISSUES_LC_DIR
        .find("**/*.csv")
        .unwrap()
        .filter_map(move |entry| {
            let path = entry.as_file()?.path();
            Some((
                path.to_str().unwrap().to_owned(),
                issue_light_curve::<T, _, _, Ph>(path).ok()?,
            ))
        })
}

pub fn iter_issue_light_curves_mag<T>() -> impl Iterator<Item = (String, MultiColorLightCurve<T>)>
where
    T: Float,
{
    iter_issues_light_curve::<T, String, MagLightCurveRecord>()
}

pub fn iter_issue_light_curves_flux<T>() -> impl Iterator<Item = (String, MultiColorLightCurve<T>)>
where
    T: Float,
{
    iter_issues_light_curve::<T, String, FluxLightCurveRecord>()
}

pub fn iter_issue_ts_mag<T>(
    band: Option<&'static str>,
) -> impl Iterator<Item = (String, TimeSeries<'static, T>)>
where
    T: Float,
{
    iter_issues_light_curve::<T, String, MagLightCurveRecord>()
        .map(move |(name, mclc)| (name, mclc.into_time_series(band)))
}

pub fn iter_issue_ts_flux<T>(
    band: Option<&'static str>,
) -> impl Iterator<Item = (String, TimeSeries<'static, T>)>
where
    T: Float,
{
    iter_issues_light_curve::<T, String, FluxLightCurveRecord>()
        .map(move |(name, mclc)| (name, mclc.into_time_series(band)))
}

lazy_static! {
    pub static ref ISSUE_LIGHT_CURVES_MAG_F64: Vec<(String, MultiColorLightCurve<f64>)> =
        iter_issue_light_curves_mag().collect();
}

lazy_static! {
    pub static ref ISSUE_LIGHT_CURVES_FLUX_F64: Vec<(String, MultiColorLightCurve<f64>)> =
        iter_issue_light_curves_flux().collect();
}

lazy_static! {
    pub static ref ISSUE_LIGHT_CURVES_ALL_F64: Vec<(String, MultiColorLightCurve<f64>)> = {
        let mut v = ISSUE_LIGHT_CURVES_MAG_F64.clone();
        v.extend_from_slice(&ISSUE_LIGHT_CURVES_FLUX_F64);
        v
    };
}
