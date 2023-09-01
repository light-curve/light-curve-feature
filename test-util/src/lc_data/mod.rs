use lazy_static::lazy_static;
use record::{FluxLightCurveRecord, MagLightCurveRecord, Record};
use types::{Error, MultiColorLightCurve};

mod csv_parser;
pub(crate) mod from_issues;
mod record;
pub(crate) mod rrlyr;
pub(crate) mod snia;
mod types;

lazy_static! {
    pub static ref ALL_LIGHT_CURVES_F64: Vec<(String, MultiColorLightCurve<f64>)> = {
        let mut v = from_issues::ISSUE_LIGHT_CURVES_ALL_F64.clone();
        v.extend_from_slice(&snia::SNIA_LIGHT_CURVES_FLUX_F64);
        v.extend_from_slice(&rrlyr::RRLYR_LIGHT_CURVES_MAG_F64);
        v
    };
}
