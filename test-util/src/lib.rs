pub use lc_data::ALL_LIGHT_CURVES_F64;
pub use lc_data::from_issues::{
    ISSUE_LIGHT_CURVES_ALL_F64, ISSUE_LIGHT_CURVES_FLUX_F64, ISSUE_LIGHT_CURVES_MAG_F64,
    issue_light_curve_flux, issue_light_curve_mag, issue_ts_flux, issue_ts_mag,
    iter_issue_light_curves_flux, iter_issue_light_curves_mag, iter_issue_ts_flux,
    iter_issue_ts_mag,
};
pub use lc_data::rrlyr::RRLYR_LIGHT_CURVES_MAG_F64;
pub use lc_data::snia::{SNIA_LIGHT_CURVES_FLUX_F64, iter_sn1a_flux_arrays, iter_sn1a_flux_ts};

mod lc_data;
