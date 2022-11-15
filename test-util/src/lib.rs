pub use lc_data::from_issues::{
    issue_light_curve_flux, issue_light_curve_mag, iter_issue_light_curves_flux,
    iter_issue_light_curves_mag, ISSUE_LIGHT_CURVES_ALL_F64, ISSUE_LIGHT_CURVES_FLUX_F64,
    ISSUE_LIGHT_CURVES_MAG_F64,
};
pub use lc_data::snia::{iter_sn1a_flux_arrays, iter_sn1a_flux_ts, SNIA_LIGHT_CURVES_FLUX_F64};
pub use lc_data::ALL_LIGHT_CURVES_F64;

mod lc_data;
