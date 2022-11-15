mod lc_data;
pub use lc_data::from_issues::{
    issue_light_curve_flux, issue_light_curve_mag, iter_issue_light_curve_flux,
    iter_issue_light_curve_mag,
};
pub use lc_data::snia::iter_sn1a_flux_ts;
