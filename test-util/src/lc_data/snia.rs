use crate::lc_data::csv_parser::mclc_from_reader;
use crate::lc_data::{MagLightCurveRecord, MultiColorLightCurve};

use include_dir::{include_dir, Dir};
use lazy_static::lazy_static;
use light_curve_feature::{Float, TimeSeries};

pub fn iter_sn1a_flux_arrays<T>() -> impl Iterator<Item = (String, MultiColorLightCurve<T>)>
where
    T: Float,
{
    // Relative to the current file
    const ZTF_IDS_CSV: &str =
        include_str!("../../../test-data/SNIa/snIa_bandg_minobs10_beforepeak3_afterpeak4.csv");

    const LC_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/../test-data/SNIa/light-curves");

    ZTF_IDS_CSV.split_terminator('\n').map(move |ztf_id| {
        let filename = format!("{}.csv", ztf_id);
        let file = LC_DIR.get_file(&filename).unwrap();
        let mclc_mag =
            mclc_from_reader::<T, _, _, MagLightCurveRecord, _>(file.contents(), |_| true).unwrap();
        let mclc_flux = mclc_mag.convert_mag_to_flux();
        (ztf_id.to_owned(), mclc_flux)
    })
}

pub fn iter_sn1a_flux_ts<T>(
    band: Option<&'static str>,
) -> impl Iterator<Item = (String, TimeSeries<'static, T>)> + 'static
where
    T: Float,
{
    iter_sn1a_flux_arrays().map(move |(name, mclc)| (name, mclc.into_time_series(band)))
}

lazy_static! {
    pub static ref SNIA_LIGHT_CURVES_FLUX_F64: Vec<(String, MultiColorLightCurve<f64>)> =
        iter_sn1a_flux_arrays().collect();
}
