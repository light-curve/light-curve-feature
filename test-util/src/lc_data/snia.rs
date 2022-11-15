use crate::lc_data::csv_parser::arrays_from_reader;
use crate::lc_data::{MagLightCurveRecord, Record, TripleArray};

use include_dir::{include_dir, Dir};
use lazy_static::lazy_static;
use light_curve_feature::ndarray::Zip;
use light_curve_feature::{Float, TimeSeries};

pub fn iter_sn1a_flux_arrays<T>() -> impl Iterator<Item = (&'static str, TripleArray<T>)>
where
    T: Float,
{
    // Relative to the current file
    const ZTF_IDS_CSV: &str =
        include_str!("../../../test-data/SNIa/snIa_bandg_minobs10_beforepeak3_afterpeak4.csv");

    const LC_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/../test-data/SNIa/light-curves");

    let zero_four = T::two() / T::five();
    let ln10_04 = T::ln(T::ten()) * zero_four;

    ZTF_IDS_CSV.split_terminator('\n').map(move |ztf_id| {
        let filename = format!("{}.csv", ztf_id);
        let file = LC_DIR.get_file(&filename).unwrap();
        let (t, m, w_m) =
            arrays_from_reader::<T, _, MagLightCurveRecord, _>(file.contents(), |record| {
                Record::<T>::band(record) == 'g'
            })
            .unwrap();
        let flux = m.mapv(|x| T::ten().powf(-zero_four * x));
        let w_flux = Zip::from(&w_m)
            .and(&flux)
            .map_collect(|&w_m, &flux| w_m * (flux * ln10_04).powi(-2));
        (ztf_id, (t, flux, w_flux))
    })
}

pub fn iter_sn1a_flux_ts<T>() -> impl Iterator<Item = (&'static str, TimeSeries<'static, T>)>
where
    T: Float,
{
    iter_sn1a_flux_arrays()
        .map(|(name, triple)| (name, TimeSeries::new(triple.0, triple.1, triple.2)))
}

lazy_static! {
    pub static ref SNIA_LIGHT_CURVES_FLUX_F64: Vec<(&'static str, TripleArray<f64>)> =
        iter_sn1a_flux_arrays().collect();
}
