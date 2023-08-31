use crate::lc_data::csv_parser::mclc_from_reader;
use crate::lc_data::types::MultiColorLightCurve;
use crate::lc_data::MagLightCurveRecord;
use conv::ConvAsUtil;

use include_dir::{include_dir, Dir};
use lazy_static::lazy_static;
use light_curve_feature::Float;

pub struct RrLyr<T> {
    pub id: usize,
    pub subtype: String,
    pub period: T,
    pub light_curve: MultiColorLightCurve<T>,
}

impl<T> RrLyr<T>
where
    T: Float,
{
    pub fn into_name_mclc(self) -> (String, MultiColorLightCurve<T>) {
        (
            format!("{} P={:.3}", self.id, self.period),
            self.light_curve,
        )
    }
}

pub fn iter_rrlyr_multi_color_lcs<T>() -> impl Iterator<Item = RrLyr<T>>
where
    T: Float,
{
    // Relative to the current file
    const SDSS82_IDS_CSV: &[u8] = include_bytes!("../../../test-data/RRLyrae/periods.csv");

    const LC_DIR: Dir = include_dir!("$CARGO_MANIFEST_DIR/../test-data/RRLyrae/light-curves");

    csv::ReaderBuilder::new()
        .from_reader(SDSS82_IDS_CSV)
        .into_records()
        .map(|record| {
            let record = record.unwrap();
            let id = record[0].parse().unwrap();
            let subtype: String = record[1].parse().unwrap();
            let period: f64 = record[2].parse().unwrap();
            let period: T = period.approx_by().unwrap();
            let filename = format!("{}.csv", id);
            let file = LC_DIR.get_file(&filename).unwrap();
            let mclc =
                mclc_from_reader::<T, _, _, MagLightCurveRecord, _>(file.contents(), |_| true)
                    .unwrap();
            RrLyr {
                id,
                subtype,
                period,
                light_curve: mclc,
            }
        })
}

lazy_static! {
    pub static ref RR_LYRAE_F64: Vec<RrLyr<f64>> = iter_rrlyr_multi_color_lcs().collect();
}

lazy_static! {
    pub static ref RRLYR_LIGHT_CURVES_MAG_F64: Vec<(String, MultiColorLightCurve<f64>)> =
        iter_rrlyr_multi_color_lcs()
            .map(|rrlyr| rrlyr.into_name_mclc())
            .collect();
}
