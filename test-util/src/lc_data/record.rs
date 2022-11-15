use conv::ConvUtil;
use light_curve_feature::Float;
use serde::de::DeserializeOwned;
use serde::Deserialize;

pub(super) trait Record<T>: DeserializeOwned {
    fn into_triple(self) -> (T, T, T);
    fn band(&self) -> char;
}

#[derive(Deserialize)]
pub(super) struct MagLightCurveRecord {
    time: f64,
    mag: f64,
    magerr: f64,
    band: char,
}

impl<T> Record<T> for MagLightCurveRecord
where
    T: Float,
{
    fn into_triple(self) -> (T, T, T) {
        (
            self.time.approx_as().unwrap(),
            self.mag.approx_as().unwrap(),
            self.magerr.approx_as().unwrap(),
        )
    }

    fn band(&self) -> char {
        self.band
    }
}

#[derive(Deserialize)]
pub(super) struct FluxLightCurveRecord {
    time: f64,
    flux: f64,
    fluxerr: f64,
    band: char,
}

impl<T> Record<T> for FluxLightCurveRecord
where
    T: Float,
{
    fn into_triple(self) -> (T, T, T) {
        (
            self.time.approx_as().unwrap(),
            self.flux.approx_as().unwrap(),
            self.fluxerr.approx_as().unwrap(),
        )
    }

    fn band(&self) -> char {
        self.band
    }
}
