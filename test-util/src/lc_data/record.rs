use conv::ConvUtil;
use light_curve_feature::Float;
use serde::Deserialize;
use serde::de::DeserializeOwned;

pub(super) trait Record<T, B>: DeserializeOwned {
    #[allow(dead_code)]
    fn into_triple(self) -> (T, T, T);
    fn into_quadruple(self) -> (T, T, T, B);
    #[allow(dead_code)]
    fn band(&self) -> B;
}

#[derive(Deserialize)]
pub(super) struct MagLightCurveRecord {
    time: f64,
    mag: f64,
    magerr: f64,
    band: String,
}

impl<T> Record<T, String> for MagLightCurveRecord
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

    fn into_quadruple(self) -> (T, T, T, String) {
        (
            self.time.approx_as().unwrap(),
            self.mag.approx_as().unwrap(),
            self.magerr.approx_as().unwrap(),
            self.band.clone(),
        )
    }

    fn band(&self) -> String {
        self.band.clone()
    }
}

#[derive(Deserialize)]
pub(super) struct FluxLightCurveRecord {
    time: f64,
    flux: f64,
    fluxerr: f64,
    band: String,
}

impl<T> Record<T, String> for FluxLightCurveRecord
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

    fn into_quadruple(self) -> (T, T, T, String) {
        (
            self.time.approx_as().unwrap(),
            self.flux.approx_as().unwrap(),
            self.fluxerr.approx_as().unwrap(),
            self.band.clone(),
        )
    }

    fn band(&self) -> String {
        self.band.clone()
    }
}
