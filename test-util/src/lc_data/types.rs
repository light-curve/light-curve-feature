use itertools::{Itertools, izip};
use light_curve_feature::ndarray::Array1;
use light_curve_feature::ndarray::Zip;
use light_curve_feature::{Float, TimeSeries};

// We cannot return `TimeSeries`, because it would cause cyclic crate dependencies
pub(super) type TripleArray<T> = (Array1<T>, Array1<T>, Array1<T>);
pub(super) type QuadrupleArray<T> = (Array1<T>, Array1<T>, Array1<T>, Array1<String>);

#[derive(Clone)]
pub struct MultiColorLightCurve<T> {
    time: Array1<T>,
    brightness: Array1<T>,
    weight: Array1<T>,
    band: Array1<String>,
}

impl<T> MultiColorLightCurve<T> {
    pub(crate) fn new(
        time: impl Into<Array1<T>>,
        brightness: impl Into<Array1<T>>,
        weight: impl Into<Array1<T>>,
        band: impl Into<Array1<String>>,
    ) -> Self {
        Self {
            time: time.into(),
            brightness: brightness.into(),
            weight: weight.into(),
            band: band.into(),
        }
    }

    pub fn into_triple(self, band: Option<&str>) -> TripleArray<T> {
        let (time, brightnes, weight): (Vec<T>, Vec<T>, Vec<T>) =
            izip!(self.time, self.brightness, self.weight, self.band)
                .filter_map(|(obs_time, obs_br, obs_w, obs_band)| {
                    if let Some(band) = band {
                        if obs_band == band {
                            Some((obs_time, obs_br, obs_w))
                        } else {
                            None
                        }
                    } else {
                        Some((obs_time, obs_br, obs_w))
                    }
                })
                .multiunzip();
        (time.into(), brightnes.into(), weight.into())
    }

    pub fn into_quadruple(self) -> QuadrupleArray<T> {
        (self.time, self.brightness, self.weight, self.band)
    }
}

impl<T> MultiColorLightCurve<T>
where
    T: Float,
{
    pub fn convert_mag_to_flux(self) -> Self {
        let zero_four = T::two() / T::five();
        let ln10_04 = T::ln(T::ten()) * zero_four;

        let Self {
            time,
            brightness: mag,
            weight: weight_mag,
            band,
        } = self;

        let flux = mag.mapv(|x| T::ten().powf(-zero_four * x));
        let weight_flux = Zip::from(&weight_mag)
            .and(&flux)
            .map_collect(|&w_m, &f| w_m * (f * ln10_04).powi(-2));

        Self {
            time,
            brightness: flux,
            weight: weight_flux,
            band,
        }
    }

    pub fn into_time_series(self, band: Option<&str>) -> TimeSeries<T> {
        let (time, brightness, weight) = self.into_triple(band);
        TimeSeries::new(time, brightness, weight)
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error(transparent)]
    CsvError(#[from] csv::Error),
}
