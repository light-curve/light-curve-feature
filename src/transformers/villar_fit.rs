use crate::transformers::transformer::*;

use conv::prelude::*;
use macro_const::macro_const;

const INPUT_FEATURE_SIZE: usize = 8;

macro_const! {
    const DOC: &str = r#"
Transform VillarFit features to be more usable

The VillarFit feature extractor returns the following features:
- mag_amplitude - amplitude in fluxes is assumed to be intersect object
  flux, so we transform it to be apparent magnitude: zp - 2.5 log10(2 * A)
- baseline_amplitude_ratio - ratio of baseline to amplitude (both taken
  in original units, not magnitudes, because baseline can be negative)
- rise_time - kept as is
- fall_time - kept as is
- plateau_rel_amplitude - kept as is
- plateau_duration - kept as is
- ln1p_villar_fit_reduced_chi2 - transformed to be less spread
  ln(1 + reduced_chi2)
"#;
}

#[doc = DOC!()]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct VillarFitTransformer<T> {
    /// Magnitude zero point to use for amplitude transformation
    pub mag_zp: T,
}

impl<T> VillarFitTransformer<T>
where
    T: Float,
{
    pub fn new(mag_zp: T) -> Self {
        Self { mag_zp }
    }

    /// ZP for AB-magnitudes and fluxes in janskys
    pub fn default_mag_zp() -> T {
        8.9_f32.value_into().unwrap()
    }

    pub fn doc() -> &'static str {
        DOC
    }
}

impl<T> Default for VillarFitTransformer<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(Self::default_mag_zp())
    }
}

impl<T> TransformerPropsTrait for VillarFitTransformer<T>
where
    T: Float,
{
    #[inline]
    fn is_size_valid(&self, size: usize) -> bool {
        size == INPUT_FEATURE_SIZE
    }

    #[inline]
    fn size_hint(&self, _size: usize) -> usize {
        7
    }

    fn names(&self, _names: &[&str]) -> Vec<String> {
        vec![
            "villar_fit_mag_full_amplitude".into(),
            "villar_fit_baseline_amplitude_ratio".into(),
            "villar_fit_rise_time".into(),
            "villar_fit_fall_time".into(),
            "villar_fit_plateau_rel_amplitude".into(),
            "villar_fit_plateau_duration".into(),
            "ln1p_villar_fit_reduced_chi2".into(),
        ]
    }

    fn descriptions(&self, _desc: &[&str]) -> Vec<String> {
        vec![
            format!(
                "Full amplitude of Villar fit in magnitudes, zp={:.2} (zp - 2.5 log10(2A))",
                self.mag_zp
            ),
            "baseline-to-amplitude ratio of the Villar function (c / A)".into(),
            "rise time of the Villar function (tau_rise)".into(),
            "fall time of the Villar function (tau_fall)".into(),
            "plateau relative amplitude of the Villar function (nu = beta gamma / A)".into(),
            "plateau duration of the Villar function (gamma)".into(),
            "natural logarithm of unity plus Villar fit quality (ln(1 + reduced_chi2))".into(),
        ]
    }
}

impl<T> TransformerTrait<T> for VillarFitTransformer<T>
where
    T: Float,
{
    fn transform(&self, x: Vec<T>) -> Vec<T> {
        let [amplitude, baseline, _reference_time, rise_time, fall_time, rel_plateau_amplitude, plateau_duration, reduced_chi2]: [T;
            INPUT_FEATURE_SIZE] = match x.try_into() {
            Ok(a) => a,
            Err(x) => panic!(
                "VillarFitTransformer: expected {} features, found {}",
                INPUT_FEATURE_SIZE,
                x.len()
            ),
        };
        let mag_amplitude = self.mag_zp - T::half() * T::five() * T::log10(T::two() * amplitude);
        let baseline_amplitude_ratio = baseline / amplitude;
        let lnp1p_reduced_chi2 = reduced_chi2.ln_1p();
        vec![
            mag_amplitude,
            baseline_amplitude_ratio,
            rise_time,
            fall_time,
            rel_plateau_amplitude,
            plateau_duration,
            lnp1p_reduced_chi2,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    check_transformer!(VillarFitTransformer<f32>);
}
