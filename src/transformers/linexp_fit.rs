use crate::transformers::transformer::*;

use conv::prelude::*;
use macro_const::macro_const;

const INPUT_FEATURE_SIZE: usize = 5;

macro_const! {
    const DOC: &str = r#"
Transform LinexpFit features to be more usable

The LinexpFit feature extractor returns the following features:
- amplitude - kept as is
- fall_slope - kept as is
- baseline - kept as is
- ln1p_linexp_fit_reduced_chi2 - transformed to be less spread
  ln(1 + reduced_chi2)
"#;
}

#[doc = DOC!()]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct LinexpFitTransformer<T> {
    /// Magnitude zero point to use for amplitude transformation
    pub mag_zp: T,
}

impl<T> LinexpFitTransformer<T>
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

impl<T> Default for LinexpFitTransformer<T>
where
    T: Float,
{
    fn default() -> Self {
        Self::new(Self::default_mag_zp())
    }
}

impl<T> TransformerPropsTrait for LinexpFitTransformer<T>
where
    T: Float,
{
    #[inline]
    fn is_size_valid(&self, size: usize) -> bool {
        size == INPUT_FEATURE_SIZE
    }

    #[inline]
    fn size_hint(&self, _size: usize) -> usize {
        4
    }

    fn names(&self, _names: &[&str]) -> Vec<String> {
        vec![
            "linexp_fit_amplitude".into(),
            "linexp_fit_fall_slope".into(),
            "linexp_fit_baseline".into(),
            "ln1p_linexp_fit_reduced_chi2".into(),
        ]
    }

    fn descriptions(&self, _desc: &[&str]) -> Vec<String> {
        vec![
            format!(
                "Full amplitude of Linexp fit in magnitudes, zp={:.2} (zp - 2.5 log10(2A))",
                self.mag_zp
            ),
            "Exponential fall slope the Linexp function (tau_fall)".into(),
            "baseline, B".into(),
            "natural logarithm of unity plus Linexp fit quality (ln(1 + reduced_chi2))".into(),
        ]
    }
}

impl<T> TransformerTrait<T> for LinexpFitTransformer<T>
where
    T: Float,
{
    fn transform(&self, x: Vec<T>) -> Vec<T> {
        let [amplitude, _reference_time, fall_slope, baseline, reduced_chi2]: [T;
            INPUT_FEATURE_SIZE] = match x.try_into() {
            Ok(a) => a,
            Err(x) => panic!(
                "LinexpFitTransformer: expected {} features, found {}",
                INPUT_FEATURE_SIZE,
                x.len()
            ),
        };
        let mag_amplitude = self.mag_zp - T::half() * T::five() * T::log10(T::two() * amplitude);
        let lnp1p_reduced_chi2 = reduced_chi2.ln_1p();
        vec![
            mag_amplitude,
            fall_slope,
            baseline,
            lnp1p_reduced_chi2,
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    check_transformer!(LinexpFitTransformer<f32>);
}
