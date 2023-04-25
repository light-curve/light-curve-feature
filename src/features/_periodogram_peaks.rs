use crate::evaluator::*;
use crate::evaluator::{Deserialize, EvaluatorInfo, EvaluatorProperties, Serialize};
use crate::peak_indices::peak_indices_reverse_sorted;
use crate::{
    number_ending, EvaluatorError, EvaluatorInfoTrait, FeatureEvaluator,
    FeatureNamesDescriptionsTrait, Float, TimeSeries,
};

use schemars::JsonSchema;
use std::iter;

macro_const! {
    const PERIODOGRAM_PEAKS_DOC: &'static str = r#"
Peak evaluator for [Periodogram]

- Depends on: **time**, **magnitude** (which have meaning of frequency and spectral density)
- Minimum number of observations: **1**
- Number of features: **2 * npeaks**
"#;
}

#[doc(hidden)]
#[doc = PERIODOGRAM_PEAKS_DOC!()]
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(
    from = "PeriodogramPeaksParameters",
    into = "PeriodogramPeaksParameters"
)]
pub struct PeriodogramPeaks {
    peaks: usize,
    properties: Box<EvaluatorProperties>,
}

impl PeriodogramPeaks {
    pub fn new(peaks: usize) -> Self {
        assert!(peaks > 0, "Number of peaks should be at least one");
        let info = EvaluatorInfo {
            size: 2 * peaks,
            min_ts_length: 1,
            t_required: true,
            m_required: true,
            w_required: false,
            sorting_required: true,
            variability_required: false,
        };
        let names = (0..peaks)
            .flat_map(|i| vec![format!("period_{}", i), format!("period_s_to_n_{}", i)])
            .collect();
        let descriptions = (0..peaks)
            .flat_map(|i| {
                vec![
                    format!(
                        "period of the {}{} highest peak of periodogram",
                        i + 1,
                        number_ending(i + 1),
                    ),
                    format!(
                        "Spectral density to spectral density standard deviation ratio of \
                            the {}{} highest peak of periodogram",
                        i + 1,
                        number_ending(i + 1)
                    ),
                ]
            })
            .collect();
        Self {
            properties: EvaluatorProperties {
                info,
                names,
                descriptions,
            }
            .into(),
            peaks,
        }
    }

    pub fn get_peaks(&self) -> usize {
        self.peaks
    }

    #[inline]
    pub fn default_peaks() -> usize {
        1
    }

    pub const fn doc() -> &'static str {
        PERIODOGRAM_PEAKS_DOC
    }
}

impl Default for PeriodogramPeaks {
    fn default() -> Self {
        Self::new(Self::default_peaks())
    }
}

impl EvaluatorInfoTrait for PeriodogramPeaks {
    fn get_info(&self) -> &EvaluatorInfo {
        &self.properties.info
    }
}

impl FeatureNamesDescriptionsTrait for PeriodogramPeaks {
    fn get_names(&self) -> Vec<&str> {
        self.properties.names.iter().map(String::as_str).collect()
    }

    fn get_descriptions(&self) -> Vec<&str> {
        self.properties
            .descriptions
            .iter()
            .map(String::as_str)
            .collect()
    }
}

impl<T> FeatureEvaluator<T> for PeriodogramPeaks
where
    T: Float,
{
    fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        let peak_indices = peak_indices_reverse_sorted(&ts.m.sample);
        Ok(peak_indices
            .iter()
            .flat_map(|&i| {
                iter::once(T::two() * T::PI() / ts.t.sample[i])
                    .chain(iter::once(ts.m.signal_to_noise(ts.m.sample[i])))
            })
            .chain(iter::repeat(T::zero()))
            .take(2 * self.peaks)
            .collect())
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "PeriodogramPeaks")]
struct PeriodogramPeaksParameters {
    peaks: usize,
}

impl From<PeriodogramPeaks> for PeriodogramPeaksParameters {
    fn from(f: PeriodogramPeaks) -> Self {
        Self { peaks: f.peaks }
    }
}

impl From<PeriodogramPeaksParameters> for PeriodogramPeaks {
    fn from(p: PeriodogramPeaksParameters) -> Self {
        Self::new(p.peaks)
    }
}

impl JsonSchema for PeriodogramPeaks {
    json_schema!(PeriodogramPeaksParameters, false);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(PeriodogramPeaks);
}
