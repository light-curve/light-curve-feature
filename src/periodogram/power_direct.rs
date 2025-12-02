use crate::float_trait::Float;
use crate::periodogram::freq::{FreqGrid, FreqGridTrait};
use crate::periodogram::power_trait::*;
use crate::periodogram::sin_cos_iterator::*;
use crate::time_series::TimeSeries;

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Direct periodogram executor
///
/// This algorithm evaluate direct calculation of Lomb-Scargle periodogram. Asymptotic time is
/// $O(N^2)$, so it is recommended to use
/// [PeriodogramPowerFft](crate::periodogram::PeriodogramPowerFft) instead
///
/// The implementation is inspired by Numerical Recipes, Press et al., 1997, Section 13.8
#[derive(Debug, Default, Clone, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(rename = "Direct")]
pub struct PeriodogramPowerDirect;

impl<T> PeriodogramPowerTrait<T> for PeriodogramPowerDirect
where
    T: Float,
{
    fn power(
        &self,
        freq: &FreqGrid<T>,
        ts: &mut TimeSeries<T>,
    ) -> Result<Vec<T>, PeriodogramPowerError> {
        let m_mean = ts.m.get_mean();

        let sin_cos_omega_tau = SinCosOmegaTau::new(freq, ts.t.as_slice().iter());
        let mut sin_cos_omega_x: Vec<_> =
            ts.t.as_slice()
                .iter()
                .map(|&x| freq.iter_sin_cos_mul(x))
                .collect();

        let power = sin_cos_omega_tau
            .take(freq.size())
            .map(|(sin_omega_tau, cos_omega_tau)| {
                let mut sum_m_sin = T::zero();
                let mut sum_m_cos = T::zero();
                let mut sum_sin2 = T::zero();
                for (s_c_omega_x, &y) in sin_cos_omega_x.iter_mut().zip(ts.m.as_slice().iter()) {
                    let (sin_omega_x, cos_omega_x) = s_c_omega_x.next().unwrap();
                    // sine and cosine of omega * (x - tau)
                    let sin = sin_omega_x * cos_omega_tau - cos_omega_x * sin_omega_tau;
                    let cos = cos_omega_x * cos_omega_tau + sin_omega_x * sin_omega_tau;
                    sum_m_sin += (y - m_mean) * sin;
                    sum_m_cos += (y - m_mean) * cos;
                    sum_sin2 += sin.powi(2);
                }
                let sum_cos2 = ts.lenf() - sum_sin2;

                if (sum_m_sin.is_zero() & sum_sin2.is_zero())
                    | (sum_m_cos.is_zero() & sum_cos2.is_zero())
                    | ts.m.get_std2().is_zero()
                {
                    T::zero()
                } else {
                    T::half() * (sum_m_sin.powi(2) / sum_sin2 + sum_m_cos.powi(2) / sum_cos2)
                        / ts.m.get_std2()
                }
            })
            .collect();
        Ok(power)
    }
}

struct SinCosOmegaTau<'a, T> {
    sin_cos_2omega_x: Vec<SinCosIterator<'a, T>>,
}

impl<'a, T: Float> SinCosOmegaTau<'a, T> {
    fn new<'t>(freq_grid: &'a FreqGrid<T>, t: impl Iterator<Item = &'t T>) -> Self {
        let sin_cos_2omega_x = t
            .map(|&x| {
                let two_x = T::two() * x;
                freq_grid.iter_sin_cos_mul(two_x)
            })
            .collect();
        Self { sin_cos_2omega_x }
    }
}

impl<'a, T: Float> Iterator for SinCosOmegaTau<'a, T> {
    type Item = (T, T);

    fn next(&mut self) -> Option<Self::Item> {
        let mut sum_sin = T::zero();
        let mut sum_cos = T::zero();
        for s_c in &mut self.sin_cos_2omega_x {
            let (sin, cos) = s_c.next().unwrap();
            sum_sin += sin;
            sum_cos += cos;
        }
        let cos2 = sum_cos / T::hypot(sum_sin, sum_cos);
        let sin = T::signum(sum_sin) * T::sqrt(T::half() * (T::one() - cos2));
        let cos = T::sqrt(T::half() * (T::one() + cos2));
        Some((sin, cos))
    }
}
