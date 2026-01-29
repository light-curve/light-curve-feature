use crate::float_trait::Float;
use crate::time_series::{DataSample, TimeSeries};

use conv::ConvUtil;
use ndarray::Array1;
use std::rc::Rc;

#[derive(Clone, Debug)]
pub struct Data<T> {
    pub t: Array1<T>,
    pub m: Array1<T>,
    pub inv_err: Array1<T>,
}

#[derive(Clone, Debug)]
pub struct NormalizedData<T> {
    pub data: Rc<Data<T>>,
    t_mean: T,
    t_std: T,
    m_mean: T,
    m_std: T,
    inv_err_scale: T,
}

impl<T> NormalizedData<T>
where
    T: Float,
{
    fn normalized<U>(ds: &mut DataSample<U>) -> (T, T, Array1<T>)
    where
        U: Float + conv::ApproxInto<T>,
    {
        let std = ds.get_std().approx_as::<T>().unwrap();
        if std.is_zero() {
            (
                ds.sample[0].approx_as::<T>().unwrap(),
                T::zero(),
                Array1::zeros(ds.sample.len()),
            )
        } else {
            let mean = ds.get_mean().approx_as::<T>().unwrap();
            let v = ds
                .sample
                .mapv(|x| (x.approx_as::<T>().unwrap() - mean) / std);
            (mean, std, v)
        }
    }

    pub fn from_ts<U>(ts: &mut TimeSeries<U>) -> Self
    where
        U: Float + conv::ApproxInto<T>,
    {
        let (t_mean, t_std, t) = Self::normalized(&mut ts.t);
        let (m_mean, m_std, m) = Self::normalized(&mut ts.m);
        let (inv_err_scale, inv_err) = if m_std.is_zero() {
            (
                T::one(),
                ts.w.sample.mapv(|x| x.approx_as::<T>().unwrap().sqrt()),
            )
        } else {
            let scale = m_std.recip();
            let inv_scale =
                ts.w.sample
                    .mapv(|x| x.approx_as::<T>().unwrap().sqrt() * m_std);
            (scale, inv_scale)
        };

        Self {
            data: Rc::new(Data { t, m, inv_err }),
            t_mean,
            t_std,
            m_mean,
            m_std,
            inv_err_scale,
        }
    }

    pub fn t_to_orig(&self, t_norm: T) -> T {
        t_norm * self.t_std + self.t_mean
    }

    pub fn t_to_orig_scale(&self, t_norm: T) -> T {
        t_norm * self.t_std
    }

    pub fn m_to_orig(&self, m_norm: T) -> T {
        m_norm * self.m_std + self.m_mean
    }

    pub fn m_to_orig_scale(&self, m_norm: T) -> T {
        m_norm * self.m_std
    }

    #[allow(dead_code)]
    pub fn inv_err_to_orig(&self, inv_err_norm: T) -> T {
        inv_err_norm * self.inv_err_scale
    }

    pub fn t_to_norm(&self, t_orig: T) -> T {
        if self.t_std.is_zero() {
            T::zero()
        } else {
            (t_orig - self.t_mean) / self.t_std
        }
    }

    pub fn t_to_norm_scale(&self, t_orig: T) -> T {
        if self.t_std.is_zero() {
            t_orig
        } else {
            t_orig / self.t_std
        }
    }

    pub fn m_to_norm(&self, m_orig: T) -> T {
        if self.m_std.is_zero() {
            T::zero()
        } else {
            (m_orig - self.m_mean) / self.m_std
        }
    }

    pub fn m_to_norm_scale(&self, m_orig: T) -> T {
        if self.m_std.is_zero() {
            m_orig
        } else {
            m_orig / self.m_std
        }
    }

    #[allow(dead_code)]
    pub fn inv_err_to_norm(&self, inv_err_orig: T) -> T {
        inv_err_orig / self.inv_err_scale
    }

    /// Get the time standard deviation (scale factor)
    pub fn t_std(&self) -> T {
        self.t_std
    }

    /// Get the magnitude/flux standard deviation (scale factor)
    pub fn m_std(&self) -> T {
        self.m_std
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::time_series::TimeSeries;

    use approx::assert_relative_eq;

    #[test]
    fn test_t_std_m_std_getters() {
        // Create time series with known statistics
        let t = vec![0.0, 10.0, 20.0, 30.0, 40.0];
        let m = vec![100.0, 200.0, 150.0, 180.0, 120.0];
        let mut ts = TimeSeries::new_without_weight(t.clone(), m.clone());
        let norm_data = NormalizedData::<f64>::from_ts(&mut ts);

        // Compute expected std manually using sample std (N-1 divisor, Bessel correction)
        // which is what DataSample::get_std() uses
        let n = t.len() as f64;
        let t_mean: f64 = t.iter().sum::<f64>() / n;
        let t_var: f64 = t.iter().map(|x| (x - t_mean).powi(2)).sum::<f64>() / (n - 1.0);
        let expected_t_std = t_var.sqrt();

        let m_mean: f64 = m.iter().sum::<f64>() / n;
        let m_var: f64 = m.iter().map(|x| (x - m_mean).powi(2)).sum::<f64>() / (n - 1.0);
        let expected_m_std = m_var.sqrt();

        assert_relative_eq!(norm_data.t_std(), expected_t_std, epsilon = 1e-10);
        assert_relative_eq!(norm_data.m_std(), expected_m_std, epsilon = 1e-10);
    }

    #[test]
    fn test_normalization_consistency() {
        // Verify that t_std and m_std are consistent with the conversion functions
        let t = vec![0.0, 10.0, 20.0, 30.0, 40.0];
        let m = vec![100.0, 200.0, 150.0, 180.0, 120.0];
        let mut ts = TimeSeries::new_without_weight(t, m);
        let norm_data = NormalizedData::<f64>::from_ts(&mut ts);

        // t_to_orig_scale should multiply by t_std
        let t_norm = 1.0;
        assert_relative_eq!(
            norm_data.t_to_orig_scale(t_norm),
            t_norm * norm_data.t_std(),
            epsilon = 1e-10
        );

        // m_to_orig_scale should multiply by m_std
        let m_norm = 1.0;
        assert_relative_eq!(
            norm_data.m_to_orig_scale(m_norm),
            m_norm * norm_data.m_std(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_zero_std_handling() {
        // When all values are the same, std should be zero
        let t = vec![5.0, 5.0, 5.0, 5.0];
        let m = vec![100.0, 100.0, 100.0, 100.0];
        let mut ts = TimeSeries::new_without_weight(t, m);
        let norm_data = NormalizedData::<f64>::from_ts(&mut ts);

        assert_eq!(norm_data.t_std(), 0.0);
        assert_eq!(norm_data.m_std(), 0.0);
    }
}
