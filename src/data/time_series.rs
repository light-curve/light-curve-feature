use crate::data::data_sample::DataSample;
use crate::float_trait::Float;

use conv::prelude::*;
use itertools::Itertools;
#[cfg(test)]
use ndarray::Array1;
use ndarray::Zip;
use ndarray_stats::SummaryStatisticsExt;

/// Time series object to be put into [Feature](crate::Feature)
///
/// This struct caches it's properties, like mean magnitude value, etc., that's why mutable
/// reference is required fot feature evaluation
#[derive(Clone, Debug)]
pub struct TimeSeries<'a, T>
where
    T: Float,
{
    pub t: DataSample<'a, T>,
    pub m: DataSample<'a, T>,
    pub w: DataSample<'a, T>,
    m_weighted_mean: Option<T>,
    m_chi2: Option<T>,
    m_reduced_chi2: Option<T>,
    t_max_m: Option<T>,
    t_min_m: Option<T>,
    plateau: Option<bool>,
}

macro_rules! time_series_getter {
    ($t: ty, $attr: ident, $getter: ident, $func: expr) => {
        // This lint is false-positive in macros
        // https://github.com/rust-lang/rust-clippy/issues/1553
        #[allow(clippy::redundant_closure_call)]
        pub fn $getter(&mut self) -> $t {
            match self.$attr {
                Some(x) => x,
                None => {
                    self.$attr = Some($func(self));
                    self.$attr.unwrap()
                }
            }
        }
    };

    ($attr: ident, $getter: ident, $func: expr) => {
        time_series_getter!(T, $attr, $getter, $func);
    };
}

impl<'a, T> TimeSeries<'a, T>
where
    T: Float,
{
    /// Construct `TimeSeries` from array-like objects
    ///
    /// `t` is time, `m` is magnitude (or flux), `w` is weights.
    ///
    /// All arrays must have the same length, `t` must increase monotonically. Input arrays could be
    /// [`ndarray::Array1`], [`ndarray::ArrayView1`], 1-D [`ndarray::CowArray`], or `&[T]`. Several
    /// features assumes that `w` array corresponds to inverse square errors of `m`.
    pub fn new(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
        w: impl Into<DataSample<'a, T>>,
    ) -> Self {
        let t = t.into();
        let m = m.into();
        let w = w.into();

        assert_eq!(
            t.sample.len(),
            m.sample.len(),
            "t and m should have the same size"
        );
        assert_eq!(
            m.sample.len(),
            w.sample.len(),
            "m and err should have the same size"
        );

        Self {
            t,
            m,
            w,
            m_weighted_mean: None,
            m_chi2: None,
            m_reduced_chi2: None,
            t_max_m: None,
            t_min_m: None,
            plateau: None,
        }
    }

    /// Construct [`TimeSeries`] from time and magnitude (flux)
    ///
    /// It is the same as [`TimeSeries::new`], but sets unity weights. It doesn't recommended to use
    /// it for features dependent on weights / observation errors like [`crate::StetsonK`] or
    /// [`crate::LinearFit`].
    pub fn new_without_weight(
        t: impl Into<DataSample<'a, T>>,
        m: impl Into<DataSample<'a, T>>,
    ) -> Self {
        let t = t.into();
        let m = m.into();

        assert_eq!(
            t.sample.len(),
            m.sample.len(),
            "t and m should have the same size"
        );

        let w = T::array0_unity().broadcast(t.sample.len()).unwrap().into();

        Self {
            t,
            m,
            w,
            m_weighted_mean: None,
            m_chi2: None,
            m_reduced_chi2: None,
            t_max_m: None,
            t_min_m: None,
            plateau: None,
        }
    }

    /// Time series length
    #[inline]
    pub fn lenu(&self) -> usize {
        self.t.sample.len()
    }

    /// Float approximating time series length
    pub fn lenf(&self) -> T {
        self.lenu().approx().unwrap()
    }

    time_series_getter!(
        m_weighted_mean,
        get_m_weighted_mean,
        |ts: &mut TimeSeries<T>| { ts.m.sample.weighted_mean(&ts.w.sample).unwrap() }
    );

    time_series_getter!(m_chi2, get_m_chi2, |ts: &mut TimeSeries<T>| {
        let m_weighed_mean = ts.get_m_weighted_mean();
        let m_chi2 = Zip::from(&ts.m.sample)
            .and(&ts.w.sample)
            .fold(T::zero(), |chi2, &m, &w| {
                chi2 + (m - m_weighed_mean).powi(2) * w
            });
        if m_chi2.is_zero() {
            ts.plateau = Some(true);
        }
        m_chi2
    });

    time_series_getter!(m_reduced_chi2, get_m_reduced_chi2, |ts: &mut TimeSeries<
        T,
    >| {
        ts.get_m_chi2() / (ts.lenf() - T::one())
    });

    time_series_getter!(bool, plateau, is_plateau, |ts: &mut TimeSeries<T>| {
        ts.m.is_all_same()
    });

    fn set_t_min_max_m(&mut self) {
        let (i_min, i_max) = self
            .m
            .as_slice()
            .iter()
            .position_minmax()
            .into_option()
            .expect("time series must be non-empty");
        self.t_min_m = Some(self.t.sample[i_min]);
        self.t_max_m = Some(self.t.sample[i_max]);
    }

    pub fn get_t_min_m(&mut self) -> T {
        if self.t_min_m.is_none() {
            self.set_t_min_max_m();
        }
        self.t_min_m.unwrap()
    }

    pub fn get_t_max_m(&mut self) -> T {
        if self.t_max_m.is_none() {
            self.set_t_min_max_m();
        }
        self.t_max_m.unwrap()
    }
}

// We really don't want it to be public, it is a private helper for test-data functions
#[cfg(test)]
impl<'a, T, D> From<(D, D, D)> for TimeSeries<'a, T>
where
    T: Float,
    D: Into<DataSample<'a, T>>,
{
    fn from(v: (D, D, D)) -> Self {
        Self::new(v.0, v.1, v.2)
    }
}

#[cfg(test)]
impl<'a, T> From<&'a (Array1<T>, Array1<T>, Array1<T>)> for TimeSeries<'a, T>
where
    T: Float,
{
    fn from(v: &'a (Array1<T>, Array1<T>, Array1<T>)) -> Self {
        Self::new(v.0.view(), v.1.view(), v.2.view())
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;

    #[test]
    fn time_series_m_weighted_mean() {
        let t: Vec<_> = (0..5).map(|i| i as f64).collect();
        let m = [
            12.77883145,
            18.89988406,
            17.55633632,
            18.36073996,
            11.83854198,
        ];
        let w = [0.1282489, 0.10576467, 0.32102692, 0.12962352, 0.10746144];
        let mut ts = TimeSeries::new(&t, &m, &w);
        // np.average(m, weights=w)
        let desired = 16.31817047752941;
        assert_relative_eq!(ts.get_m_weighted_mean(), desired, epsilon = 1e-6);
    }

    #[test]
    fn time_series_m_reduced_chi2() {
        let t: Vec<_> = (0..5).map(|i| i as f64).collect();
        let m = [
            12.77883145,
            18.89988406,
            17.55633632,
            18.36073996,
            11.83854198,
        ];
        let w = [0.1282489, 0.10576467, 0.32102692, 0.12962352, 0.10746144];
        let mut ts = TimeSeries::new(&t, &m, &w);
        let desired = 1.3752251301435465;
        assert_relative_eq!(ts.get_m_reduced_chi2(), desired, epsilon = 1e-6);
    }
}
