use crate::evaluator::*;

use conv::ConvUtil;

// Gaussian-consistency normalization constant 1 / (sqrt(2) * Phi^{-1}(5/8)),
// matching statsmodels.robust.scale.qn_scale.
const QN_C: f64 = 2.219144465985076;

macro_const! {
    const DOC: &str = r"
Qn — a robust measure of the magnitude scale

The Qn estimator of Rousseeuw & Croux (1993), a robust alternative to the standard deviation
with 50% breakdown point and ~82% Gaussian efficiency (much higher than the median absolute
deviation). Unlike the MAD it does not assume a symmetric distribution and uses pairwise
differences rather than deviations from a center:
$$
Q_n = c \cdot \left\{ |m_i - m_j| ~:~ i < j \right\}_{(k)},
$$
the $k$-th order statistic of all $\binom{N}{2}$ pairwise absolute differences, where
$h = \lfloor N/2 \rfloor + 1$, $k = \binom{h}{2}$, and $c = 1 / (\sqrt2\, \Phi^{-1}(5/8)) \approx
2.219$ makes the estimator consistent with the standard deviation for Gaussian noise.

- Depends on: **magnitude**
- Minimum number of observations: **2**
- Number of features: **1**

Note: this is a naive $O(N^2)$ implementation (in time and memory); it is intended for typical
light-curve lengths.

Rousseeuw & Croux 1993 [DOI:10.1080/01621459.1993.10476408](https://doi.org/10.1080/01621459.1993.10476408)
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub struct QnScale {}

lazy_info!(
    QN_SCALE_INFO,
    QnScale,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
    variability_required: false,
);

impl QnScale {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

impl FeatureNamesDescriptionsTrait for QnScale {
    fn get_names(&self) -> Vec<&str> {
        vec!["qn"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["Qn robust scale estimator of magnitude"]
    }
}

impl<T> FeatureEvaluator<T> for QnScale
where
    T: Float,
{
    fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        let m =
            ts.m.sample
                .as_slice_memory_order()
                .expect("TimeSeries::m::sample is supposed to be contiguous");
        let n = m.len();
        let h = n / 2 + 1;
        let k = h * (h - 1) / 2; // 1-based k; k >= 1 since n >= 2

        // All pairwise absolute differences |m_i - m_j| for i < j.
        let mut diffs: Vec<T> = Vec::with_capacity(n * (n - 1) / 2);
        for (i, &mi) in m.iter().enumerate() {
            for &mj in &m[i + 1..] {
                diffs.push((mi - mj).abs());
            }
        }

        // k-th smallest difference (1-based k -> 0-based k - 1).
        let kth = *diffs
            .select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap())
            .1;

        let c = QN_C.approx_as::<T>().unwrap();
        Ok(vec![c * kth])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(QnScale);

    feature_test!(
        qn_scale,
        [QnScale::new()],
        // statsmodels.robust.scale.qn_scale
        [15.53401126189553],
        [1.0_f64, 2.0, 4.0, 7.0, 11.0, 16.0, 22.0, 100.0],
    );

    #[test]
    fn matches_statsmodels_odd_n() {
        // statsmodels.robust.scale.qn_scale([3,1,4,1,5,9,2,6,5,3,5]) == 2.219144465985076
        let m = [3.0_f64, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0, 5.0, 3.0, 5.0];
        let mut ts = TimeSeries::new_without_weight(&m, &m);
        let actual = QnScale::new().eval(&mut ts).unwrap();
        assert!((actual[0] - 2.219144465985076).abs() < 1e-12);
    }

    #[test]
    fn two_points() {
        // n = 2: Qn = c * |m0 - m1| = 2.219144... * 2
        let m = [10.0_f64, 12.0];
        let mut ts = TimeSeries::new_without_weight(&m, &m);
        let actual = QnScale::new().eval(&mut ts).unwrap();
        assert!((actual[0] - 4.438288931970152).abs() < 1e-12);
    }
}
