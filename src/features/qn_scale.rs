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

The $O(N \log N)$ algorithm of Croux & Rousseeuw (1992) is used (no $O(N^2)$ materialization
of the pairwise differences).

- Depends on: **magnitude**
- Minimum number of observations: **2**
- Number of features: **1**

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
        let c = QN_C.approx_as::<T>().unwrap();
        Ok(vec![c * qn_unscaled_fast(m)])
    }
}

/// Weighted high median: the smallest `a[j]` such that the total weight of all `a[i] <= a[j]`
/// is strictly greater than half of the total weight.
fn whimed<T: Float>(a: &[T], w: &[i64]) -> T {
    let mut a_cp = a.to_vec();
    let mut w_cp = w.to_vec();
    let mut n = a_cp.len();
    let wtot: i64 = w_cp.iter().sum();
    let mut wrest: i64 = 0;
    // Reused scratch for the per-iteration median selection (avoids reallocating).
    let mut scratch: Vec<T> = vec![T::zero(); n];
    loop {
        // trial = the (n / 2)-th order statistic (0-based) of the current candidates.
        let mid = n / 2;
        let buf = &mut scratch[..n];
        buf.copy_from_slice(&a_cp[..n]);
        buf.select_nth_unstable_by(mid, |x, y| x.partial_cmp(y).unwrap());
        let trial = buf[mid];

        // Iterating over exact-length slices lets the optimizer drop bounds checks.
        let mut wleft = 0i64;
        let mut wright = 0i64;
        let mut wcur = 0i64;
        for (&ai, &wi) in a_cp[..n].iter().zip(&w_cp[..n]) {
            wcur += wi;
            if ai < trial {
                wleft += wi;
            } else if ai > trial {
                wright += wi;
            }
        }
        let wmid = wcur - wleft - wright;

        if 2 * (wrest + wleft) > wtot {
            // The answer is among the smaller candidates.
            let mut kc = 0;
            for i in 0..n {
                if a_cp[i] < trial {
                    a_cp[kc] = a_cp[i];
                    w_cp[kc] = w_cp[i];
                    kc += 1;
                }
            }
            n = kc;
        } else if 2 * (wrest + wleft + wmid) <= wtot {
            // The answer is among the larger candidates.
            let mut kc = 0;
            for i in 0..n {
                if a_cp[i] > trial {
                    a_cp[kc] = a_cp[i];
                    w_cp[kc] = w_cp[i];
                    kc += 1;
                }
            }
            n = kc;
            wrest += wleft + wmid;
        } else {
            return trial;
        }
    }
}

/// Unscaled Qn: the `k`-th order statistic of the pairwise absolute differences, computed with
/// the naive `O(N^2)` algorithm. Kept as a reference/oracle for the fast implementation.
#[cfg(test)]
fn qn_unscaled_naive<T: Float>(m: &[T]) -> T {
    let n = m.len();
    let h = n / 2 + 1;
    let k = h * (h - 1) / 2; // 1-based

    let mut diffs: Vec<T> = Vec::with_capacity(n * (n - 1) / 2);
    for (i, &mi) in m.iter().enumerate() {
        for &mj in &m[i + 1..] {
            diffs.push((mi - mj).abs());
        }
    }
    *diffs
        .select_nth_unstable_by(k - 1, |a, b| a.partial_cmp(b).unwrap())
        .1
}

/// Unscaled Qn computed with the `O(N log N)` algorithm of Croux & Rousseeuw (1992),
/// ported from the reference C implementation in R's `robustbase`.
fn qn_unscaled_fast<T: Float>(x: &[T]) -> T {
    let n = x.len();
    let ni = n as i64;

    let mut y: Vec<T> = x.to_vec();
    y.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

    let h = n / 2 + 1;
    let k = (h * (h - 1) / 2) as i64; // raw 1-based order-statistic index
    let nn2 = ni * (ni + 1) / 2;
    let mut nl = nn2;
    let mut nr = ni * ni;
    let knew = k + nl;

    // y is 0-based but the algorithm indexes pairwise differences as y[i] - y[n - col];
    // `at` converts those (always in [0, n)) 1-based column offsets to a slice index.
    let at = |a: i64| -> usize {
        debug_assert!((0..ni).contains(&a), "qn index {a} out of [0, {ni})");
        a as usize
    };

    let mut left = vec![0i64; n];
    let mut right = vec![0i64; n];
    for (i, l) in left.iter_mut().enumerate() {
        *l = ni - i as i64 + 1;
    }
    let parity = (n % 2) as f64;
    let k_l =
        (5.0 - 1.75 * parity + (0.3939 - 0.0067 * parity) * (n as f64) * ((n - 1) as f64)) as i64;
    if k >= k_l {
        right.fill(ni);
    } else {
        let hi = h as i64;
        for (i, r) in right.iter_mut().enumerate() {
            let ii = i as i64;
            *r = if ii <= hi { ni } else { ni - (ii - hi) };
        }
    }

    let mut work: Vec<T> = vec![T::zero(); n];
    let mut weight: Vec<i64> = vec![0; n];
    let mut p = vec![0i64; n];
    let mut q = vec![0i64; n];

    while nr - nl > ni {
        let mut j = 0usize;
        for i in 1..n {
            if left[i] <= right[i] {
                weight[j] = right[i] - left[i] + 1;
                let jh = left[i] + weight[j] / 2;
                work[j] = y[i] - y[at(ni - jh)];
                j += 1;
            }
        }
        let trial = whimed(&work[..j], &weight[..j]);

        // p[i] = number of differences in row i strictly below `trial` (two-pointer).
        let mut jp: i64 = 0;
        for i in (0..n).rev() {
            while jp < ni && y[i] - y[at(ni - jp - 1)] < trial {
                jp += 1;
            }
            p[i] = jp;
        }
        // q[i] = number of differences in row i at or below `trial`.
        let mut jq: i64 = ni + 1;
        for i in 0..n {
            loop {
                let id = ni - jq + 1;
                if !(0..ni).contains(&id) {
                    break;
                }
                if y[i] - y[id as usize] <= trial {
                    break;
                }
                jq -= 1;
            }
            q[i] = jq;
        }

        let sump: i64 = p.iter().sum();
        let sumq: i64 = q.iter().map(|&v| v - 1).sum();

        if knew <= sump {
            right.copy_from_slice(&p);
            nr = sump;
        } else if knew > sumq {
            left.copy_from_slice(&q);
            nl = sumq;
        } else {
            return trial; // sump < knew <= sumq
        }
    }

    // Few candidates remain: collect and select directly.
    let mut rest: Vec<T> = Vec::new();
    for i in 1..n {
        let mut jj = left[i];
        while jj <= right[i] {
            rest.push(y[i] - y[at(ni - jj)]);
            jj += 1;
        }
    }
    let knew2 = (knew - (nl + 1)).clamp(0, rest.len() as i64 - 1) as usize;
    *rest
        .select_nth_unstable_by(knew2, |a, b| a.partial_cmp(b).unwrap())
        .1
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

    #[test]
    fn fast_matches_naive() {
        let mut rng = StdRng::seed_from_u64(0);
        for n in 2..=60_usize {
            for _ in 0..20 {
                // Mix of continuous values and small integers to exercise ties.
                let m: Vec<f64> = (0..n)
                    .map(|_| {
                        if rng.random::<bool>() {
                            rng.random_range(-5.0..5.0)
                        } else {
                            f64::from(rng.random_range(-3..3))
                        }
                    })
                    .collect();
                let fast = qn_unscaled_fast(&m);
                let naive = qn_unscaled_naive(&m);
                assert!(
                    (fast - naive).abs() < 1e-12,
                    "n={n}: fast={fast} naive={naive} m={m:?}"
                );
            }
        }
    }
}
