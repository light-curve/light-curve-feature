use crate::evaluator::*;

macro_const! {
    const DOC: &str = r"
Skewness of magnitude $G_1$

$$
G_1 \equiv \frac{N}{(N - 1)(N - 2)} \frac{\sum_i(m_i - \langle m \rangle)^3}{\sigma_m^3},
$$
where $N$ is the number of observations,
$\langle m \rangle$ is the mean magnitude,
$\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}$ is the magnitude standard deviation.

- Depends on: **magnitude**
- Minimum number of observations: **3**
- Number of features: **1**

[Wikipedia](https://en.wikipedia.org/wiki/Skewness#Sample_skewness)
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]
pub struct Skew {}

lazy_info!(
    SKEW_INFO,
    Skew,
    size: 1,
    min_ts_length: 3,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl Skew {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

impl FeatureNamesDescriptionsTrait for Skew {
    fn get_names(&self) -> Vec<&str> {
        vec!["skew"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["skew of magnitude sample"]
    }
}

impl<T> FeatureEvaluator<T> for Skew
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let m_std = get_nonzero_m_std(ts)?;
        let m_mean = ts.m.get_mean();
        let n = ts.lenf();
        let n_1 = n - T::one();
        let n_2 = n_1 - T::one();
        let third_moment =
            ts.m.sample
                .fold(T::zero(), |sum, &m| sum + (m - m_mean).powi(3));
        Ok(vec![third_moment / m_std.powi(3) * n / (n_1 * n_2)])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(Skew);

    feature_test!(
        skew,
        [Skew::new()],
        [0.4626804756753222],
        [2.0_f32, 3.0, 5.0, 7.0, 11.0, 13.0],
    );
}
