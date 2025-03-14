use crate::evaluator::*;

macro_const! {
    const DOC: &str = r"
    Robust median statistic

    $$
    \frac1{N-1} \sum_{i=0}^{N-1} \frac{|m_i - \mathrm{median}(m_i)|}{\sigma_i}
    $$
    For non-variable data, it should be less than one.

    - Depends on: **magnitude**, **errors**
    - Minimum number of observations: **2**
    - Number of features: **1**

    Enoch, Brown, Burgasser 2003. [DOI:10.1086/376598](https://www.doi.org/10.1086/376598)
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]

pub struct Roms {}

impl Roms {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

lazy_info!(
    ROMS_INFO,
    Roms,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: true,
    sorting_required: false,
);

impl FeatureNamesDescriptionsTrait for Roms {
    fn get_names(&self) -> Vec<&str> {
        vec!["roms"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["robust median statistic"]
    }
}

impl<T> FeatureEvaluator<T> for Roms
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;
        let n = ts.lenf();
        let m_median = ts.m.get_median();
        let tmp_sum =
            ts.m.as_slice()
                .iter()
                .zip(ts.w.as_slice().iter())
                .map(|(&m, &w)| ((m - m_median).abs() * w.sqrt()))
                .filter(|&x| x.is_finite())
                .sum::<T>();
        let value = tmp_sum / (n - T::one());
        Ok(vec![value])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::extractor::FeatureExtractor;
    use crate::tests::*;

    check_feature!(Roms);

    feature_test!(
        roms,
        [Roms::new()],
        [2.6035533],
        [1.0_f32, 2.0, 3.0, 4.0, 5.0],
        [1.0_f32, 1.0, 2.0, 3.0, 5.0],
        [1.0_f32, 4.0, 1.0, 2.0, 4.0],
    );

    #[test]
    fn roms_const_data() {
        let eval = Roms::default();
        let x = linspace(0.0_f32, 10.0, 10);
        let y = vec![1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let z = vec![1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let mut ts = TimeSeries::new(&x, &y, &z);
        let actual = eval.eval(&mut ts).unwrap();
        let desired = [0.0_f32];
        all_close(&actual, &desired, 1e-10);
    }
}
