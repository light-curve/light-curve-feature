use crate::evaluator::*;
use crate::time_series::DataSample;
use conv::prelude::*;

macro_const! {
    const DOC: &'static str = r#"
Otsu threshholding algorithm

Difference of subset means, standard deviation of the lower subset, standard deviation of the upper
subset and lower-to-all observation count ratio for two subsets of magnitudes obtained by Otsu's
method split. Otsu's method is used to perform automatic thresholding. The algorithm returns a
single threshold that separate values into two classes. This threshold is determined by minimizing
intra-class intensity variance, or equivalently, by maximizing inter-class variance.

- Depends on: **magnitude**
- Minimum number of observations: **2**
- Number of features: **4**

Otsu, Nobuyuki 1979. [DOI:10.1109/tsmc.1979.4310076](https://doi.org/10.1109/tsmc.1979.4310076)

Matwey Kornilov's Otsu thresholding algorithm realization was used as a reference:
    http://curl.sai.msu.ru/hg/home/matwey/domecam/file/tip/include/otsu.h
    https://ieeexplore.ieee.org/document/9170791
"#;
}

#[doc = DOC ! ()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema)]
pub struct OtsuSplit {}

lazy_info!(
    OTSU_SPLIT_INFO,
    OtsuSplit,
    size: 4,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: false,
    sorting_required: false,
);

impl OtsuSplit {
    pub fn new() -> Self {
        Self {}
    }

    pub fn doc() -> &'static str {
        DOC
    }
}

impl FeatureNamesDescriptionsTrait for OtsuSplit {
    fn get_names(&self) -> Vec<&str> {
        vec![
            "otsu_mean_diff",
            "otsu_std_lower",
            "otsu_std_upper",
            "otsu_lower_to_all_ratio",
        ]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["difference between mean values of Otsu split subsets",
             "standard deviation for observations below the threshold given by Otsu method",
             "standard deviation for observations above the threshold given by Otsu method",
             "ratio of quantity of observations bellow the threshold given by Otsu method to quantity of all observations"]
    }
}

impl<T> FeatureEvaluator<T> for OtsuSplit
where
    T: Float,
{
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts_length(ts)?;

        let msorted = ts.m.get_sorted();

        if msorted.minimum() == msorted.maximum() {
            return Err(EvaluatorError::FlatTimeSeries);
        }

        let mut delta_mean = T::zero();
        let mut w: usize = 0;
        let mean = ts.m.get_mean();
        let count = ts.lenu();
        let mut last_variance = T::zero();

        for &m in msorted.iter() {
            w += 1;
            delta_mean += mean - m;

            if w == count {
                break;
            }

            let variance = delta_mean * delta_mean
                / (count - w).value_as::<T>().unwrap()
                / w.value_as::<T>().unwrap();

            if variance < last_variance {
                break;
            }
            last_variance = variance;
        }

        let std_lower;
        let std_upper;
        let mean_lower;
        let mean_upper;

        if w == 2 {
            std_lower = T::zero();
            mean_lower = msorted[0];
        } else {
            let mut lower: DataSample<_> = msorted[0..w - 1].into();
            std_lower = lower.get_std();
            mean_lower = lower.get_mean()
        }

        if (count - w) == 0 {
            std_upper = T::zero();
            mean_upper = msorted[count - 1];
        } else {
            let mut upper: DataSample<_> = msorted[w..count].into();
            std_upper = upper.get_std();
            mean_upper = upper.get_mean();
        }

        // let mean_diff = upper.get_mean() - lower.get_mean();
        let mean_diff = mean_upper - mean_lower;
        let lower_to_all = (w - 1).value_as::<T>().unwrap() / count.value_as::<T>().unwrap();

        Ok(vec![mean_diff, std_lower, std_upper, lower_to_all])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(OtsuSplit);

    feature_test!(
        otsu_split,
        [OtsuSplit::new()],
        [1.0, 0.0, 0.0, 0.25],
        [0.5, 1.5, 1.5, 1.5],
    );

    #[test]
    fn otsu_split_plateau() {
        let eval = OtsuSplit::new();
        let x = [1.5, 1.5, 1.5, 1.5];
        let mut ts = TimeSeries::new_without_weight(&x, &x);
        assert_eq!(eval.eval(&mut ts), Err(EvaluatorError::FlatTimeSeries));
    }
}
