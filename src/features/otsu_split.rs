use crate::evaluator::*;
use crate::time_series::DataSample;
use conv::prelude::*;
use ndarray::{ArrayView1, Axis};

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

    pub fn threshold<'a, 'b, T>(
        ds: &'b mut DataSample<'a, T>,
    ) -> Result<(T, ArrayView1<'b, T>, ArrayView1<'b, T>), EvaluatorError>
    where
        'a: 'b,
        T: Float,
    {
        if ds.sample.len() < 2 {
            return Err(EvaluatorError::ShortTimeSeries {
                actual: ds.sample.len(),
                minimum: 2,
            });
        }

        let mut delta_mean = T::zero();
        let mut w = 0;
        let mean = ds.get_mean();
        let count = ds.sample.len();
        let sorted = ds.get_sorted();

        if sorted.minimum() == sorted.maximum() {
            return Err(EvaluatorError::FlatTimeSeries);
        }

        let mut last_variance = T::zero();

        for &m in sorted.iter() {
            w += 1;
            delta_mean += mean - m;

            let variance = delta_mean * delta_mean
                / (count - w).value_as::<T>().unwrap()
                / w.value_as::<T>().unwrap();

            if variance < last_variance {
                break;
            }
            last_variance = variance;
        }

        let (lower, upper) = sorted.0.view().split_at(Axis(0), w - 1);
        Ok((sorted.0[w - 1], lower, upper))
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

        let (_, lower, upper) = Self::threshold(&mut ts.m)?;
        let mut lower: DataSample<_> = lower.into();
        let mut upper: DataSample<_> = upper.into();

        let std_lower = if lower.sample.len() == 1 {
            T::zero()
        } else {
            lower.get_std()
        };
        let mean_lower = lower.get_mean();

        let std_upper = if upper.sample.len() == 1 {
            T::zero()
        } else {
            upper.get_std()
        };
        let mean_upper = upper.get_mean();

        let mean_diff = mean_upper - mean_lower;
        let lower_to_all = lower.sample.len().value_as::<T>().unwrap() / ts.lenf();

        Ok(vec![mean_diff, std_lower, std_upper, lower_to_all])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;
    use ndarray::array;

    check_feature!(OtsuSplit);

    feature_test!(
        otsu_split,
        [OtsuSplit::new()],
        [
            0.725,
            0.012909944487358068,
            0.07071067811865482,
            0.6666666666666666
        ],
        [0.51, 0.52, 0.53, 0.54, 1.2, 1.3],
    );

    feature_test!(
        otsu_split_min_observations,
        [OtsuSplit::new()],
        [0.01, 0.0, 0.0, 0.5],
        [0.51, 0.52],
    );

    feature_test!(
        otsu_split_lower,
        [OtsuSplit::new()],
        [1.0, 0.0, 0.0, 0.25],
        [0.5, 1.5, 1.5, 1.5],
    );

    feature_test!(
        otsu_split_upper,
        [OtsuSplit::new()],
        [1.0, 0.0, 0.0, 0.75],
        [0.5, 0.5, 0.5, 1.5],
    );

    #[test]
    fn otsu_threshold() {
        let mut ds = vec![0.5, 0.5, 0.5, 1.5].into();
        let (expected_threshold, expected_lower, expected_upper) =
            (1.5, array![0.5, 0.5, 0.5], array![1.5]);
        let (actual_threshold, actual_lower, actual_upper) =
            OtsuSplit::threshold(&mut ds).expect("input is not flat");
        assert_eq!(expected_threshold, actual_threshold);
        assert_eq!(expected_lower, actual_lower);
        assert_eq!(expected_upper, actual_upper);
    }

    #[test]
    fn otsu_split_plateau() {
        let eval = OtsuSplit::new();
        let x = [1.5, 1.5, 1.5, 1.5];
        let mut ts = TimeSeries::new_without_weight(&x, &x);
        assert_eq!(eval.eval(&mut ts), Err(EvaluatorError::FlatTimeSeries));
    }
}
