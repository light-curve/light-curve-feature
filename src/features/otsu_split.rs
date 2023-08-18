use crate::evaluator::*;
use crate::time_series::DataSample;
use conv::prelude::*;
use ndarray::{s, Array1, ArrayView1, Axis, Zip};
use ndarray_stats::QuantileExt;

macro_const! {
    const DOC: &'static str = r#"
Otsu threshholding algorithm

Difference of subset means, standard deviation of the lower subset, standard deviation of the upper
subset and lower-to-all observation count ratio for two subsets of magnitudes obtained by Otsu's
method split. Otsu's method is used to perform automatic thresholding. The algorithm returns a
single threshold that separate values into two classes. This threshold is determined by minimizing
intra-class intensity variance, or equivalently, by maximizing inter-class variance.
The algorithm returns the minimum threshold which corresponds to the absolute maximum of the inter-class variance.

- Depends on: **magnitude**
- Minimum number of observations: **2**
- Number of features: **4**

Otsu, Nobuyuki 1979. [DOI:10.1109/tsmc.1979.4310076](https://doi.org/10.1109/tsmc.1979.4310076)
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

        let count = ds.sample.len();
        let countf = count.approx().unwrap();
        let sorted = ds.get_sorted();

        if sorted.minimum() == sorted.maximum() {
            return Err(EvaluatorError::FlatTimeSeries);
        }

        // size is (count - 1)
        let cumsum1: Array1<_> = sorted
            .iter()
            .take(count - 1)
            .scan(T::zero(), |state, &m| {
                *state += m;
                Some(*state)
            })
            .collect();

        let cumsum2: Array1<_> = sorted
            .iter()
            .rev()
            .scan(T::zero(), |state, &m| {
                *state += m;
                Some(*state)
            })
            .collect();
        let cumsum2 = cumsum2.slice(s![0..count - 1; -1]);

        let amounts = Array1::linspace(T::one(), (count - 1).approx().unwrap(), count - 1);
        let mean1 = Zip::from(&cumsum1)
            .and(&amounts)
            .map_collect(|&c, &a| c / a);
        let mean2 = Zip::from(&cumsum2)
            .and(amounts.slice(s![..;-1]))
            .map_collect(|&c, &a| c / a);

        let inter_class_variance =
            Zip::from(&amounts)
                .and(&mean1)
                .and(&mean2)
                .map_collect(|&a, &m1, &m2| {
                    let w1 = a / countf;
                    let w2 = T::one() - w1;
                    w1 * w2 * (m1 - m2).powi(2)
                });

        let index = inter_class_variance.argmax().unwrap();

        let (lower, upper) = sorted.0.view().split_at(Axis(0), index + 1);
        Ok((sorted.0[index + 1], lower, upper))
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
        let lower_to_all = lower.sample.len().approx_as::<T>().unwrap() / ts.lenf();

        Ok(vec![mean_diff, std_lower, std_upper, lower_to_all])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;

    use crate::tests::*;

    use approx::assert_relative_eq;
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
    fn otsu_two_max() {
        let mut ds = vec![-1.5, 0.5, 0.5, 1.5].into();
        let (expected_threshold, expected_lower, expected_upper) =
            (0.5, array![-1.5], array![0.5, 0.5, 1.5]);
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

    #[test]
    fn otsu_split_small() {
        let eval = OtsuSplit::new();
        let mut ts = light_curve_feature_test_util::issue_light_curve_mag::<f32, _>(
            "light-curve-feature-72/1.csv",
        )
        .into_triple(None)
        .into();
        let desired = [
            3.0221021243981205,
            0.8847146372743603,
            0.8826366394647659,
            0.507,
        ];
        let actual = eval.eval(&mut ts).unwrap();
        assert_relative_eq!(&desired[..], &actual[..], epsilon = 1e-6);
    }

    #[test]
    #[ignore] // This test takes a long time and requires lots of memory
    fn no_overflow() {
        // It should be large enough to trigger the overflow
        const N: usize = (1 << 25) + 57;
        let feature = OtsuSplit::new();
        let t = Array1::linspace(0.0_f32, 1.0, N);
        let mut ts = TimeSeries::new_without_weight(t.view(), t.view());
        // This should not panic
        let [mean_diff, _std_lower, _std_upper, lower_to_all]: [f32; 4] =
            feature.eval(&mut ts).unwrap().try_into().unwrap();
        assert_relative_eq!(mean_diff, 0.5, epsilon = 1e-3);
        assert_relative_eq!(lower_to_all, 0.5, epsilon = 1e-6);
    }
}
