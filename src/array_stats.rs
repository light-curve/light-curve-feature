//! Simple array statistics functions replacing ndarray-stats dependency

use crate::float_trait::Float;
use ndarray::{ArrayRef1, Zip};

/// Find the index of the maximum element in an array
pub fn argmax<T>(arr: &ArrayRef1<T>) -> Option<usize>
where
    T: Float,
{
    if arr.is_empty() {
        return None;
    }

    let (idx, _) = arr
        .iter()
        .enumerate()
        .fold((0, arr[0]), |(max_idx, max_val), (idx, &val)| {
            if val > max_val {
                (idx, val)
            } else {
                (max_idx, max_val)
            }
        });

    Some(idx)
}

/// Compute the weighted mean of an array
pub fn weighted_mean<T>(values: &ArrayRef1<T>, weights: &ArrayRef1<T>) -> Option<T>
where
    T: Float,
{
    if values.is_empty() || values.len() != weights.len() {
        return None;
    }

    let (sum, weight_sum) = Zip::from(values)
        .and(weights)
        .fold((T::zero(), T::zero()), |(sum, weight_sum), &v, &w| {
            (sum + v * w, weight_sum + w)
        });

    if weight_sum.is_zero() {
        None
    } else {
        Some(sum / weight_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array1;

    #[test]
    fn test_argmax_basic() {
        let arr = Array1::from(vec![1.0f64, 3.0, 2.0, 5.0, 4.0]);
        assert_eq!(argmax(&arr), Some(3));
    }

    #[test]
    fn test_argmax_first_element() {
        let arr = Array1::from(vec![5.0f64, 1.0, 2.0]);
        assert_eq!(argmax(&arr), Some(0));
    }

    #[test]
    fn test_argmax_last_element() {
        let arr = Array1::from(vec![1.0f64, 2.0, 5.0]);
        assert_eq!(argmax(&arr), Some(2));
    }

    #[test]
    fn test_argmax_empty() {
        let arr: Array1<f64> = Array1::from(vec![]);
        assert_eq!(argmax(&arr), None);
    }

    #[test]
    fn test_weighted_mean_basic() {
        let values = Array1::from(vec![1.0f64, 2.0, 3.0]);
        let weights = Array1::from(vec![1.0f64, 1.0, 1.0]);
        // Simple average: (1 + 2 + 3) / 3 = 2.0
        let result = weighted_mean(&values, &weights).unwrap();
        assert!((result - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean_different_weights() {
        let values = Array1::from(vec![1.0f64, 2.0]);
        let weights = Array1::from(vec![1.0f64, 3.0]);
        // (1*1 + 2*3) / (1 + 3) = 7/4 = 1.75
        let result = weighted_mean(&values, &weights).unwrap();
        assert!((result - 1.75).abs() < 1e-10);
    }

    #[test]
    fn test_weighted_mean_empty() {
        let values: Array1<f64> = Array1::from(vec![]);
        let weights: Array1<f64> = Array1::from(vec![]);
        assert!(weighted_mean(&values, &weights).is_none());
    }
}
