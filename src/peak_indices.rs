use crate::float_trait::Float;
use crate::types::ArrayRef1;

/// Find local maxima of the array and return their indices
pub fn peak_indices<T>(a: &ArrayRef1<T>) -> Vec<usize>
where
    T: Float,
{
    a.iter()
        .enumerate()
        .fold(
            (vec![], T::infinity(), false),
            |(mut v, prev_x, prev_is_rising), (i, &x)| {
                let is_rising = x > prev_x;
                if prev_is_rising && (!is_rising) {
                    v.push(i - 1);
                }
                (v, x, is_rising)
            },
        )
        .0
}

/// Find local maxima of the array and return their indices sorted in descending peak value
pub fn peak_indices_reverse_sorted<T>(a: &ArrayRef1<T>) -> Vec<usize>
where
    T: Float,
{
    let mut v = peak_indices(a);
    v[..].sort_unstable_by(|&y, &x| a[x].partial_cmp(&a[y]).unwrap());
    v
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use light_curve_common::linspace;

    macro_rules! peak_indices {
        ($name: ident, $desired: expr, $x: expr $(,)?) => {
            #[test]
            fn $name() {
                let arr = ndarray::arr1(&$x);
                assert_eq!(peak_indices_reverse_sorted(&arr), $desired);
            }
        };
    }

    peak_indices!(
        peak_indices_three_points_peak,
        [1_usize],
        [0.0_f32, 1.0, 0.0]
    );
    peak_indices!(
        peak_indices_three_points_plateau,
        [] as [usize; 0],
        [0.0_f32, 0.0, 0.0]
    );
    peak_indices!(
        peak_indices_three_points_dip,
        [] as [usize; 0],
        [0.0_f32, -1.0, 0.0]
    );
    peak_indices!(peak_indices_long_plateau, [] as [usize; 0], [0.0_f32; 100]);
    peak_indices!(
        peak_indices_sawtooth,
        (1..=99) // the first and the last point cannot be peak
            .filter(|i| i % 2 == 0)
            .collect::<Vec<_>>(),
        (0..=100)
            .map(|i| { if i % 2 == 0 { 1.0_f32 } else { 0.0_f32 } })
            .collect::<Vec<_>>(),
    );
    peak_indices!(
        peak_indices_one_peak,
        [50],
        linspace(-5.0_f32, 5.0, 101)
            .iter()
            .map(|&x| f32::exp(-0.5 * x * x))
            .collect::<Vec<_>>(),
    );
}
