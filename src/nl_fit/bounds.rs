pub(super) fn within_bounds<T>(x: &[T], lower: &[T], upper: &[T]) -> bool
where
    T: PartialOrd,
{
    for i in 0..x.len() {
        if x[i] < lower[i] || x[i] > upper[i] {
            return false;
        }
    }
    true
}
