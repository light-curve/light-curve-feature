pub(super) fn within_bounds<T, const NPARAMS: usize>(
    x: &[T; NPARAMS],
    lower: &[T; NPARAMS],
    upper: &[T; NPARAMS],
) -> bool
where
    T: PartialOrd,
{
    for i in 0..NPARAMS {
        if x[i] < lower[i] || x[i] > upper[i] {
            return false;
        }
    }
    true
}
