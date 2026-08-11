pub(super) fn within_bounds<T>(x: &[T], lower: &[T], upper: &[T]) -> bool
where
    T: PartialOrd,
{
    x.iter()
        .zip(lower)
        .zip(upper)
        .all(|((xi, li), ui)| xi >= li && xi <= ui)
}
