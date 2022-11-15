use light_curve_feature::ndarray::Array1;

// We cannot return `TimeSeries`, because it would cause cyclic crate dependencies
pub(super) type TripleArray<T> = (Array1<T>, Array1<T>, Array1<T>);

#[derive(Debug, thiserror::Error)]
pub(super) enum Error {
    #[error(transparent)]
    CsvError(#[from] csv::Error),
}
