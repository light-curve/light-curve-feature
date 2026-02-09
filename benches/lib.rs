#[macro_use]
extern crate criterion;

mod extractor;
use extractor::bench_extractor;

mod fit;
use fit::{bench_fit_snia, bench_fit_straight_line};

mod periodogram;
use periodogram::{bench_periodogram, bench_periodogram_fft_backends, bench_raw_fft_backends};

mod sin_cos;
use sin_cos::bench_sin_cos;

mod peak_indices;
use peak_indices::bench_peak_indices;

criterion_group!(benches_extractor, bench_extractor<f64>);
criterion_group!(benches_fit, bench_fit_straight_line, bench_fit_snia);
criterion_group!(
    benches_periodogram,
    bench_periodogram,
    bench_periodogram_fft_backends,
    bench_raw_fft_backends
);
criterion_group!(benches_recurrent_sin_cos, bench_sin_cos);
criterion_group!(benches_statistics, bench_peak_indices);

criterion_main!(
    benches_extractor,
    benches_fit,
    benches_periodogram,
    benches_recurrent_sin_cos,
    benches_statistics
);
