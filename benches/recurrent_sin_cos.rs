use criterion::Criterion;
use light_curve_feature::RecurrentSinCos;
use light_curve_feature::periodogram::sin_cos_iterator::SinCosIterator;
use std::hint::black_box;

fn plain(n: usize, x: f64) {
    for i in 1..=n {
        black_box(f64::sin_cos(x * (i as f64)));
    }
}

fn rec(n: usize, x: f64) {
    for s_c in RecurrentSinCos::with_zero_first(x).take(n) {
        black_box(s_c);
    }
}

fn wrap_rec(n: usize, x: f64) {
    for s_c in SinCosIterator::LinearGrid(RecurrentSinCos::with_zero_first(x)).take(n) {
        black_box(s_c);
    }
}

pub fn bench_recurrent_sin_cos(c: &mut Criterion) {
    const COUNTS: [usize; 4] = [1, 10, 100, 1000];

    for &n in COUNTS.iter() {
        c.bench_function(format!("Plain sin_cos {}", n).as_str(), |b| {
            b.iter(|| plain(black_box(n), 0.01))
        });
        c.bench_function(format!("Rec sin_cos {}", n).as_str(), |b| {
            b.iter(|| rec(black_box(n), 0.01))
        });
        c.bench_function(format!("Wrapped Rec sin_cos {}", n).as_str(), |b| {
            b.iter(|| wrap_rec(black_box(n), 0.01))
        });
    }
}
