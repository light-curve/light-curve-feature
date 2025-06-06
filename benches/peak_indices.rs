use criterion::Criterion;

use light_curve_common::linspace;
use light_curve_feature::peak_indices;
use std::hint::black_box;

macro_rules! gen_bench {
    ($c: expr_2021, $name: expr_2021, $func: ident, $x: expr_2021 $(,)?) => {
        $c.bench_function($name, move |b| b.iter(|| black_box($func(&$x))));
    };
}

pub fn bench_peak_indices(c: &mut Criterion) {
    macro_rules! b {
        ($name: expr_2021, $x: expr_2021 $(,)?) => {
            gen_bench!(c, $name, peak_indices, $x);
        };
    }

    b!("peak_indices_three_points_f32", [0.0_f32, 1.0, 0.0]);
    b!("peak_indices_three_points_f64", [0.0_f64, 1.0, 0.0]);
    b!("peak_indices_plateau", [0.0_f32; 100]);
    b!(
        "peak_indices_gauss",
        linspace(-5.0_f32, 3.0, 100)
            .iter()
            .map(|&x| f32::exp(-0.5 * x * x))
            .collect::<Vec<_>>(),
    );
    b!(
        "peak_indices_sawtooth",
        (0..=100)
            .map(|i| { if i % 2 == 0 { 1.0_f32 } else { 0.0_f32 } })
            .collect::<Vec<_>>(),
    );
}
