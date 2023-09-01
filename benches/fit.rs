use criterion::{black_box, Criterion};
use light_curve_common::linspace;
#[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
use light_curve_feature::CeresCurveFit;
#[cfg(feature = "gsl")]
use light_curve_feature::LmsderCurveFit;
use light_curve_feature::{
    fit_straight_line, BazinFit, CurveFitAlgorithm, FeatureEvaluator, LnPrior, TimeSeries,
};
use light_curve_feature_test_util::iter_sn1a_flux_ts;
use rand::prelude::*;

pub fn bench_fit_straight_line(c: &mut Criterion) {
    const N: usize = 1000;

    let x = linspace(0.0_f64, 1.0, N);
    let y: Vec<_> = x.iter().map(|&x| x + thread_rng().gen::<f64>()).collect();
    let w: Vec<_> = x
        .iter()
        .map(|_| thread_rng().gen_range(10.0, 100.0))
        .collect();
    let ts = TimeSeries::new(&x, &y, &w);

    c.bench_function("Straight line fit w/o noise", |b| {
        b.iter(|| fit_straight_line(black_box(&ts), false));
    });
    c.bench_function("Straight line fit w/ noise", |b| {
        b.iter(|| fit_straight_line(black_box(&ts), true));
    });
}

pub fn bench_fit_snia(c: &mut Criterion) {
    const N: usize = 100;
    let mut ts_: Vec<_> = iter_sn1a_flux_ts::<f64>(Some("g"))
        .take(N)
        .map(|ts| ts.1)
        .collect();

    let curve_fitters: Vec<(_, CurveFitAlgorithm)> = vec![
        // (
        //     "Bazin SN Ia fit: MCMC 1024 iterations",
        //     McmcCurveFit::new(1024, None).into(),
        // ),
        #[cfg(feature = "gsl")]
        ("Bazin SN Ia fit: Lmsder", LmsderCurveFit::new(20).into()),
        #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
        (
            "Bazin SN Ia fit: Ceres",
            CeresCurveFit::new(20, Some(3.0)).into(),
        ),
    ];

    for (name, curve_fit) in curve_fitters {
        c.bench_function(name, |b| {
            let feature = BazinFit::new(
                curve_fit.clone(),
                LnPrior::none(),
                BazinFit::default_inits_bounds(),
            );
            b.iter(|| {
                ts_.iter_mut().for_each(|ts| {
                    let _ = feature.eval(black_box(ts));
                });
            });
        });
    }
}
