use criterion::Criterion;
use light_curve_feature::multicolor::features::{
    ColorOfMaximum, ColorOfMedian, ColorOfMinimum, ColorSpread, MultiColorPeriodogram,
    MultiColorPeriodogramNormalisation,
};
use light_curve_feature::*;
use light_curve_feature_test_util::SNIA_LIGHT_CURVES_FLUX_F64;
use std::collections::BTreeSet;
use std::hint::black_box;

pub fn bench_multicolor(c: &mut Criterion) {
    let passbands = [StringPassband::from("R"), StringPassband::from("g")];
    let required: BTreeSet<_> = passbands.iter().cloned().collect();

    // All SNIa data
    let all_data: Vec<_> = SNIA_LIGHT_CURVES_FLUX_F64
        .iter()
        .map(|(_name, mclc)| {
            let (t, m, w, bands) = mclc.clone().into_quadruple();
            let bands: Vec<StringPassband> = bands
                .iter()
                .map(|s| StringPassband::from(s.as_str()))
                .collect();
            (t, m, w, bands)
        })
        .collect();

    // Only curves with both required passbands (for ColorOf* and MultiColorPeriodogram)
    let two_band_data: Vec<_> = all_data
        .iter()
        .filter(|(_, _, _, bands)| {
            let present: BTreeSet<_> = bands.iter().cloned().collect();
            required.is_subset(&present)
        })
        .collect();

    let color_spread: MultiColorFeature<StringPassband, f64> =
        ColorSpread::new(passbands.clone()).into();
    let color_of_median: MultiColorFeature<StringPassband, f64> =
        ColorOfMedian::new(passbands.clone()).into();
    let color_of_maximum: MultiColorFeature<StringPassband, f64> =
        ColorOfMaximum::new(passbands.clone()).into();
    let color_of_minimum: MultiColorFeature<StringPassband, f64> =
        ColorOfMinimum::new(passbands.clone()).into();
    let extractor: MultiColorExtractor<StringPassband, f64> = MultiColorExtractor::new(vec![
        ColorSpread::new(passbands.clone()).into(),
        ColorOfMedian::new(passbands.clone()).into(),
        ColorOfMaximum::new(passbands.clone()).into(),
        ColorOfMinimum::new(passbands.clone()).into(),
    ]);
    let mcperiodogram: MultiColorFeature<StringPassband, f64> =
        MultiColorPeriodogram::<StringPassband, f64, Feature<f64>>::new(
            5,
            MultiColorPeriodogramNormalisation::Count,
            passbands.clone(),
        )
        .into();

    macro_rules! bench_feature {
        ($name:expr, $feat:expr, $data:expr) => {
            c.bench_function(&format!("MultiColor {} [SNIa flux f64]", $name), |b| {
                b.iter(|| {
                    for (t, m, w, bands) in $data {
                        let mut mcts = MultiColorTimeSeries::from_flat(
                            t.view(),
                            m.view(),
                            w.view(),
                            black_box(bands.as_slice()),
                        );
                        let _v = $feat.eval_multicolor(black_box(&mut mcts)).ok();
                    }
                });
            });
        };
    }

    bench_feature!("ColorSpread", color_spread, &two_band_data);
    bench_feature!("ColorOfMedian", color_of_median, &two_band_data);
    bench_feature!("ColorOfMaximum", color_of_maximum, &two_band_data);
    bench_feature!("ColorOfMinimum", color_of_minimum, &two_band_data);

    c.bench_function(
        "MultiColor MultiColorExtractor (all color features) [SNIa flux f64]",
        |b| {
            b.iter(|| {
                for (t, m, w, bands) in &two_band_data {
                    let mut mcts = MultiColorTimeSeries::from_flat(
                        t.view(),
                        m.view(),
                        w.view(),
                        black_box(bands.as_slice()),
                    );
                    let _v = extractor.eval_multicolor(black_box(&mut mcts)).ok();
                }
            });
        },
    );

    bench_feature!("MultiColorPeriodogram", mcperiodogram, &two_band_data);
}
