use criterion::Criterion;
use light_curve_feature::multicolor::features::{
    ColorOfMaximum, ColorOfMedian, ColorOfMinimum, ColorSpread, MultiColorPeriodogram,
    MultiColorPeriodogramNormalisation,
};
use light_curve_feature::*;
use light_curve_feature_test_util::SNIA_LIGHT_CURVES_FLUX_F64;
use std::collections::BTreeSet;
use std::hint::black_box;

/// Benchmark `from_flat` and `from_flat_borrowed` specifically for the interleaved-passband
/// code paths fixed in https://github.com/light-curve/light-curve-feature/issues/296 and
/// https://github.com/light-curve/light-curve-feature/issues/297.
pub fn bench_multicolor_from_flat(c: &mut Criterion) {
    const N: usize = 3000;
    const K: usize = 3;
    // Must be sorted: from_flat_borrowed requires sorted uniq_passbands for binary_search
    // in the original code path; alphabetical order is g < i < r.
    let band_names = ["g", "i", "r"];
    let uniq_passbands: Vec<StringPassband> = band_names
        .iter()
        .map(|s| StringPassband::from(*s))
        .collect();

    let t: Vec<f64> = (0..N).map(|i| i as f64).collect();
    let m: Vec<f64> = vec![1.0_f64; N];
    let w: Vec<f64> = vec![1.0_f64; N];

    // Interleaved: g, r, i, g, r, i, ...  (worst case for old chunk_by)
    let bands_interleaved: Vec<StringPassband> = (0..N)
        .map(|i| StringPassband::from(band_names[i % K]))
        .collect();

    // Sorted by band: all g, then all r, then all i  (best case for old chunk_by)
    let bands_sorted: Vec<StringPassband> = (0..N)
        .map(|i| StringPassband::from(band_names[i / (N / K)]))
        .collect();

    let feature: MultiColorFeature<StringPassband, f64> =
        ColorSpread::new(uniq_passbands.clone()).into();

    // Benchmark the construction + ensure_mapping path (from_flat → chunk_by / direct dispatch)
    c.bench_function("from_flat interleaved passbands (N=3000, K=3)", |b| {
        b.iter(|| {
            let mut mcts = MultiColorTimeSeries::from_flat(
                t.as_slice(),
                m.as_slice(),
                w.as_slice(),
                black_box(bands_interleaved.as_slice()),
            );
            let _v = feature.eval_multicolor(black_box(&mut mcts)).ok();
        });
    });

    c.bench_function("from_flat sorted passbands (N=3000, K=3)", |b| {
        b.iter(|| {
            let mut mcts = MultiColorTimeSeries::from_flat(
                t.as_slice(),
                m.as_slice(),
                w.as_slice(),
                black_box(bands_sorted.as_slice()),
            );
            let _v = feature.eval_multicolor(black_box(&mut mcts)).ok();
        });
    });

    // Benchmark from_flat_borrowed (pointer equality vs binary search fix, issue #297)
    let bands_borrowed_interleaved: Vec<&StringPassband> =
        (0..N).map(|i| &uniq_passbands[i % K]).collect();
    let bands_borrowed_sorted: Vec<&StringPassband> =
        (0..N).map(|i| &uniq_passbands[i / (N / K)]).collect();

    c.bench_function(
        "from_flat_borrowed interleaved passbands (N=3000, K=3)",
        |b| {
            b.iter(|| {
                let mut mcts = MultiColorTimeSeries::from_flat_borrowed(
                    t.as_slice(),
                    m.as_slice(),
                    w.as_slice(),
                    black_box(bands_borrowed_interleaved.clone()),
                    &uniq_passbands,
                );
                let _v = feature.eval_multicolor(black_box(&mut mcts)).ok();
            });
        },
    );

    c.bench_function("from_flat_borrowed sorted passbands (N=3000, K=3)", |b| {
        b.iter(|| {
            let mut mcts = MultiColorTimeSeries::from_flat_borrowed(
                t.as_slice(),
                m.as_slice(),
                w.as_slice(),
                black_box(bands_borrowed_sorted.clone()),
                &uniq_passbands,
            );
            let _v = feature.eval_multicolor(black_box(&mut mcts)).ok();
        });
    });

    // --- Missing-band and imbalanced cases (capacity allocation stress tests) ---
    //
    // With vec![] capacity=0, each band's Vec reallocates ~log2(N/K) times.
    // These cases isolate whether the dynamic growth cost is significant.

    // Only 2 of 3 bands present (one missing): K_actual=2, N=3000 interleaved g/i
    let bands_2of3: Vec<StringPassband> = (0..N)
        .map(|i| StringPassband::from(band_names[i % 2]))
        .collect();
    let bands_2of3_borrowed: Vec<&StringPassband> =
        (0..N).map(|i| &uniq_passbands[i % 2]).collect();

    c.bench_function("from_flat interleaved 2-of-3 bands present (N=3000)", |b| {
        b.iter(|| {
            let mut mcts = MultiColorTimeSeries::from_flat(
                t.as_slice(),
                m.as_slice(),
                w.as_slice(),
                black_box(bands_2of3.as_slice()),
            );
            let _v = feature.eval_multicolor(black_box(&mut mcts)).ok();
        });
    });

    c.bench_function(
        "from_flat_borrowed interleaved 2-of-3 bands present (N=3000)",
        |b| {
            b.iter(|| {
                let mut mcts = MultiColorTimeSeries::from_flat_borrowed(
                    t.as_slice(),
                    m.as_slice(),
                    w.as_slice(),
                    black_box(bands_2of3_borrowed.clone()),
                    &uniq_passbands,
                );
                let _v = feature.eval_multicolor(black_box(&mut mcts)).ok();
            });
        },
    );

    // Heavily imbalanced: band[0] gets 90%, band[1] gets 9%, band[2] gets 1%
    let bands_imbalanced: Vec<StringPassband> = (0..N)
        .map(|i| {
            let b = if i < N * 9 / 10 {
                0
            } else if i < N * 99 / 100 {
                1
            } else {
                2
            };
            StringPassband::from(band_names[b])
        })
        .collect();

    c.bench_function("from_flat imbalanced bands 90/9/1% (N=3000, K=3)", |b| {
        b.iter(|| {
            let mut mcts = MultiColorTimeSeries::from_flat(
                t.as_slice(),
                m.as_slice(),
                w.as_slice(),
                black_box(bands_imbalanced.as_slice()),
            );
            let _v = feature.eval_multicolor(black_box(&mut mcts)).ok();
        });
    });
}

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
