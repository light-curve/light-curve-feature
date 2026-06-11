use criterion::Criterion;
use light_curve_feature::multicolor::features::{
    ColorOfMaximum, ColorOfMedian, ColorOfMinimum, ColorSpread, MultiColorPeriodogram,
    MultiColorPeriodogramNormalisation,
};
use light_curve_feature::*;
use light_curve_feature_test_util::SNIA_LIGHT_CURVES_FLUX_F64;
use std::collections::BTreeSet;
use std::hint::black_box;

/// Inner helper: run all from_flat / from_flat_borrowed bench variants for a given band set.
/// `band_names` must be sorted (required by from_flat_borrowed's binary_search contract).
fn bench_from_flat_variants(c: &mut Criterion, band_names: &[&str], n: usize) {
    let k = band_names.len();
    assert_eq!(n % k, 0, "N must be divisible by K for even split");

    let uniq_passbands: Vec<StringPassband> = band_names
        .iter()
        .map(|s| StringPassband::from(*s))
        .collect();
    let t: Vec<f64> = (0..n).map(|i| i as f64).collect();
    let m: Vec<f64> = vec![1.0_f64; n];
    let w: Vec<f64> = vec![1.0_f64; n];

    // Interleaved: b0, b1, …, bK-1, b0, b1, …  (worst case for old chunk_by)
    let bands_interleaved: Vec<StringPassband> = (0..n)
        .map(|i| StringPassband::from(band_names[i % k]))
        .collect();
    // Sorted by band: all b0, then all b1, …  (best case for old chunk_by)
    let bands_sorted: Vec<StringPassband> = (0..n)
        .map(|i| StringPassband::from(band_names[i / (n / k)]))
        .collect();
    // (K-1) of K bands present, interleaved — one band missing
    let bands_km1: Vec<StringPassband> = (0..n)
        .map(|i| StringPassband::from(band_names[i % (k - 1)]))
        .collect();
    // Heavily imbalanced: first band gets 90 %, second 9 %, rest share 1 %
    let bands_imbalanced: Vec<StringPassband> = (0..n)
        .map(|i| {
            let b = if i < n * 9 / 10 {
                0
            } else if i < n * 99 / 100 {
                1
            } else {
                2.min(k - 1)
            };
            StringPassband::from(band_names[b])
        })
        .collect();

    let feature: MultiColorFeature<StringPassband, f64> =
        ColorSpread::new(uniq_passbands.clone()).into();

    let bands_borrowed_interleaved: Vec<&StringPassband> =
        (0..n).map(|i| &uniq_passbands[i % k]).collect();
    let bands_borrowed_sorted: Vec<&StringPassband> =
        (0..n).map(|i| &uniq_passbands[i / (n / k)]).collect();

    macro_rules! bench_flat {
        ($label:expr, $bands:expr) => {
            c.bench_function($label, |b| {
                b.iter(|| {
                    let mut mcts = MultiColorTimeSeries::from_flat(
                        t.as_slice(),
                        m.as_slice(),
                        w.as_slice(),
                        black_box($bands.as_slice()),
                    );
                    let _v = feature.eval_multicolor(black_box(&mut mcts)).ok();
                });
            });
        };
    }
    macro_rules! bench_borrowed {
        ($label:expr, $bands:expr) => {
            c.bench_function($label, |b| {
                b.iter(|| {
                    let mut mcts = MultiColorTimeSeries::from_flat_borrowed(
                        t.as_slice(),
                        m.as_slice(),
                        w.as_slice(),
                        black_box($bands.clone()),
                        &uniq_passbands,
                    );
                    let _v = feature.eval_multicolor(black_box(&mut mcts)).ok();
                });
            });
        };
    }

    bench_flat!(
        &format!("from_flat interleaved (N={n}, K={k})"),
        bands_interleaved
    );
    bench_flat!(&format!("from_flat sorted (N={n}, K={k})"), bands_sorted);
    bench_flat!(
        &format!("from_flat {}-of-{k} bands interleaved (N={n})", k - 1),
        bands_km1
    );
    bench_flat!(
        &format!("from_flat imbalanced 90/9/1% (N={n}, K={k})"),
        bands_imbalanced
    );
    bench_borrowed!(
        &format!("from_flat_borrowed interleaved (N={n}, K={k})"),
        bands_borrowed_interleaved
    );
    bench_borrowed!(
        &format!("from_flat_borrowed sorted (N={n}, K={k})"),
        bands_borrowed_sorted
    );
}

/// Benchmark `from_flat` and `from_flat_borrowed` for K=3 and K=6 passbands.
/// Covers interleaved, sorted, missing-band, and imbalanced cases.
/// Fixes: https://github.com/light-curve/light-curve-feature/issues/296 and /297.
pub fn bench_multicolor_from_flat(c: &mut Criterion) {
    // K=3, sorted alphabetically: g < i < r
    bench_from_flat_variants(c, &["g", "i", "r"], 3000);
    // K=6, LSST-like bands sorted alphabetically: g < i < r < u < y < z
    bench_from_flat_variants(c, &["g", "i", "r", "u", "y", "z"], 3000);
}

/// Micro-benchmark the per-observation passband lookup inside `from_flat_with_passband_vec`.
/// Parameterised over K (number of unique passbands) so we can see where linear scan wins.
pub fn bench_passband_lookup(c: &mut Criterion) {
    const N: usize = 3000;
    for k in [2_usize, 3, 4, 6, 8, 12, 16, 32] {
        let band_names: Vec<String> = (0..k)
            .map(|i| format!("{}", char::from(b'a' + i as u8)))
            .collect();
        let uniq: Vec<StringPassband> = band_names
            .iter()
            .map(|s| StringPassband::from(s.as_str()))
            .collect();
        let obs: Vec<StringPassband> = (0..N)
            .map(|i| StringPassband::from(band_names[i % k].as_str()))
            .collect();

        c.bench_function(&format!("passband_lookup/linear K={k} N={N}"), |b| {
            b.iter(|| {
                let refs: Vec<&StringPassband> = obs
                    .iter()
                    .map(|p| black_box(&uniq).iter().find(|q| *q == p).unwrap())
                    .collect();
                black_box(refs)
            });
        });

        c.bench_function(&format!("passband_lookup/binary K={k} N={N}"), |b| {
            b.iter(|| {
                let refs: Vec<&StringPassband> = obs
                    .iter()
                    .map(|p| {
                        let idx = black_box(&uniq).binary_search(p).unwrap();
                        &uniq[idx]
                    })
                    .collect();
                black_box(refs)
            });
        });
    }
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
