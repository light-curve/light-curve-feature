use conv::ConvUtil;
use criterion::{BatchSize, Criterion};
use light_curve_feature::*;
use light_curve_feature_test_util::iter_sn1a_flux_ts;
use ndarray::Array1;
use rand::prelude::*;
use rand_distr::StandardNormal;
use std::any::type_name;
use std::hint::black_box;

pub fn bench_extractor<T>(c: &mut Criterion)
where
    T: Float + 'static,
    StandardNormal: Distribution<T>,
{
    const N: [usize; 2] = [100, 1000];

    let features: Vec<Feature<_>> = vec![
        Amplitude::default().into(),
        AndersonDarlingNormal::default().into(),
        BeyondNStd::default().into(),
        Cusum::default().into(),
        Eta::default().into(),
        EtaE::default().into(),
        ExcessVariance::default().into(),
        InterPercentileRange::default().into(),
        Kurtosis::default().into(),
        LinearFit::default().into(),
        LinearTrend::default().into(),
        MagnitudePercentageRatio::default().into(),
        MaximumSlope::default().into(),
        Mean::default().into(),
        MeanVariance::default().into(),
        Median::default().into(),
        MedianAbsoluteDeviation::default().into(),
        MedianBufferRangePercentage::default().into(),
        OtsuSplit::default().into(),
        PercentAmplitude::default().into(),
        PercentDifferenceMagnitudePercentile::default().into(),
        ReducedChi2::default().into(),
        Skew::default().into(),
        StandardDeviation::default().into(),
        StetsonK::default().into(),
        WeightedMean::default().into(),
    ];

    let observation_count_vec: Vec<_> = (0..20)
        .map(|_| ObservationCount::default().into())
        .collect();

    let beyond_n_std_vec: Vec<_> = (1usize..21)
        .map(|i| BeyondNStd::new(i.value_as::<f32>().unwrap() / 10.0).into())
        .collect();

    let mut bins = Bins::default();
    bins.add_feature(StetsonK::default().into());

    let mut periodogram = Periodogram::default();
    periodogram.set_max_freq_factor(10.0);

    let names_fes: Vec<_> = features
        .iter()
        .map(|f| (f.get_names()[0], FeatureExtractor::new(vec![f.clone()])))
        .chain(std::iter::once((
            "all non-meta features",
            FeatureExtractor::new(features.clone()),
        )))
        .chain(std::iter::once((
            "multiple ObservationCount",
            FeatureExtractor::new(observation_count_vec.clone()),
        )))
        .chain(std::iter::once((
            "multiple BeyondNStd",
            FeatureExtractor::new(beyond_n_std_vec),
        )))
        .chain(std::iter::once((
            "BazinFit",
            FeatureExtractor::new(vec![BazinFit::default().into()]),
        )))
        .chain(std::iter::once((
            "Bins",
            FeatureExtractor::new(vec![bins.into()]),
        )))
        .chain(std::iter::once((
            "Periodogram",
            FeatureExtractor::new(vec![periodogram.into()]),
        )))
        .chain(std::iter::once((
            "VillarFit",
            FeatureExtractor::new(vec![VillarFit::default().into()]),
        )))
        .chain(std::iter::once((
            "LinearFit",
            FeatureExtractor::new(vec![LinearFit::default().into()]),
        )))
        .chain(std::iter::once((
            "AndersonDarlingNormal",
            FeatureExtractor::new(vec![AndersonDarlingNormal::default().into()]),
        )))
        .chain(std::iter::once((
            "BiweightScale",
            FeatureExtractor::new(vec![BiweightScale::default().into()]),
        )))
        .chain(std::iter::once((
            "Eta",
            FeatureExtractor::new(vec![Eta::default().into()]),
        )))
        .chain(std::iter::once((
            "EtaE",
            FeatureExtractor::new(vec![EtaE::default().into()]),
        )))
        .chain(std::iter::once((
            "ExcessVariance",
            FeatureExtractor::new(vec![ExcessVariance::default().into()]),
        )))
        .chain(std::iter::once((
            "Kurtosis",
            FeatureExtractor::new(vec![Kurtosis::default().into()]),
        )))
        .chain(std::iter::once((
            "LaflerKinmanStringLength",
            FeatureExtractor::new(vec![LaflerKinmanStringLength::default().into()]),
        )))
        .chain(std::iter::once((
            "ReducedChi2",
            FeatureExtractor::new(vec![ReducedChi2::default().into()]),
        )))
        .chain(std::iter::once((
            "Roms",
            FeatureExtractor::new(vec![Roms::default().into()]),
        )))
        .chain(std::iter::once((
            "Skew",
            FeatureExtractor::new(vec![Skew::default().into()]),
        )))
        .chain(std::iter::once((
            "StetsonK",
            FeatureExtractor::new(vec![StetsonK::default().into()]),
        )))
        .collect();

    for &n in N.iter() {
        // Pristine, never-evaluated time series: `TimeSeries` memoizes derived statistics, so
        // it must not be reused across iterations or all but the first would time a cache hit.
        let ts = randts(n);
        for (name, fe) in names_fes.iter() {
            c.bench_function(
                format!("FeatureExtractor {}: [{}; {}]", name, n, type_name::<T>()).as_str(),
                |b| {
                    b.iter_batched_ref(
                        || ts.clone(),
                        |ts| {
                            let _v = fe.eval(black_box(ts)).unwrap();
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
        }
    }

    {
        let n = 10;
        let ts = randts(n);
        let fe = FeatureExtractor::new(observation_count_vec);
        c.bench_function(
            format!("Multiple ObservationCount {}", type_name::<T>()).as_str(),
            |b| {
                b.iter_batched_ref(
                    || ts.clone(),
                    |ts| {
                        let _v = fe.eval(black_box(ts)).unwrap();
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }

    {
        let real_data: Vec<_> = iter_sn1a_flux_ts::<T>(Some("g"))
            .map(|(_ztf_id, ts)| ts)
            .collect();
        #[allow(clippy::vec_init_then_push)]
        let curve_fits: Vec<CurveFitAlgorithm> = {
            let mut curve_fits = vec![];
            #[cfg(feature = "gsl")]
            {
                curve_fits.push(LmsderCurveFit::new(5).into());
                curve_fits.push(LmsderCurveFit::new(10).into());
                curve_fits.push(LmsderCurveFit::new(15).into());
            }
            curve_fits.push(McmcCurveFit::new(128, None).into());
            curve_fits.push(McmcCurveFit::new(1024, None).into());
            #[cfg(feature = "gsl")]
            {
                curve_fits.push(McmcCurveFit::new(128, Some(LmsderCurveFit::new(5).into())).into());
                curve_fits
                    .push(McmcCurveFit::new(1024, Some(LmsderCurveFit::new(10).into())).into());
            }
            curve_fits
        };
        for curve_fit in curve_fits.into_iter() {
            let features: Vec<Feature<_>> = vec![
                BazinFit::new(
                    curve_fit.clone(),
                    LnPrior::none(),
                    BazinFit::default_inits_bounds(),
                )
                .into(),
                VillarFit::new(
                    curve_fit,
                    LnPrior::none(),
                    VillarFit::default_inits_bounds(),
                )
                .into(),
            ];
            for f in features {
                c.bench_function(
                    format!("SN Ia {:?} {}", f, type_name::<T>()).as_str(),
                    |b| {
                        b.iter_batched_ref(
                            || real_data.clone(),
                            |real_data| {
                                real_data.iter_mut().for_each(|ts| {
                                    let _v = f.eval(black_box(ts)).unwrap();
                                });
                            },
                            BatchSize::SmallInput,
                        );
                    },
                );
            }
        }
    }
}

fn randvec<T>(n: usize) -> Array1<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    (0..n)
        .map(|_| {
            let x: T = rand::rng().sample(StandardNormal);
            x
        })
        .collect()
}

fn randspace<T>(n: usize) -> Array1<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    let mut x = randvec::<T>(n);
    x.as_slice_mut()
        .unwrap()
        .sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    x
}

pub fn randts<T>(n: usize) -> TimeSeries<'static, T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    let t = randspace(n);
    let m = randvec(n);
    let w = randvec(n).mapv(|x: T| x.powi(2));
    TimeSeries::new(t, m, w)
}
