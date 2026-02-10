use criterion::Criterion;
use light_curve_common::linspace;
use light_curve_feature::TimeSeries;
use light_curve_feature::periodogram::fft_trait::{FftInputArray, FftOutputArray};
use light_curve_feature::periodogram::{
    AverageNyquistFreq, DefaultPeriodogramPowerFft, FreqGridStrategy, MedianNyquistFreq,
    Periodogram, PeriodogramNormalization, PeriodogramPower, PeriodogramPowerDirect, RustFft,
};
#[cfg(feature = "fftw")]
use light_curve_feature::periodogram::{FftwFft, PeriodogramPowerFft};
use std::hint::black_box;

pub fn bench_periodogram(c: &mut Criterion) {
    let ns_power_resolution: [(Vec<usize>, PeriodogramPower<f32>, f32); 2] = [
        (vec![10, 100, 1000], PeriodogramPowerDirect.into(), 5.0),
        (
            vec![10, 100, 1000, 10000, 1000000],
            DefaultPeriodogramPowerFft::new().into(),
            10.0,
        ),
    ];
    const PERIOD: f32 = 0.22;
    let nyquist = AverageNyquistFreq;

    for (ns, power, resolution) in ns_power_resolution.iter() {
        let freq_grid_strategy = FreqGridStrategy::dynamic(*resolution, 1.0, nyquist);
        for &n in ns {
            let x = linspace(0.0_f32, 1.0, n);
            let y: Vec<_> = x
                .iter()
                .map(|&x| 3.0 * f32::sin(2.0 * std::f32::consts::PI / PERIOD * x + 0.5) + 4.0)
                .collect();
            c.bench_function(
                format!("Periodogram: {n} length, {power:?}").as_str(),
                |b| {
                    b.iter(|| {
                        let mut ts = TimeSeries::new_without_weight(&x, &y);
                        let periodogram = Periodogram::from_t(
                            power.clone(),
                            &x,
                            &freq_grid_strategy,
                            PeriodogramNormalization::default(),
                        )
                        .unwrap();
                        periodogram.power(black_box(&mut ts));
                    })
                },
            );
        }
    }
}

/// Benchmark comparing RustFFT vs FFTW backends for periodogram computation
pub fn bench_periodogram_fft_backends(c: &mut Criterion) {
    use rand::prelude::*;

    const PERIOD: f64 = 0.22;
    let nyquist = MedianNyquistFreq;

    // Test various sizes - powers of 2 work best for FFT
    let sizes: Vec<usize> = vec![64, 256, 1024, 4096, 16384, 65536];

    for &n in &sizes {
        // Generate non-regular (irregular) time grid - more realistic for astronomical observations
        let mut rng = StdRng::seed_from_u64(n as u64);
        let mut x: Vec<f64> = (0..n).map(|_| rng.random::<f64>()).collect();
        x.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let y: Vec<_> = x
            .iter()
            .map(|&x| 3.0 * f64::sin(2.0 * std::f64::consts::PI / PERIOD * x + 0.5) + 4.0)
            .collect();

        let freq_grid_strategy = FreqGridStrategy::dynamic(10.0, 1.0, nyquist);

        // Benchmark RustFFT backend (default Fft variant, always available)
        {
            let power: PeriodogramPower<f64> = DefaultPeriodogramPowerFft::new().into();
            c.bench_function(
                format!("Periodogram FFT backend: RustFFT, n={n}").as_str(),
                |b| {
                    b.iter(|| {
                        let mut ts = TimeSeries::new_without_weight(&x, &y);
                        let periodogram = Periodogram::from_t(
                            power.clone(),
                            &x,
                            &freq_grid_strategy,
                            PeriodogramNormalization::default(),
                        )
                        .unwrap();
                        periodogram.power(black_box(&mut ts));
                    })
                },
            );
        }

        // Benchmark FFTW backend (only when available)
        #[cfg(feature = "fftw")]
        {
            let power = PeriodogramPower::<f64>::FftFftw(PeriodogramPowerFft::new());
            c.bench_function(
                format!("Periodogram FFT backend: FFTW, n={n}").as_str(),
                |b| {
                    b.iter(|| {
                        let mut ts = TimeSeries::new_without_weight(&x, &y);
                        let periodogram = Periodogram::from_t(
                            power.clone(),
                            &x,
                            &freq_grid_strategy,
                            PeriodogramNormalization::default(),
                        )
                        .unwrap();
                        periodogram.power(black_box(&mut ts));
                    })
                },
            );
        }
    }
}

/// Benchmark the raw FFT operation (without periodogram overhead)
pub fn bench_raw_fft_backends(c: &mut Criterion) {
    use light_curve_feature::periodogram::Fft;

    // Test various sizes
    let sizes: Vec<usize> = vec![256, 1024, 4096, 16384, 65536, 262144];

    for &n in &sizes {
        // Generate random input data
        let input: Vec<f64> = (0..n).map(|i| (i as f64 * 0.1).sin()).collect();

        // Benchmark RustFFT
        {
            c.bench_function(format!("Raw FFT: RustFFT, n={n}").as_str(), |b| {
                let mut fft: RustFft<f64> = Fft::new();
                let mut x = <RustFft<f64> as Fft<f64>>::InputArray::new_with_size(n);
                let mut y = <RustFft<f64> as Fft<f64>>::OutputArray::new_with_size(n / 2 + 1);

                b.iter(|| {
                    // Copy input data
                    x.as_mut().copy_from_slice(&input);
                    fft.fft(black_box(&mut x), black_box(&mut y));
                })
            });
        }

        // Benchmark FFTW (only when available)
        #[cfg(feature = "fftw")]
        {
            c.bench_function(format!("Raw FFT: FFTW, n={n}").as_str(), |b| {
                let mut fft: FftwFft<f64> = Fft::new();
                let mut x = <FftwFft<f64> as Fft<f64>>::InputArray::new_with_size(n);
                let mut y = <FftwFft<f64> as Fft<f64>>::OutputArray::new_with_size(n / 2 + 1);

                b.iter(|| {
                    // Copy input data
                    x.as_mut().copy_from_slice(&input);
                    fft.fft(black_box(&mut x), black_box(&mut y));
                })
            });
        }
    }
}
