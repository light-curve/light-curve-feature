# `light-curve-feature`

`light-curve-feature` is a part of [`light-curve`](https://github.com/light-curve) family that
implements extraction of numerous light curve features used in astrophysics.

If you are looking for Python bindings for this package, please see <https://github.com/light-curve/light-curve-python>

[![docs.rs badge](https://docs.rs/light-curve-feature/badge.svg)](https://docs.rs/light-curve-feature)
![testing](https://github.com/light-curve/light-curve-feature/actions/workflows/test.yml/badge.svg)
[![pre-commit.ci status](https://results.pre-commit.ci/badge/github/light-curve/light-curve-feature/master.svg)](https://results.pre-commit.ci/latest/github/light-curve/light-curve-feature/master)

All features are available in [Feature](crate::Feature) enum, and the recommended way to extract multiple features at
once is [FeatureExtractor](crate::FeatureExtractor) struct built from a `Vec<Feature>`. Data is represented by
[TimeSeries](crate::TimeSeries) struct built from time, magnitude (or flux) and weight arrays, all having the same length. Note
that multiple features interpret weight array as inversed squared observation errors.

```rust
use light_curve_feature::prelude::*;

// Let's find amplitude and reduced Chi-squared of the light curve
let fe = FeatureExtractor::from_features(vec![
    Amplitude::new().into(),
    ReducedChi2::new().into()
]);
// Define light curve
let time = [0.0, 1.0, 2.0, 3.0, 4.0];
let magn = [-1.0, 2.0, 1.0, 3.0, 4.5];
let weights = [5.0, 10.0, 2.0, 10.0, 5.0]; // inverse squared magnitude errors
let mut ts = TimeSeries::new(&time, &magn, &weights);
// Get results and print
let result = fe.eval(&mut ts)?;
let names = fe.get_names();
println!("{:?}", names.iter().zip(result.iter()).collect::<Vec<_>>());
# Ok::<(), EvaluatorError>(())
```

There are a couple of meta-features, which transform a light curve before feature extraction. For example
[Bins](crate::Bins) feature accumulates data inside time-windows and extracts features from this new light curve.

```rust
use light_curve_feature::prelude::*;
use ndarray::Array1;

// Define features, "raw" MaximumSlope and binned with zero offset and 1-day window
let max_slope: Feature<_> = MaximumSlope::default().into();
let bins: Feature<_> = {
    let mut bins = Bins::new(1.0, 0.0);
    bins.add_feature(max_slope.clone());
    bins.into()
};
let fe = FeatureExtractor::from_features(vec![max_slope, bins]);
// Define light curve
let time = [0.1, 0.2, 1.1, 2.1, 2.1];
let magn = [10.0, 10.1, 10.5, 11.0, 10.9];
// We don't need weight for MaximumSlope, this would assign unity weight
let mut ts = TimeSeries::new_without_weight(&time, &magn);
// Get results and print
let result = fe.eval(&mut ts)?;
println!("{:?}", result);
# Ok::<(), EvaluatorError>(())
```

### Cargo features

The crate is configured with the following Cargo features:
- `ceres-system` and `ceres-source` - enable [Ceres Solver](http://ceres-solver.org) support for non-linear fitting. The former
  uses system-wide installation of Ceres, the latter builds Ceres from source and links it statically. The latter overrides the former. See [`ceres-solver-rs` crate](https://github.com/light-curve/ceres-solver-rs) for details
- `fftw-system`, `fftw-source` (enabled by default) and `fftw-mkl` - enable [FFTW](http://www.fftw.org) support for Fourier transforms needed by `Periodogram`. The
  first uses system-wide installation of FFTW, the second builds FFTW from source and links it statically, the last downloads and links statically Intel MKL instead of FFTW.
- `gsl` - enables [GNU Scientific Library](https://www.gnu.org/software/gsl/) support for non-linear fitting.
- `default` - enables `fftw-source` feature only, has no side effects.

### Development

**Setting up**

Install Rust toolchain, the preferred way is [rustup](https://rustup.rs).

Clone the repository recursively and run tests with default features:
```bash
git clone --recursive https://github.com/light-curve/light-curve-feature
cd light-curve-feature
cargo test
```

Install the required system libraries (Ceres Solver, FFTW, GSL)
```bash
# On macOS:
brew install ceres-solver fftw gsl
# On Debian-like:
apt install libceres-dev libfftw3-dev libgsl-dev
```

Run tests with these native libraries.
Note that Ceres could require manual `CPATH` specification, like `CPATH=/opt/homebrew/include`:
```bash
cargo test --no-default-features --features ceres-system,fftw-system,gsl
```

You may also run benchmarks, but be patient
```bash
cargo bench --no-default-features --features ceres-system,fftw-system,gsl
```

See `examples`, `.github/workflows` and tests for examples of the code usage.

**Formatting and linting**

We format and check the code with the standard Rust tools: `cargo fmt` and `cargo clippy`.
Please use clippy's `#[allow]` as precise as possible and leave code comments if it is not obvious why its usage is required.

We use [pre-commit](https://pre-commit.com) for running some linters locally before commiting.
Please consider installing it and initializing in the repo with `pre-commit init`.
However pre-commit.ci and GitHub Actions will varify `cargo fmt` and `cargo clippy` for PRs.

Generally, we are aimed to test all user-level code, add unit-tests to your non-trivial PRs.
Currently we have no `unsafe` code in this repo and we are aimed to avoid it in the future.

**Implementing a new feature evaluator**

Your new feature evaluator code should go to at least three files:

1. New file inside `src/features` directory
2. Publically import the new struct inside `src/features/mod.rs`
3. Add it as a new variant of `Feature` enum inside `src/feature.rs`

### Citation

If you found this project useful for your research please cite [Malanchev et al., 2021](https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M/abstract)

```bibtex
@ARTICLE{2021MNRAS.502.5147M,
       author = {{Malanchev}, K.~L. and {Pruzhinskaya}, M.~V. and {Korolev}, V.~S. and {Aleo}, P.~D. and {Kornilov}, M.~V. and {Ishida}, E.~E.~O. and {Krushinsky}, V.~V. and {Mondon}, F. and {Sreejith}, S. and {Volnova}, A.~A. and {Belinski}, A.~A. and {Dodin}, A.~V. and {Tatarnikov}, A.~M. and {Zheltoukhov}, S.~G. and {(The SNAD Team)}},
        title = "{Anomaly detection in the Zwicky Transient Facility DR3}",
      journal = {\mnras},
     keywords = {methods: data analysis, astronomical data bases: miscellaneous, stars: variables: general, Astrophysics - Instrumentation and Methods for Astrophysics, Astrophysics - Solar and Stellar Astrophysics},
         year = 2021,
        month = apr,
       volume = {502},
       number = {4},
        pages = {5147-5175},
          doi = {10.1093/mnras/stab316},
archivePrefix = {arXiv},
       eprint = {2012.01419},
 primaryClass = {astro-ph.IM},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2021MNRAS.502.5147M},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```
