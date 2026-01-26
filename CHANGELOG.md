# Changelog

All notable changes to `light-curve-feature` will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Implement `PartialEq` for `Feature` enum and all constituent types, enabling equality comparisons between feature
  instances https://github.com/light-curve/light-curve-feature/issues/228
- Add NUTS (No-U-Turn Sampler) as an alternative fitting algorithm via `NutsCurveFit`, providing gradient-based Hamiltonian Monte Carlo optimization, gated behind the `nuts` cargo feature https://github.com/light-curve/light-curve-feature/pull/245
- Add `LnPriorEvaluator` trait for evaluating ln(prior) without serialization requirements, enabling more flexible prior handling in curve fitting https://github.com/light-curve/light-curve-feature/pull/248
- Add gradient support to `LnPrior` traits, enabling NUTS sampler to use prior gradients for improved sampling efficiency https://github.com/light-curve/light-curve-feature/pull/250

### Changed

- **Breaking** Update `ndarray` from 0.16 to 0.17 and `ndarray-stats` from 0.6 to 0.7. Functions that accept array
  inputs now use `&ArrayRef1<T>` instead of `impl Into<ArrayView1<T>>` following ndarray 0.17 best
  practices https://github.com/light-curve/light-curve-feature/issues/239
- **Breaking** Float parameters in many places now use `ordered_float::NotNan` type, enabling `Eq` and `Hash` trait
  derivation and
  ensuring parameters cannot be
  NaN https://github.com/light-curve/light-curve-feature/issues/233 https://github.com/light-curve/light-curve-feature/pull/242
- **Breaking** `MixLnPrior1D::new` signature changed from accepting `Vec` to
  slice. https://github.com/light-curve/light-curve-feature/pull/242
- **Breaking** `CurveFitTrait::curve_fit` now accepts priors implementing `LnPriorEvaluator` trait instead of bare `Fn` closures, providing better type safety and extensibility https://github.com/light-curve/light-curve-feature/pull/248
- Bump `ceres-solver` from 0.4.0 to 0.5.0 https://github.com/light-curve/light-curve-feature/pull/246
- Refactored prior transformation to use trait-based `TransformedLnPrior` wrapper instead of closures, improving debuggability and avoiding function wrapping https://github.com/light-curve/light-curve-feature/pull/248

### Deprecated

--

### Removed

--

### Fixed

--

### Security

--

# [0.10.0] 2025 June 4

### Changed

- **Breaking** Periodogram `FreqGrid` is refactored, now Periodogram uses new
  `FreqGridStrategy` https://github.com/light-curve/light-curve-feature/pull/205 https://github.com/light-curve/light-curve-feature/pull/206
- `FreqGrid` now supports "fixed" grids, including an arbitrary sorted
  array https://github.com/light-curve/light-curve-feature/pull/207
- Bump dev dependency `criterion` from 0.5 to 0.6 https://github.com/light-curve/light-curve-feature/pull/210

# [0.9.0] 2025 May 22

### Changed

- **Build breaking**: bump minimum supported Rust version (MSRV) from 1.67 to
  1.85 https://github.com/light-curve/light-curve-feature/pull/200 https://github.com/light-curve/light-curve-feature/pull/203
- Also update Rust Edition to 2024 https://github.com/light-curve/light-curve-feature/pull/203
- Bump `itertools` from ^0.13 to ^0.14 https://github.com/light-curve/light-curve-feature/pull/190

# [0.8.1] 2025 January 7

### Changed

- Bump `ceres-solver` to 0.4.0 to solve build issues on macOS

# [0.8.0] 2024 December 3

### Added

- Expose `nl_fit::CurveFitResult`

### Changed

- **Breaking** Update ndarray from 0.15 to 0.16 https://github.com/light-curve/light-curve-feature/pull/175

# [0.7.1] 2024 April 25

### Added

- `Roms`: Robust median statistic feature, thanks @GaluTi for their first
  contribution! https://github.com/light-curve/light-curve-feature/issues/103 https://github.com/light-curve/light-curve-feature/pull/160

# [0.7.0] 2024 February 26

### Changed

- **Build breaking**: bump `ceres-solver` to 0.3.0, which requires C++ version of ceres-solver v2.2.0 for `ceres-system`
  Cargo feature

# [0.6.0] 2023 September 19

### Changed

- **Build breaking** Minimum supported Rust version (MSRV) 1.57 -> 1.67 (released
  2023-01-26) https://github.com/light-curve/light-curve-feature/pull/125 https://github.com/light-curve/light-curve-feature/pull/130
- New `LinexpFit`feature for parametric model fit. Thanks @erusseil for his first
  contribution. https://github.com/light-curve/light-curve-feature/pull/122
- Feature's `doc()` static method is `const` now https://github.com/light-curve/light-curve-feature/pull/110
- Bump `itertools` from ^0.10 to ^0.11 https://github.com/light-curve/light-curve-feature/pull/131
- Bump `fftw` from ^0.7 to ^0.8 https://github.com/light-curve/light-curve-feature/pull/130

# [0.5.5] 2023 March 30

### Added

- Feature transformations and `Transformed` feature. New `Transformer` enum and structs implementing new
  `TransformerTrait` are added to represent various useful transformations of features. `Transformed` feature is a
  wrapper around a feature and a transformer. It implements `FeatureEvaluator` trait and applies the transformer to the
  feature
  value. https://github.com/light-curve/light-curve-feature/issues/107 https://github.com/light-curve/light-curve-feature/pull/108

### Fixed

- `Bins::min_ts_length()` always returned 1. For this feature this value is always a lower limit, because binning could
  reduce number of observations to 1. However, now Bins updates this value according to underlying features to give a
  more reasonable lower limit. https://github.com/light-curve/light-curve-feature/pull/109

## [0.5.4] 2023 March 15

### Added

- `CeresCurveFit` and optional `ceres-source` and `ceres-system` Cargo features to use Ceres Solver for curve
  fitting https://github.com/light-curve/light-curve-feature/issues/92 https://github.com/light-curve/light-curve-feature/pull/98 https://github.com/light-curve/light-curve-feature/pull/100

## [0.5.3] 2023 February 16

### Added

- Re-export of `ndarray`

### Changed

- Use `enum_dispatch` 0.3.9 (updated from 0.3.7) crate to implement `LnPriorTrait` for
  `LnPrior` https://github.com/light-curve/light-curve-feature/pull/6

### Fixed

- Overflows in `DataSample`, `MedianBufferRangePercentage`, `BeyondNStd`, `FreqGrid` and
  `OtsuSplit` https://github.com/light-curve/light-curve-feature/issues/95 https://github.com/light-curve/light-curve-feature/pull/96
- `thread_local<1.1.7` because it requires Rust toolchain 1.59

## [0.5.2] 2022 November 10

### Fixed

- Fixed `OtsuSplit` threshold implementation https://github.com/light-curve/light-curve-feature/issues/72

## [0.5.1] 2022 November 1

### Added

- `OtsuSplit` feature evaluator https://github.com/light-curve/light-curve-feature/pull/58

## [0.5.0] 2022 June 14

### Changed

- **Breaking** MSRV 1.56 -> 1.57

## [0.4.6] 2022 June 14

### Fixed

- Remove profile section from Cargo.toml because it is not supported by our MSRV 1.56

## [0.4.5] 2022 June 10

### Added

- CI: build on Windows w/o GSL

### Fixed

- Fix `NaN` panic in MCMC https://github.com/light-curve/light-curve-feature/issues/51
- Make it possible to run tests, benchmarks and examples without `gsl` feature

## [0.4.4] 2022 June 3

### Fixed

- Overflow in `{Bazin,Villar}Fit` https://github.com/light-curve/light-curve-feature/issues/48

## [0.4.3] 2022 May 12

### Added

- `NyquistFreq` constructor static methods

### Fixed

- Make `FixedNyquistFreq` public

## [0.4.2] 2022 May 12

### Added

- `Cargo.toml` keywords and categories
- Add `Fixed(FixedNyquistFreq)` variant of `NyquistFreq` which defines a fixed Nyquist frequency to use for pariodogram

### Changed

- The project repository was split from other 'light-curve*' crates and moved
  into <https://gituhb.com/light-curve/light-curve-feature>
- `light-curve-common` is a dev-dependency now
- CI: split Github Actions workflow into pieces

### Removed

- Unused `dyn-clonable` dependency

## [0.4.1] 2021 Dec 15

### Changed

- `BazinLnPrior`, `BazinInitsBounds`, `VillarLnPrior`, `VillarINitisBounds` are public now

### Fixed

- Fixed amplitude prior of `VillarLnPrior::Hosseinzadeh2020`
- The example plotted wrong graphs for the Villar function

## [0.4.0] 2021 Dec 10

### Added

- `prelude` module to allow a user importing all traits
- `McmcCurveFit` uses new `LnPrior` objects which holds natural logarithm of priors for parameters. `BazinFit` and
  `VillarFit` requires this object to be specified
- `VillarFit` could use `VillarLnPrior::Hosseinzadeh2020` adopted from Hosseinzadeh et al 2020 paper (aka Superphot
  paper)
- `FeatureExtractor::from_features()` as a specified version of `new()` required less if not none type annotations

### Changed

- Rust edition 2021
- Minimum Rust version is 1.56
- `FeatureEvaluator` trait is split into three: `FeatureEvaluator`, `EvaluatorInfoTrait` and
  `FeatureNamesDescriptionsTrait`
- `BazinFit` and `VillarFit` are reimplemented using new traits, which all are included into `prelude`
- `VillarFit` now uses a different parameter set to fix issue with non-physical negative flux fits, relative plateau
  amplitude `nu` replaces plateau slope `beta`. It is a breaking change
- `BazinFit` and `VillarFit` name and description for `t_0` parameter are changed replacing "peak" to "reference",
  because this time moment does not correspond to the light-curve peak
- `BazinFit` and `VillarFit` have two new fields (and require two new argments in there `new` constructors): `ln_prior`
  and `inits_bounds`. The last one supports custom initial guess and boundaries for the optimization problem
- MCMC uses more diverse initial guesses which take into account boundary conditions

### Removed

- `features::antifeatures` submodule is removed and all its features moved to the parent `features` submodule

### Fixed

- Update `clap` to `3.0.0-rc.0`, it is used for the example executable only
- `EtaE` and `MaximumSlope` docs updated to highlight cadence dependency of these features

## [0.3.3] 2021 Oct 14

### Fixed

- Equation in `VillarFit` was different from the paper
- `VillarFit`'s and `BazinFit`'s amplitude, tau_rise, tau_fall and plateau duration (for `VillarFit`) are insured to be
  positive

## [0.3.2] 2021 Aug 30

### Changed

- Rust-specific docs for `BazinFit` and `VillarFit` are moved from struct docs to `::new()` docs

### Fixed

- Equation rendering in `VillarFit` HTML docs

## [0.3.1] 2021 Aug 16

### Changed

- `periodogram` module and `_PeriodogramPeaks` are hidden from the docs
- Update katex for <http://docs.rs> to 0.13.13

### Fixed

- Docs for `Extractor`, `FeatureEvaluator`, `AndersonDarlingNormal`, `Bins`, `Cusum`, `EtaE`, `InterPercentileRange`,
  `LinearFit`, `LinearTrend`, `Median`, `PercentAmplitude`, `Periodogram`, `ReducedChi2`, `VillarFit` and `BazinFit`

## [0.3.0] 2021 Aug 10

### Added

- This `CHANGELOG.md` file
- `Feature` enum containing all available features, it implements `FeatureEvaluator`
- (De)serialization with [`serde`](http://serde.rs) is implemented for all features
- JSON schema generation with [`schemars`](http://graham.cool/schemars/) is implemented for all features
- `TimeSeries` and `DataSample` use `ndarray::CowArray` to hold data, their constructors accept `ArrayBase` objects
- Static method `::doc()` for every feature, it returns language-agnostic feature evaluator description
- `examples` directory with an example which fits and plots some SN Ia light curves
- "private" sub-crate `light-curve-feature-test-util` with common tools for tests, benchmarks and examples

### Changed

- `FeatureExtractor`, `Bins` and `Periodogram` accepts `Feature` enum objects instead of `Box<FeatureEvaluator>`
- Periodogram-related `NyquistFreq` and `PeriodogramPower` are changed from traits to enums
- `TimeSeries::new` accepts non-optional weights, use `TimeSeries::new_without_weight` to initialize time series with
  unity weight array
- `BazinFit` is parameterized by a curve-fit algorithm, MCMC and GSL's LMSDER are available, but the last one requires
  non-default `gsl` Cargo feature. MCMC becomes the default algorithm, some wide boundary conditions are included
- Rename `BazinFit::get_names()[1]` from "bazin_fit_offset" to "bazin_fit_baseline"
- Add `VillarFit` feature for the Villar function [arXiv:1905.07422](http://arxiv.org/abs/1905.07422), see `BazinFit`
  above for technical details
- `LinearTrend` requires at least three observations and returns three values: slope, its error and standard deviation
  of noise (new)
- Publicly exported stuff

## [0.2.x]

—

## [0.1.x]

—
