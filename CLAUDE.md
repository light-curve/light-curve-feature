# CLAUDE.md - AI Assistant Guide for light-curve-feature

This document provides comprehensive guidance for AI assistants working with the `light-curve-feature` codebase.

## Project Overview

`light-curve-feature` is a Rust library for extracting numerous light curve features used in astrophysics. It processes noisy time series data from astronomical observations to compute statistical and model-fit features used for anomaly detection and classification of variable stars and transient objects.

**Repository**: https://github.com/light-curve/light-curve-feature
**Current Version**: 0.10.0
**Rust Edition**: 2024
**MSRV**: 1.85
**License**: GPL-3.0-or-later

Python bindings are available separately at https://github.com/light-curve/light-curve-python

## Codebase Structure

```
light-curve-feature/
├── src/
│   ├── lib.rs                 # Module organization and re-exports
│   ├── feature.rs             # Main Feature enum (all feature variants)
│   ├── evaluator.rs           # Core FeatureEvaluator trait definitions
│   ├── extractor.rs           # FeatureExtractor for bulk extraction
│   ├── time_series.rs         # TimeSeries and DataSample structures
│   ├── error.rs               # EvaluatorError enum
│   ├── float_trait.rs         # Generic Float trait (f32/f64)
│   ├── prelude.rs             # Convenience re-exports
│   ├── macros.rs              # Proc-macro helpers
│   ├── types.rs               # Type aliases (CowArray1)
│   ├── sorted_array.rs        # Cached sorted array wrapper
│   ├── straight_line_fit.rs   # Linear regression utilities
│   ├── lnerfc.rs              # Math helpers (ln complementary error function)
│   ├── peak_indices.rs        # Peak finding utilities
│   ├── tests.rs               # Shared test macros
│   │
│   ├── features/              # Individual feature implementations (38+ files)
│   │   ├── mod.rs             # Feature module exports
│   │   ├── amplitude.rs       # Half amplitude feature
│   │   ├── bazin_fit.rs       # Bazin function curve fit
│   │   ├── periodogram.rs     # Periodogram-based features
│   │   ├── bins.rs            # Meta-feature for binning
│   │   └── ...                # Other statistical/model features
│   │
│   ├── periodogram/           # FFT-based periodogram (7 modules)
│   │   ├── mod.rs             # Main periodogram module
│   │   ├── fft.rs             # FFT wrapper using FFTW
│   │   ├── freq.rs            # FreqGridStrategy implementations
│   │   ├── power_fft.rs       # FFT-based power calculation
│   │   ├── power_direct.rs    # Direct power calculation
│   │   ├── power_trait.rs     # Trait definitions
│   │   └── sin_cos_iterator.rs
│   │
│   ├── nl_fit/                # Non-linear curve fitting (9+ modules)
│   │   ├── mod.rs             # Module organization
│   │   ├── curve_fit.rs       # CurveFitTrait definitions
│   │   ├── evaluator.rs       # Fit feature evaluators
│   │   ├── data.rs            # Normalized fitting data
│   │   ├── bounds.rs          # Parameter bounds handling
│   │   ├── mcmc.rs            # MCMC fitting algorithm
│   │   ├── ceres.rs           # Ceres Solver (optional)
│   │   ├── lmsder.rs          # LMSDER/GSL (optional)
│   │   ├── nuts.rs            # NUTS sampler (optional)
│   │   └── prior/             # Prior probability distributions
│   │
│   └── transformers/          # Feature transformation pipeline (7 modules)
│       ├── mod.rs
│       ├── transformer.rs     # Core transformer trait
│       ├── clipped_lg.rs      # Clipped logarithm
│       ├── composed.rs        # Transformer composition
│       └── *_fit.rs           # Fit-specific transformers
│
├── test-util/                 # Test utilities sub-crate
│   ├── Cargo.toml
│   └── src/                   # Test data loading, CSV parsing
│
├── test-data/                 # Git submodule with test datasets
├── benches/lib.rs             # Criterion benchmarks
├── examples/                  # Example programs
├── Cargo.toml                 # Main package configuration
├── CHANGELOG.md               # Version history
└── README.md                  # Project documentation
```

## Key Architectural Concepts

### Core Data Types

- **`TimeSeries<T>`**: Main data structure holding time, magnitude/flux, and weight arrays
- **`DataSample<T>`**: Wrapper providing cached statistics (min, max, mean, median, etc.)
- **`Feature<T>`**: Enum aggregating all feature implementations via `enum_dispatch`
- **`FeatureExtractor<T, F>`**: Bulk feature extraction from multiple features

### Core Traits

- **`FeatureEvaluator<T>`**: Main trait for feature implementations
- **`EvaluatorInfoTrait`**: Metadata about features (min_ts_length, is_t_required, etc.)
- **`FeatureNamesDescriptionsTrait`**: Human-readable descriptions
- **`CurveFitTrait`**: Non-linear fitting algorithms
- **`LnPrior`/`LnPriorEvaluator`**: Prior probability distributions for Bayesian fitting
- **`TransformerTrait`**: Feature value transformations

### Design Patterns

1. **Generic over floats**: All features use `T: Float` for f32/f64 support
2. **Enum dispatch**: Uses `enum_dispatch` for efficient runtime polymorphism
3. **Lazy evaluation**: Sorted arrays and statistics computed on-demand with caching
4. **Macro metaprogramming**: `lazy_info!()`, `transformer_eval!()`, `fit_eval!()` macros

## Cargo Features

```toml
# Default: fftw-source only
default = ["fftw-source"]

# FFTW options (mutually exclusive preference: mkl > source > system)
fftw-source    # Build FFTW from source (default)
fftw-system    # Use system FFTW installation
fftw-mkl       # Use Intel MKL

# Non-linear fitting solvers
ceres-source   # Build Ceres Solver from source
ceres-system   # Use system Ceres installation
gsl            # GNU Scientific Library for LMSDER algorithm
nuts           # NUTS sampler for Bayesian fitting

# Common feature combinations for development:
# --features ceres-source,fftw-source,gsl     # Full features, source builds
# --features ceres-system,fftw-system,gsl     # Full features, system libs
```

## Development Setup

### System Dependencies

```bash
# macOS
brew install ceres-solver cmake fftw gsl fontconfig

# Debian/Ubuntu
sudo apt-get install build-essential cmake libceres-dev libfftw3-dev libgsl-dev libfontconfig-dev
```

### Clone and Build

```bash
git clone --recursive https://github.com/light-curve/light-curve-feature
cd light-curve-feature

# Build with all features
cargo build --no-default-features --features ceres-source,fftw-source,gsl

# Run tests
cargo test --no-default-features --features ceres-source,fftw-source,gsl
```

Note: On ARM macOS, Ceres may require `CPATH=/opt/homebrew/include`

## Code Quality Requirements

### Formatting and Linting

```bash
# Format check (must pass with no changes)
cargo fmt -- --check

# Clippy (all warnings are errors)
cargo clippy --all-targets --no-default-features --features=ceres-source,fftw-source,gsl -- -D warnings

# Also check test-util crate
cd test-util && cargo fmt -- --check && cargo clippy --all-targets -- -D warnings && cd ..
```

### Rust Style Guidelines

- Follow standard Rust formatting via `cargo fmt`
- Address all clippy warnings
- Use clippy `#[allow]` directives as precisely as possible (specific to the item)
- Add comments explaining non-obvious clippy suppressions
- Prefer explicit over implicit when clarity benefits outweigh brevity

### Safety Requirements

- **No unsafe code**: This codebase has zero unsafe code and should stay that way
- Avoid `unwrap()` in library code - use proper `Result` error handling
- Prefer static guarantees over runtime checks
- Do not introduce security vulnerabilities (command injection, etc.)

## Implementing New Features

When adding a new feature evaluator, modify these files:

1. **Create** `src/features/your_feature.rs` - Implement the feature
2. **Export** in `src/features/mod.rs` - Add public import
3. **Register** in `src/feature.rs` - Add variant to `Feature` enum

### Feature Implementation Checklist

- [ ] Implement `FeatureEvaluator<T>` trait
- [ ] Use `lazy_info!()` macro for metadata (via `EvaluatorInfoTrait`)
- [ ] Document with LaTeX formulas for mathematical expressions
- [ ] Specify data requirements (`is_t_required`, `is_m_required`, `is_w_required`)
- [ ] Set `min_ts_length` (minimum observations needed)
- [ ] Implement `Default` trait with sensible defaults
- [ ] Implement `serde::Serialize` and `serde::Deserialize`
- [ ] Add `schemars` JSON schema support
- [ ] Write unit tests using `feature_test!()` macro
- [ ] Document references to papers/algorithms

## Testing

### Running Tests

```bash
# Standard tests with default features
cargo test

# Full feature tests
cargo test --no-default-features --features ceres-source,fftw-source,gsl

# Release mode tests (catches optimization-related issues)
cargo test --profile=release-with-debug --no-default-features --features ceres-source,fftw-source,gsl
```

### Test Organization

- Unit tests in same file as implementation (`#[cfg(test)]` modules)
- Shared test utilities in `src/tests.rs` and `test-util/` crate
- Test data in `test-data/` submodule (pre-loaded datasets)
- Use `feature_test!()` and `check_feature!()` macros

### Test Data Datasets

- `RRLYR_LIGHT_CURVES_MAG_F64` - RR Lyrae variable stars
- `SNIA_LIGHT_CURVES_FLUX_F64` - Type Ia supernovae
- `ISSUE_LIGHT_CURVES_*` - Regression test cases from GitHub issues

## Error Handling

- Use `EvaluatorError` enum from `src/error.rs`
- Use `thiserror` for error type derivation
- Return `Result` types consistently in public APIs
- Provide meaningful error messages

## Documentation Standards

- All public APIs require rustdoc documentation
- Use `///` for item docs, `//!` for module docs
- Mathematical formulas use LaTeX via KaTeX (see `katex-header.html`)
- Include code examples where helpful
- Cite relevant papers for feature implementations

## Updating the Changelog

The project maintains a `CHANGELOG.md` following the [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format and [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

### When to Update

Update the changelog for any user-facing changes:
- New features or feature evaluators
- Bug fixes
- Breaking API changes
- Dependency updates (especially breaking ones)
- Performance improvements
- Documentation improvements (if significant)

### Changelog Structure

```markdown
## [Unreleased]

### Added
- New features go here

### Changed
- Changes to existing functionality

### Deprecated
- Features marked for removal

### Removed
- Removed features

### Fixed
- Bug fixes

### Security
- Security-related fixes
```

### Entry Format Guidelines

1. **Add entries to `[Unreleased]` section** - Never add to released versions
2. **Use the correct category** based on change type
3. **Start with a capital letter**, no period at end
4. **Include GitHub links** at the end of each entry:
   ```markdown
   - Add `NewFeature` for computing X https://github.com/light-curve/light-curve-feature/pull/123
   ```
5. **Mark breaking changes** with `**Breaking**` prefix:
   ```markdown
   - **Breaking** Rename `old_method` to `new_method` https://github.com/light-curve/light-curve-feature/pull/456
   ```
6. **Mark build-breaking changes** (MSRV, dependency requirements) with `**Build breaking**`:
   ```markdown
   - **Build breaking**: bump minimum supported Rust version (MSRV) from 1.67 to 1.85
   ```
7. **Use empty placeholder** `--` for unused sections in unreleased
8. **Wrap long entries** with proper indentation:
   ```markdown
   - Add NUTS (No-U-Turn Sampler) as an alternative fitting algorithm via `NutsCurveFit`, providing
     gradient-based Hamiltonian Monte Carlo optimization, gated behind the `nuts` cargo
     feature https://github.com/light-curve/light-curve-feature/pull/245
   ```
9. **Credit contributors** for first-time contributions:
   ```markdown
   - `Roms`: Robust median statistic feature, thanks @GaluTi for their first
     contribution! https://github.com/light-curve/light-curve-feature/pull/160
   ```

### Examples by Change Type

**Adding a new feature:**
```markdown
### Added
- Add `MyNewFeature` evaluator for computing statistical measure X https://github.com/light-curve/light-curve-feature/pull/XXX
```

**Breaking API change:**
```markdown
### Changed
- **Breaking** `SomeStruct::new` signature changed from accepting `Vec` to slice https://github.com/light-curve/light-curve-feature/pull/XXX
```

**Dependency bump:**
```markdown
### Changed
- Bump `some-crate` from 1.0 to 2.0 https://github.com/light-curve/light-curve-feature/pull/XXX
```

**Bug fix:**
```markdown
### Fixed
- Fix overflow in `DataSample` when processing large arrays https://github.com/light-curve/light-curve-feature/issues/XXX https://github.com/light-curve/light-curve-feature/pull/YYY
```

**New cargo feature:**
```markdown
### Added
- Add `new-feature` cargo feature enabling optional functionality https://github.com/light-curve/light-curve-feature/pull/XXX
```

### Version Release Format

When a version is released (maintainers only), the format changes to:
```markdown
# [0.10.0] 2025 June 4

### Changed
- ...
```

Note: Only include sections that have entries (no empty `--` placeholders in releases).

## CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/test.yml`) runs:

| Job | Purpose |
|-----|---------|
| `build` | Multiple feature combinations on Ubuntu |
| `msrv-build` | Minimum Supported Rust Version (1.85) |
| `test` | Tests on Ubuntu x64 and ARM |
| `test-release` | Release mode with debug symbols |
| `examples` | Run example programs |
| `fmt` | Formatting check |
| `clippy` | Linting with warnings-as-errors |
| `fmt-test-util` | Format check for test-util |
| `clippy-test-util` | Clippy for test-util |
| `coverage` | LLVM coverage with Codecov |
| `macos` | macOS build |
| `windows` | Windows build (no GSL) |

## Performance Considerations

- LTO enabled in release profile for optimization
- Single codegen unit for maximum optimization
- Benchmarks available: `cargo bench --no-default-features --features ceres-source,fftw-source,gsl`
- Be mindful of floating-point precision in astronomical calculations
- Test numerical stability with edge cases

## Common Commands Quick Reference

```bash
# Development
cargo check --all-targets --no-default-features --features=ceres-source,fftw-source,gsl
cargo test --no-default-features --features=ceres-source,fftw-source,gsl
cargo fmt && cargo clippy --all-targets --no-default-features --features=ceres-source,fftw-source,gsl -- -D warnings

# Benchmarks
cargo bench --no-default-features --features ceres-source,fftw-source,gsl

# Documentation
cargo doc --no-default-features --features ceres-source,fftw-source,gsl --open

# Run example
cargo run --example plot_snia_curve_fits --no-default-features --features=ceres-source,fftw-source,gsl -- -n=1
```

## Important Notes for AI Assistants

1. **Read before editing**: Always read existing code before suggesting modifications
2. **Minimal changes**: Make focused, minimal changes - avoid over-engineering
3. **No unsafe code**: Never introduce unsafe blocks
4. **Test coverage**: Add tests for non-trivial changes
5. **Feature flags**: Gate optional dependencies behind cargo features
6. **Weights interpretation**: Weights in `TimeSeries` are inverse squared observation errors
7. **Generic floats**: Features must work with both f32 and f64
8. **Pre-commit**: Run `pre-commit run --all-files` before committing
9. **Submodules**: Clone with `--recursive` for test data

## Citation

If implementing features based on research, cite relevant papers. The project itself should be cited per README instructions (Malanchev et al., 2021, MNRAS 502, 5147).
