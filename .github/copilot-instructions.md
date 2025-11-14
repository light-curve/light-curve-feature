# Project Overview

`light-curve-feature` is a Rust library for extracting numerous light curve features used in astrophysics. It provides feature extraction from noisy time series data, specifically designed for astronomical observations. The library is part of the `light-curve` family and has Python bindings available separately.

Key components:
- **Feature extractors**: Implement various light curve features for astrophysical analysis
- **TimeSeries**: Core data structure for time, magnitude/flux, and weight arrays
- **FeatureExtractor**: Main interface for extracting multiple features at once
- **Meta-features**: Transform light curves before feature extraction (e.g., Bins)

## Coding Standards

### Rust Style
- Follow standard Rust formatting: `cargo fmt` must pass without changes
- Use `cargo clippy` for linting; all warnings must be addressed
- Use clippy's `#[allow]` directives as precisely as possible (specific to the item that needs it)
- Add code comments to explain non-obvious clippy suppressions
- Prefer explicit over implicit when clarity benefits outweigh brevity

### Code Organization
- New feature evaluators require changes in at least three files:
  1. New file in `src/features/` directory
  2. Public import in `src/features/mod.rs`
  3. New variant in `Feature` enum in `src/feature.rs`

### Safety and Correctness
- **No unsafe code**: This repository contains no `unsafe` code and aims to keep it that way
- Avoid `unwrap()` in library code; use proper error handling with `Result` types
- Prefer static guarantees over runtime checks when possible

## Testing Requirements

### Coverage
- All user-level code should be tested
- Add unit tests for non-trivial changes and PRs
- Test edge cases and error conditions
- Use property-based testing where appropriate

### Test Organization
- Unit tests go in the same file as the code being tested
- Integration tests go in the `test-util` directory
- Test data files go in the `test-data` directory

### Running Tests
```bash
# Standard test with default features
cargo test

# Test with system libraries (requires system dependencies)
cargo test --no-default-features --features ceres-system,fftw-system,gsl

# Run all tests with all feature combinations in CI
# (see .github/workflows/test.yml for complete test matrix)
```

## Dependencies and Features

### Cargo Features
- `default`: enables `fftw-source` feature only
- `ceres-system` / `ceres-source`: Enable Ceres Solver support for non-linear fitting
- `fftw-system` / `fftw-source` / `fftw-mkl`: Enable FFTW support for Fourier transforms
- `gsl`: Enable GNU Scientific Library support for non-linear fitting

### External Dependencies
When adding or modifying features that require external libraries:
- Document system requirements in README
- Ensure feature flags properly gate the dependency
- Test with both system and source variants where applicable

### Version Constraints
- Minimum Supported Rust Version (MSRV): Specified in `Cargo.toml` as `rust-version`
- All dependencies must be compatible with MSRV
- Use `cargo-msrv` to verify changes don't break MSRV

## Development Workflow

### Setting Up Development Environment
1. Install Rust via [rustup](https://rustup.rs)
2. Install system dependencies (see README for platform-specific commands)
3. Clone with submodules: `git clone --recursive`
4. Run initial checks: `cargo check` and `cargo test`

### Pre-commit Hooks
- Use [pre-commit](https://pre-commit.com) for automated checks
- Initialize with `pre-commit install` in the repository
- Pre-commit runs: trailing-whitespace, end-of-file-fixer, check-yaml, check-toml, and more
- Rust checks (fmt, cargo-check, clippy) are handled by GitHub Actions due to pre-commit.ci limitations

### Pull Request Workflow
1. Create feature branch from `master`
2. Make minimal, focused changes
3. Run `cargo fmt` before committing
4. Run `cargo clippy` and address all warnings
5. Add/update tests for your changes
6. Ensure all tests pass locally
7. Submit PR with clear description of changes

## Documentation Standards

### Code Documentation
- All public APIs must have rustdoc documentation
- Include examples in documentation where helpful
- Use `///` for item documentation, `//!` for module documentation
- Use code blocks with language tags: ` ```rust `

### Documentation Features
- Math rendering: Use KaTeX for mathematical expressions in docs
- The crate uses custom HTML header for KaTeX (see `katex-header.html`)

### README and CHANGELOG
- Keep README.md up to date with API changes
- Follow changelog format in CHANGELOG.md
- Document breaking changes clearly

## Performance Considerations

### Release Builds
- LTO is enabled in release profile for optimal performance
- Single codegen unit for maximum optimization
- Benchmarks available via `cargo bench` (requires patience, they take time)

### Numerical Precision
- Be mindful of floating-point precision in astronomical calculations
- Use appropriate numerical algorithms for light curve analysis
- Test numerical stability with edge cases

## Security Practices

- No hardcoded secrets or credentials
- Validate all external input (time series data, configuration)
- Use secure defaults for optional parameters
- Review any network access or file I/O carefully

## Error Handling

- Use custom error types (see `src/error.rs`)
- Implement `thiserror` for error definitions
- Provide meaningful error messages that help users debug issues
- Use `Result` types consistently in public APIs

## Common Patterns

### Feature Implementation
When implementing new features:
- Extend the `Evaluator` trait
- Add proper documentation with references to papers/algorithms
- Include sensible defaults via `Default` trait
- Implement `serde::Serialize` and `serde::Deserialize` for serialization
- Add schema support via `schemars` for JSON schema generation

### Working with TimeSeries
- Use `TimeSeries` struct for all time series data
- Remember that weights are interpreted as inverse squared observation errors
- Handle NaN and infinite values appropriately
- Sort data when required by the algorithm

## Citation

If implementing features based on research, cite the relevant papers in documentation. The project itself should be cited per the README instructions (Malanchev et al., 2021).
