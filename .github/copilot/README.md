# GitHub Copilot Agent Environment

This directory contains the setup script for GitHub Copilot's coding agent environment.

## Setup Script

The `agent-setup.sh` script configures the development environment for `light-curve-feature` by:

1. **Installing Ubuntu packages**: All system dependencies required for building and testing
   - `build-essential` - C/C++ compiler and build tools
   - `cmake` - Build system generator
   - `libfftw3-dev` - Fast Fourier Transform library
   - `libgsl-dev` - GNU Scientific Library
   - `libfontconfig-dev` - Font configuration library (for examples)

2. **Setting up Rust toolchains**:
   - Installs rustup if not present
   - Installs stable toolchain and sets it as default
   - Installs MSRV (Minimum Supported Rust Version) toolchain from `Cargo.toml`
   - Installs rustfmt and clippy components

3. **Initializing git submodules**:
   - Recursively initializes the `test-data` submodule

## Usage

The script is automatically executed by GitHub Copilot when setting up the agent environment.

To manually run the setup:

```bash
./.github/copilot/agent-setup.sh
```

## Verification

After running the setup, you can verify the environment:

```bash
# Check installed Rust toolchains
rustup toolchain list

# Check submodule status
git submodule status

# Build the project
cargo build --no-default-features --features=ceres-source,fftw-source,gsl

# Run tests
cargo test --no-default-features --features=ceres-source,fftw-source,gsl
```

## Based On

This setup is based on the CI workflows in `.github/workflows/test.yml`, ensuring consistency between the agent environment and continuous integration.
