#!/bin/bash
# GitHub Copilot Agent Environment Setup Script
# This script sets up the development environment for light-curve-feature
# Based on the CI workflows in .github/workflows/test.yml

set -euo pipefail

echo "Setting up GitHub Copilot agent environment for light-curve-feature..."

# Update package lists
echo "Updating package lists..."
sudo apt-get update -qq

# Install required Ubuntu packages
# These packages are needed for building and testing the project
echo "Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    libfftw3-dev \
    libgsl-dev \
    libfontconfig-dev

echo "System dependencies installed successfully."

# Install Rust toolchains
echo "Installing Rust toolchains..."

# Install rustup if not already installed
if ! command -v rustup &> /dev/null; then
    echo "Installing rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain none
    source "$HOME/.cargo/env"
fi

# Install stable toolchain
echo "Installing Rust stable toolchain..."
rustup toolchain install stable --profile minimal
rustup default stable

# Extract MSRV from Cargo.toml
MSRV=$(grep '^rust-version = ' Cargo.toml | grep -o '[0-9.]\+' || echo "")
if [ -n "$MSRV" ]; then
    echo "Installing Rust MSRV toolchain: $MSRV..."
    rustup toolchain install "$MSRV" --profile minimal
    echo "MSRV toolchain $MSRV installed successfully."
else
    echo "Warning: Could not extract MSRV from Cargo.toml"
fi

# Show installed toolchains
echo "Installed Rust toolchains:"
rustup toolchain list

# Initialize git submodules
echo "Initializing git submodules..."
if [ -f .gitmodules ]; then
    git submodule update --init --recursive
    echo "Git submodules initialized successfully."
else
    echo "No .gitmodules file found, skipping submodule initialization."
fi

# Install cargo components for stable toolchain
echo "Installing cargo components for stable toolchain..."
rustup component add rustfmt clippy

echo ""
echo "✓ GitHub Copilot agent environment setup complete!"
echo ""
echo "Available Rust toolchains:"
rustup toolchain list
echo ""
echo "You can now build and test the project with:"
echo "  cargo build --no-default-features --features=ceres-source,fftw-source,gsl"
echo "  cargo test --no-default-features --features=ceres-source,fftw-source,gsl"
