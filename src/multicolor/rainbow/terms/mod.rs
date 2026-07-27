//! Pluggable bolometric / temperature / spectral terms composed by [RainbowFit](super::RainbowFit).
//!
//! Each term contributes a small, fixed number of named parameters (see each module's
//! `params()`), a physical-unit initial guess (`initial_guess()`), and physical-unit box bounds
//! (`bounds()`). See [super::fit] for how a parameter's bounds drive its optimizer
//! reparametrization.

pub mod bolometric;
pub mod common;
pub mod spectral;
pub mod temperature;

pub use bolometric::Bolometric;
pub use spectral::Spectral;
pub use temperature::Temperature;

/// The largest number of parameters any single bolometric or temperature term has
/// (`Doublexp` / `DelayedSigmoid`, both 5). Used to size fixed-length stack buffers and avoid
/// heap allocation in the per-point model evaluation hot path.
pub(crate) const MAX_LOCAL_PARAMS: usize = 5;
