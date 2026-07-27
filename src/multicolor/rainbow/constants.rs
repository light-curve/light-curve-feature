//! CGS physical constants (CODATA 2018) used by [super::terms::spectral]'s blackbody-family SEDs.

/// Planck constant, erg*s
pub const PLANCK_H: f64 = 6.62607004e-27;
/// Speed of light, cm/s
pub const SPEED_OF_LIGHT: f64 = 2.99792458e10;
/// Boltzmann constant, erg/K
pub const BOLTZMANN_K: f64 = 1.380649e-16;
/// Stefan-Boltzmann constant, erg/(cm^2 s K^4)
pub const SIGMA_SB: f64 = 5.6703744191844314e-05;
