mod bounds;

#[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
pub mod ceres;
#[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
pub use ceres::CeresCurveFit;

#[cfg(any(
    feature = "gsl",
    any(feature = "ceres-source", feature = "ceres-system")
))]
mod constants;

pub mod curve_fit;
pub use curve_fit::{CurveFitAlgorithm, CurveFitResult, CurveFitTrait};

pub mod data;

pub mod evaluator;

#[cfg(feature = "gsl")]
pub mod lmsder;
#[cfg(feature = "gsl")]
pub use lmsder::LmsderCurveFit;

pub mod mcmc;
pub use mcmc::McmcCurveFit;

pub mod prior;
pub use prior::ln_prior::LnPrior;
pub use prior::ln_prior_1d::LnPrior1D;

#[cfg(test)]
pub trait HyperdualFloat: hyperdual::Float {
    fn half() -> Self;
    fn two() -> Self;
}
#[cfg(test)]
impl<T> HyperdualFloat for T
where
    T: hyperdual::Float,
{
    #[inline]
    fn half() -> Self {
        Self::from(0.5).unwrap()
    }

    #[inline]
    fn two() -> Self {
        Self::from(2.0).unwrap()
    }
}
#[cfg(not(test))]
pub trait HyperdualFloat: crate::Float {}
#[cfg(not(test))]
impl<T> HyperdualFloat for T where T: crate::Float {}

pub trait LikeFloat:
    HyperdualFloat + std::ops::AddAssign<Self> + std::ops::MulAssign<Self> + Sized
{
    fn logistic(x: Self) -> Self {
        (Self::one() + Self::exp(-x)).recip()
    }
}

impl<T> LikeFloat for T where
    T: HyperdualFloat + std::ops::AddAssign<Self> + std::ops::MulAssign<Self>
{
}
