use enum_dispatch::enum_dispatch;
use ordered_float::NotNan;
use schemars::JsonSchema;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::hash::Hash;

#[enum_dispatch]
pub trait LnPrior1DTrait:
    Clone + Debug + Serialize + DeserializeOwned + PartialEq + Eq + Hash
{
    /// Evaluate the natural logarithm of the prior at x
    ///
    /// If `grad` is `Some`, the gradient d(ln_prior)/dx is also computed and stored in it.
    fn ln_prior_1d(&self, x: f64, grad: Option<&mut f64>) -> f64;
}

/// Natural logarithm of prior for a single parameter of the curve-fit problem
#[enum_dispatch(LnPrior1DTrait)]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum LnPrior1D {
    None(NoneLnPrior1D),
    LogNormal(LogNormalLnPrior1D),
    LogUniform(LogUniformLnPrior1D),
    Normal(NormalLnPrior1D),
    Uniform(UniformLnPrior1D),
    Mix(MixLnPrior1D),
}

impl LnPrior1D {
    pub fn none() -> Self {
        NoneLnPrior1D {}.into()
    }

    pub fn log_normal(mu: f64, std: f64) -> Self {
        LogNormalLnPrior1D::new(mu, std).into()
    }

    pub fn log_uniform(left: f64, right: f64) -> Self {
        LogUniformLnPrior1D::new(left, right).into()
    }

    pub fn normal(mu: f64, std: f64) -> Self {
        NormalLnPrior1D::new(mu, std).into()
    }

    pub fn uniform(left: f64, right: f64) -> Self {
        UniformLnPrior1D::new(left, right).into()
    }

    pub fn mix(weight_prior_pairs: &[(f64, LnPrior1D)]) -> Self {
        MixLnPrior1D::new(weight_prior_pairs).into()
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub struct NoneLnPrior1D {}

impl LnPrior1DTrait for NoneLnPrior1D {
    fn ln_prior_1d(&self, _x: f64, grad: Option<&mut f64>) -> f64 {
        if let Some(g) = grad {
            *g = 0.0;
        }
        0.0
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(
    into = "LogNormalLnPrior1DParameters",
    from = "LogNormalLnPrior1DParameters"
)]
pub struct LogNormalLnPrior1D {
    mu: NotNan<f64>,
    inv_std2: NotNan<f64>,
    ln_prob_coeff: NotNan<f64>,
}

impl LogNormalLnPrior1D {
    pub fn new(mu: f64, std: f64) -> Self {
        Self {
            mu: NotNan::new(mu).expect("mu must be not NaN"),
            inv_std2: NotNan::new(std.powi(-2)).expect("std must be positive and finite"),
            ln_prob_coeff: NotNan::new(-f64::ln(std) - 0.5 * f64::ln(std::f64::consts::TAU))
                .expect("std must be positive and finite"),
        }
    }

    fn mu(&self) -> f64 {
        self.mu.into_inner()
    }

    fn inv_std2(&self) -> f64 {
        self.inv_std2.into_inner()
    }

    fn ln_prob_coeff(&self) -> f64 {
        self.ln_prob_coeff.into_inner()
    }
}

impl LnPrior1DTrait for LogNormalLnPrior1D {
    fn ln_prior_1d(&self, x: f64, grad: Option<&mut f64>) -> f64 {
        let ln_x = f64::ln(x);
        let diff = self.mu() - ln_x;
        let ln_prior = self.ln_prob_coeff() - 0.5 * diff.powi(2) * self.inv_std2() - ln_x;

        if let Some(g) = grad {
            *g = (diff * self.inv_std2() - 1.0) / x;
        }

        ln_prior
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "NormalLnPrior1D")]
struct LogNormalLnPrior1DParameters {
    mu: f64,
    std: f64,
}

impl From<LogNormalLnPrior1D> for LogNormalLnPrior1DParameters {
    fn from(f: LogNormalLnPrior1D) -> Self {
        Self {
            mu: f.mu(),
            std: f.inv_std2().recip().sqrt(),
        }
    }
}

impl From<LogNormalLnPrior1DParameters> for LogNormalLnPrior1D {
    fn from(f: LogNormalLnPrior1DParameters) -> Self {
        Self::new(f.mu, f.std)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(
    into = "LogUniformLnPrior1DParameters",
    from = "LogUniformLnPrior1DParameters"
)]
pub struct LogUniformLnPrior1D {
    ln_range: std::ops::RangeInclusive<NotNan<f64>>,
    ln_prob_coeff: NotNan<f64>,
}

impl LogUniformLnPrior1D {
    pub fn new(left: f64, right: f64) -> Self {
        assert!(left < right);
        let ln_left = NotNan::new(f64::ln(left)).expect("left must be positive and finite");
        let ln_right = NotNan::new(f64::ln(right)).expect("right must be positive and finite");
        Self {
            ln_range: ln_left..=ln_right,
            ln_prob_coeff: NotNan::new(-f64::ln(ln_right.into_inner() - ln_left.into_inner()))
                .expect("right must be larger than left"),
        }
    }

    fn ln_left(&self) -> f64 {
        self.ln_range.start().into_inner()
    }

    fn ln_right(&self) -> f64 {
        self.ln_range.end().into_inner()
    }

    fn ln_prob_coeff(&self) -> f64 {
        self.ln_prob_coeff.into_inner()
    }
}

impl LnPrior1DTrait for LogUniformLnPrior1D {
    fn ln_prior_1d(&self, x: f64, grad: Option<&mut f64>) -> f64 {
        let ln_x = if let Ok(ln_x) = NotNan::new(f64::ln(x)) {
            ln_x
        } else {
            if let Some(g) = grad {
                *g = 0.0;
            }
            return f64::NEG_INFINITY;
        };
        if self.ln_range.contains(&ln_x) {
            if let Some(g) = grad {
                *g = -1.0 / x;
            }
            self.ln_prob_coeff() - ln_x.into_inner()
        } else {
            if let Some(g) = grad {
                *g = 0.0;
            }
            f64::NEG_INFINITY
        }
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "LogUniformLnPrior")]
struct LogUniformLnPrior1DParameters {
    ln_range: std::ops::RangeInclusive<f64>,
}

impl From<LogUniformLnPrior1D> for LogUniformLnPrior1DParameters {
    fn from(f: LogUniformLnPrior1D) -> Self {
        Self {
            ln_range: f.ln_left()..=f.ln_right(),
        }
    }
}

impl From<LogUniformLnPrior1DParameters> for LogUniformLnPrior1D {
    fn from(f: LogUniformLnPrior1DParameters) -> Self {
        Self::new(f.ln_range.start().exp(), f.ln_range.end().exp())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(into = "NormalLnPrior1DParameters", from = "NormalLnPrior1DParameters")]
pub struct NormalLnPrior1D {
    mu: NotNan<f64>,
    inv_std2: NotNan<f64>,
    ln_prob_coeff: NotNan<f64>,
}

impl NormalLnPrior1D {
    pub fn new(mu: f64, std: f64) -> Self {
        Self {
            mu: NotNan::new(mu).expect("mu must be not NaN"),
            inv_std2: NotNan::new(std.powi(-2)).expect("std must be positive and finite"),
            ln_prob_coeff: NotNan::new(-f64::ln(std) - 0.5 * f64::ln(std::f64::consts::TAU))
                .expect("std must be positive and finite"),
        }
    }

    fn mu(&self) -> f64 {
        self.mu.into_inner()
    }

    fn inv_std2(&self) -> f64 {
        self.inv_std2.into_inner()
    }

    fn ln_prob_coeff(&self) -> f64 {
        self.ln_prob_coeff.into_inner()
    }
}

impl LnPrior1DTrait for NormalLnPrior1D {
    fn ln_prior_1d(&self, x: f64, grad: Option<&mut f64>) -> f64 {
        let diff = self.mu() - x;
        let ln_prior = self.ln_prob_coeff() - 0.5 * diff.powi(2) * self.inv_std2();

        if let Some(g) = grad {
            *g = diff * self.inv_std2();
        }

        ln_prior
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "NormalLnPrior1D")]
struct NormalLnPrior1DParameters {
    mu: f64,
    std: f64,
}

impl From<NormalLnPrior1D> for NormalLnPrior1DParameters {
    fn from(f: NormalLnPrior1D) -> Self {
        Self {
            mu: f.mu(),
            std: f.inv_std2().recip().sqrt(),
        }
    }
}

impl From<NormalLnPrior1DParameters> for NormalLnPrior1D {
    fn from(f: NormalLnPrior1DParameters) -> Self {
        Self::new(f.mu, f.std)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
#[serde(
    into = "UniformLnPrior1DParameters",
    from = "UniformLnPrior1DParameters"
)]
pub struct UniformLnPrior1D {
    range: std::ops::RangeInclusive<NotNan<f64>>,
    ln_prob: NotNan<f64>,
}

impl UniformLnPrior1D {
    pub fn new(left: f64, right: f64) -> Self {
        let left = NotNan::new(left).expect("left must be finite");
        let right = NotNan::new(right).expect("right must be finite");
        Self {
            range: left..=right,
            ln_prob: NotNan::new(-f64::ln(right.into_inner() - left.into_inner()))
                .expect("right must be larger than left"),
        }
    }

    fn left(&self) -> f64 {
        self.range.start().into_inner()
    }

    fn right(&self) -> f64 {
        self.range.end().into_inner()
    }

    fn ln_prob(&self) -> f64 {
        self.ln_prob.into_inner()
    }
}

impl LnPrior1DTrait for UniformLnPrior1D {
    fn ln_prior_1d(&self, x: f64, grad: Option<&mut f64>) -> f64 {
        let x = if let Ok(x) = NotNan::new(x) {
            x
        } else {
            if let Some(g) = grad {
                *g = 0.0;
            }
            return f64::NEG_INFINITY;
        };
        if self.range.contains(&x) {
            if let Some(g) = grad {
                // Uniform prior: ln_prior is constant within bounds
                // So gradient is 0
                *g = 0.0;
            }
            self.ln_prob()
        } else {
            if let Some(g) = grad {
                *g = 0.0;
            }
            f64::NEG_INFINITY
        }
    }
}

#[derive(Serialize, Deserialize, JsonSchema)]
#[serde(rename = "UniformLnPrior")]
struct UniformLnPrior1DParameters {
    range: std::ops::RangeInclusive<f64>,
}

impl From<UniformLnPrior1D> for UniformLnPrior1DParameters {
    fn from(f: UniformLnPrior1D) -> Self {
        Self {
            range: f.left()..=f.right(),
        }
    }
}

impl From<UniformLnPrior1DParameters> for UniformLnPrior1D {
    fn from(f: UniformLnPrior1DParameters) -> Self {
        Self::new(*f.range.start(), *f.range.end())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub struct MixLnPrior1D {
    mix: Vec<(NotNan<f64>, LnPrior1D)>,
}

impl MixLnPrior1D {
    /// Create MixLnPrior1D from pairs of a weight (positive number) and an instance of `LnPrior1D`
    pub fn new(weight_prior_pairs: &[(f64, LnPrior1D)]) -> Self {
        let total_weight: f64 = weight_prior_pairs.iter().map(|(weight, _)| *weight).sum();
        let weight_prior_pairs = weight_prior_pairs
            .iter()
            .map(|(weight, prior)| {
                assert!(*weight > 0.0, "weights must be positive and finite");
                (
                    NotNan::new(*weight / total_weight)
                        .expect("weights must be positive and finite"),
                    prior.clone(),
                )
            })
            .collect::<Vec<(NotNan<f64>, LnPrior1D)>>();
        Self {
            mix: weight_prior_pairs,
        }
    }
}

impl LnPrior1DTrait for MixLnPrior1D {
    fn ln_prior_1d(&self, x: f64, grad: Option<&mut f64>) -> f64 {
        let mut total_prob = 0.0;
        let mut total_grad_weighted = 0.0;

        for (weight, prior) in &self.mix {
            let mut component_grad = 0.0;
            let ln_prob_i = prior.ln_prior_1d(
                x,
                if grad.is_some() {
                    Some(&mut component_grad)
                } else {
                    None
                },
            );
            let prob_i = f64::exp(ln_prob_i);
            let weighted_prob = weight.into_inner() * prob_i;
            total_prob += weighted_prob;

            if grad.is_some() {
                total_grad_weighted += weighted_prob * component_grad;
            }
        }

        if let Some(g) = grad {
            *g = total_grad_weighted / total_prob;
        }

        f64::ln(total_prob)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use hyperdual::Hyperdual;

    // Generic helper functions for computing ln_prior with any float type
    fn ln_prior_none<T: num_traits::Float>(_x: T) -> T {
        T::zero()
    }

    fn ln_prior_uniform<T: num_traits::Float>(x: T, left: f64, right: f64) -> T {
        if x >= T::from(left).unwrap() && x <= T::from(right).unwrap() {
            -T::from(f64::ln(right - left)).unwrap()
        } else {
            T::neg_infinity()
        }
    }

    fn ln_prior_normal<T: num_traits::Float>(x: T, mu: f64, std: f64) -> T {
        let mu_t = T::from(mu).unwrap();
        let std_t = T::from(std).unwrap();
        let diff = mu_t - x;
        let inv_std2 = T::one() / (std_t * std_t);
        let ln_prob_coeff =
            -T::ln(std_t) - T::from(0.5).unwrap() * T::ln(T::from(std::f64::consts::TAU).unwrap());
        ln_prob_coeff - T::from(0.5).unwrap() * diff * diff * inv_std2
    }

    fn ln_prior_log_normal<T: num_traits::Float>(x: T, mu: f64, std: f64) -> T {
        let ln_x = T::ln(x);
        let mu_t = T::from(mu).unwrap();
        let std_t = T::from(std).unwrap();
        let diff = mu_t - ln_x;
        let inv_std2 = T::one() / (std_t * std_t);
        let ln_prob_coeff =
            -T::ln(std_t) - T::from(0.5).unwrap() * T::ln(T::from(std::f64::consts::TAU).unwrap());
        ln_prob_coeff - T::from(0.5).unwrap() * diff * diff * inv_std2 - ln_x
    }

    fn ln_prior_log_uniform<T: num_traits::Float>(x: T, left: f64, right: f64) -> T {
        let ln_x = T::ln(x);
        let ln_left = T::from(f64::ln(left)).unwrap();
        let ln_right = T::from(f64::ln(right)).unwrap();
        if ln_x >= ln_left && ln_x <= ln_right {
            -T::ln(ln_right - ln_left) - ln_x
        } else {
            T::neg_infinity()
        }
    }

    // Test gradient using dual numbers
    fn test_prior_gradient<F, P>(f: F, test_values: &[f64], prior: P)
    where
        F: Fn(Hyperdual<f64, 2>) -> Hyperdual<f64, 2>,
        P: LnPrior1DTrait,
    {
        for &x in test_values {
            let mut actual_grad = 0.0;
            let actual_ln_p = prior.ln_prior_1d(x, Some(&mut actual_grad));

            // Skip if the result is not finite
            if !actual_ln_p.is_finite() {
                continue;
            }

            // Compute gradient using dual numbers
            let mut x_dual = Hyperdual::<f64, 2>::from_real(x);
            x_dual[1] = 1.0; // Set epsilon for first derivative
            let ln_p_dual = f(x_dual);
            let expected_grad = ln_p_dual[1];

            assert_relative_eq!(actual_grad, expected_grad, epsilon = 1e-9);
        }
    }

    #[test]
    fn test_none_gradient() {
        let prior = NoneLnPrior1D {};
        test_prior_gradient(ln_prior_none, &[0.0, 1.0, 5.0, 10.0], prior);
    }

    #[test]
    fn test_uniform_gradient() {
        let prior = UniformLnPrior1D::new(0.0, 10.0);
        test_prior_gradient(|x| ln_prior_uniform(x, 0.0, 10.0), &[1.0, 5.0, 9.0], prior);
    }

    #[test]
    fn test_normal_gradient() {
        let prior = NormalLnPrior1D::new(5.0, 2.0);
        test_prior_gradient(
            |x| ln_prior_normal(x, 5.0, 2.0),
            &[0.0, 3.0, 5.0, 7.0, 10.0],
            prior,
        );
    }

    #[test]
    fn test_log_normal_gradient() {
        let prior = LogNormalLnPrior1D::new(1.0, 0.5);
        test_prior_gradient(
            |x| ln_prior_log_normal(x, 1.0, 0.5),
            &[0.5, 1.0, 2.0, 3.0, 5.0],
            prior,
        );
    }

    #[test]
    fn test_log_uniform_gradient() {
        let prior = LogUniformLnPrior1D::new(1.0, 10.0);
        test_prior_gradient(
            |x| ln_prior_log_uniform(x, 1.0, 10.0),
            &[1.5, 3.0, 5.0, 7.0, 9.0],
            prior,
        );
    }

    #[test]
    fn test_mix_gradient() {
        let mix = MixLnPrior1D::new(&[
            (0.5, LnPrior1D::normal(5.0, 1.0)),
            (0.5, LnPrior1D::normal(10.0, 1.0)),
        ]);

        // For mix, use numerical differentiation since it's complex
        for x in &[3.0, 5.0, 7.0, 9.0, 11.0] {
            let mut grad = 0.0;
            let ln_p = mix.ln_prior_1d(*x, Some(&mut grad));

            if !ln_p.is_finite() {
                continue;
            }

            // Numerical gradient
            let eps = 1e-8;
            let ln_p_plus = mix.ln_prior_1d(*x + eps, None);
            let ln_p_minus = mix.ln_prior_1d(*x - eps, None);
            let numerical_grad = (ln_p_plus - ln_p_minus) / (2.0 * eps);

            assert_relative_eq!(grad, numerical_grad, epsilon = 1e-6);
        }
    }
}
