use crate::evaluator::*;
use crate::nl_fit::{
    CurveFitAlgorithm, CurveFitResult, CurveFitTrait, LikeFloat, LnPrior, McmcCurveFit,
    data::NormalizedData, evaluator::*,
};

use conv::ConvUtil;

const NPARAMS: usize = 4;

macro_const! {
    const DOC: &str = r"
Linexp function fit

Four fit parameters and goodness of fit (reduced $\chi^2$) of the Linexp function developed for
core-collapsed supernovae:

$$
f(t) = A \frac{(t-t_0)}{\tau} \times \exp{\left(\frac{(t-t_0)}{\tau}\right)} + B.
$$

Note, that the Linexp function is developed to be used with fluxes, not magnitudes.

- Depends on: **time**, **flux**, **flux error**
- Minimum number of observations: **5**
- Number of features: **5**

";
}

#[doc = DOC!()]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, PartialEq)]
pub struct LinexpFit {
    algorithm: CurveFitAlgorithm,
    ln_prior: LinexpLnPrior,
    inits_bounds: LinexpInitsBounds,
}

impl LinexpFit {
    /// New [LinexpFit] instance
    ///
    /// `algorithm` specifies which optimization method is used, it is an instance of the
    /// [CurveFitAlgorithm], currently supported algorithms are [MCMC](McmcCurveFit),
    /// [LMSDER](crate::nl_fit::LmsderCurveFit) (a Levenberg–Marquard algorithm modification,
    /// requires `gsl` Cargo feature), and [Ceres](crate::nl_fit::CeresCurveFit) (trust-region
    /// algorithm, requires `ceres` Cargo feature).
    ///
    /// `ln_prior` is an instance of [LinexpLnPrior] and specifies the natural logarithm of the prior
    /// to use. Some curve-fit algorithms doesn't support this and ignores the prior
    pub fn new<BLP>(
        algorithm: CurveFitAlgorithm,
        ln_prior: BLP,
        inits_bounds: LinexpInitsBounds,
    ) -> Self
    where
        BLP: Into<LinexpLnPrior>,
    {
        Self {
            algorithm,
            ln_prior: ln_prior.into(),
            inits_bounds,
        }
    }

    /// Default [McmcCurveFit] for [LinexpFit]
    #[inline]
    pub fn default_algorithm() -> CurveFitAlgorithm {
        McmcCurveFit::new(
            McmcCurveFit::default_niterations(),
            McmcCurveFit::default_fine_tuning_algorithm(),
        )
        .into()
    }

    /// Default [LnPrior] for [LinexpFit]
    #[inline]
    pub fn default_ln_prior() -> LinexpLnPrior {
        LnPrior::none().into()
    }

    #[inline]
    pub fn default_inits_bounds() -> LinexpInitsBounds {
        LinexpInitsBounds::Default
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

impl Default for LinexpFit {
    fn default() -> Self {
        Self::new(
            Self::default_algorithm(),
            Self::default_ln_prior(),
            Self::default_inits_bounds(),
        )
    }
}

lazy_info!(
    LINEXP_FIT_INFO,
    LinexpFit,
    size: NPARAMS + 1,
    min_ts_length: NPARAMS + 1,
    t_required: true,
    m_required: true,
    w_required: true,
    sorting_required: true, // improve reproducibility
);

struct Params<'a, T> {
    internal: &'a [T; NPARAMS],
    external: [T; NPARAMS],
}

impl<T> Params<'_, T>
where
    T: LikeFloat,
{
    #[inline]
    fn a(&self) -> T {
        self.external[0]
    }

    #[inline]
    fn sgn_a(&self) -> T {
        self.internal[0].signum()
    }

    #[inline]
    fn t0(&self) -> T {
        self.external[1]
    }

    #[inline]
    fn tau(&self) -> T {
        self.external[2]
    }

    #[inline]
    fn sgn_tau(&self) -> T {
        self.internal[2].signum()
    }

    #[inline]
    fn b(&self) -> T {
        self.external[3]
    }
}

impl<T, U> FitModelTrait<T, U, NPARAMS> for LinexpFit
where
    T: Float + Into<U>,
    U: LikeFloat,
{
    fn model(t: T, param: &[U; NPARAMS]) -> U
    where
        T: Float + Into<U>,
        U: LikeFloat,
    {
        let t: U = t.into();
        let x = Params {
            internal: param,
            external: Self::internal_to_dimensionless(param),
        };
        let dt = (t - x.t0()) / x.tau();
        x.b() + x.a() * dt * U::exp(-dt)
    }
}

impl<T> FitFunctionTrait<T, NPARAMS> for LinexpFit where T: Float {}

impl<T> FitDerivalivesTrait<T, NPARAMS> for LinexpFit
where
    T: Float,
{
    fn derivatives(t: T, param: &[T; NPARAMS], jac: &mut [T; NPARAMS]) {
        let x = Params {
            internal: param,
            external: Self::internal_to_dimensionless(param),
        };
        let dt = (t - x.t0()) / x.tau();
        let exp = T::exp(-dt);

        // a
        jac[0] = x.sgn_a() * dt * exp;
        // t0
        jac[1] = x.a() * exp / x.tau() * (dt - T::one());
        // tau
        jac[2] = jac[1] * x.sgn_tau() * dt;
        // b
        jac[3] = T::one();
    }
}

impl<T> FitInitsBoundsTrait<T, NPARAMS> for LinexpFit
where
    T: Float,
{
    fn init_and_bounds_from_ts(&self, ts: &mut TimeSeries<T>) -> FitInitsBoundsArrays<NPARAMS> {
        match &self.inits_bounds {
            LinexpInitsBounds::Default => LinexpInitsBounds::default_from_ts(ts),
            LinexpInitsBounds::Arrays(arrays) => arrays.as_ref().clone(),
            LinexpInitsBounds::OptionArrays(opt_arr) => {
                opt_arr.unwrap_with(&LinexpInitsBounds::default_from_ts(ts))
            }
        }
    }
}

impl FitParametersOriginalDimLessTrait<NPARAMS> for LinexpFit {
    fn orig_to_dimensionless(
        norm_data: &NormalizedData<f64>,
        orig: &[f64; NPARAMS],
    ) -> [f64; NPARAMS] {
        [
            norm_data.m_to_norm_scale(orig[0]), // A amplitude
            norm_data.t_to_norm(orig[1]),       // t_0 reference_time
            norm_data.t_to_norm_scale(orig[2]), // tau fall time
            norm_data.m_to_norm(orig[3]),       // b baseline
        ]
    }

    fn dimensionless_to_orig(
        norm_data: &NormalizedData<f64>,
        norm: &[f64; NPARAMS],
    ) -> [f64; NPARAMS] {
        [
            norm_data.m_to_orig_scale(norm[0]), // A amplitude
            norm_data.t_to_orig(norm[1]),       // t_0 reference_time
            norm_data.t_to_orig_scale(norm[2]), // tau fall time
            norm_data.m_to_orig(norm[3]),       // b baseline
        ]
    }
}

impl<U> FitParametersInternalDimlessTrait<U, NPARAMS> for LinexpFit
where
    U: LikeFloat,
{
    fn dimensionless_to_internal(params: &[U; NPARAMS]) -> [U; NPARAMS] {
        *params
    }

    fn internal_to_dimensionless(params: &[U; NPARAMS]) -> [U; NPARAMS] {
        [params[0].abs(), params[1], params[2].abs(), params[3]]
    }
}

impl FitParametersInternalExternalTrait<NPARAMS> for LinexpFit {}

impl FitFeatureEvaluatorGettersTrait<NPARAMS> for LinexpFit {
    fn get_algorithm(&self) -> &CurveFitAlgorithm {
        &self.algorithm
    }

    fn ln_prior_from_ts<T: Float>(&self, ts: &mut TimeSeries<T>) -> LnPrior<NPARAMS> {
        self.ln_prior.ln_prior_from_ts(ts)
    }
}

impl FeatureNamesDescriptionsTrait for LinexpFit {
    fn get_names(&self) -> Vec<&str> {
        vec![
            "linexp_fit_amplitude",
            "linexp_fit_reference_time",
            "linexp_fit_fall_time",
            "linexp_fit_baseline",
            "linexp_fit_reduced_chi2",
        ]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![
            "Amplitude of the Linexp function (A)",
            "reference time of the Linexp fit (t0)",
            "fall time of the Linexp function (tau)",
            "baseline of the Linexp function (B)",
            "Linexp fit quality (reduced chi2)",
        ]
    }
}

impl<T> FeatureEvaluator<T> for LinexpFit
where
    T: Float,
{
    fit_eval!();
}

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema, Default, PartialEq)]
#[non_exhaustive]
pub enum LinexpInitsBounds {
    #[default]
    Default,
    Arrays(Box<FitInitsBoundsArrays<NPARAMS>>),
    OptionArrays(Box<OptionFitInitsBoundsArrays<NPARAMS>>),
}

impl LinexpInitsBounds {
    pub fn arrays(init: [f64; NPARAMS], lower: [f64; NPARAMS], upper: [f64; NPARAMS]) -> Self {
        Self::Arrays(FitInitsBoundsArrays::new(init, lower, upper).into())
    }

    pub fn option_arrays(
        init: [Option<f64>; NPARAMS],
        lower: [Option<f64>; NPARAMS],
        upper: [Option<f64>; NPARAMS],
    ) -> Self {
        Self::OptionArrays(OptionFitInitsBoundsArrays::new(init, lower, upper).into())
    }

    fn default_from_ts<T: Float>(ts: &mut TimeSeries<T>) -> FitInitsBoundsArrays<NPARAMS> {
        let t_min: f64 = ts.t.get_min().value_into().unwrap();
        let t_max: f64 = ts.t.get_max().value_into().unwrap();
        let t_amplitude = t_max - t_min;
        let t_peak: f64 = ts.get_t_max_m().value_into().unwrap();
        let m_min: f64 = ts.m.get_min().value_into().unwrap();
        let m_max: f64 = ts.m.get_max().value_into().unwrap();
        let m_amplitude = m_max - m_min;

        // Analytical solution is amplitude = Flux_amplitude * exp(1)
        let a_init = m_amplitude * 3.0;
        let (a_lower, a_upper) = (0.0, 100.0 * m_amplitude);

        let tau_init = t_amplitude * 0.25;
        let (tau_lower, tau_upper) = (0.0, t_amplitude * 10000.0);

        // - tau_init comes from analytical solution of the Linexp function peak
        // We multiply by 1.5 because from experience the minimizer behaves better
        // when the initial guess overshoots to the left compared to the right

        let t0_init = t_peak - 1.5 * tau_init;
        let (t0_lower, t0_upper) = (t_min - 10.0 * t_amplitude, t_max + 10.0 * t_amplitude);

        let b_init = m_min;
        let (b_lower, b_upper) = (m_min - 100.0 * m_amplitude, m_max + 100.0 * m_amplitude);

        FitInitsBoundsArrays {
            init: [a_init, t0_init, tau_init, b_init].into(),
            lower: [a_lower, t0_lower, tau_lower, b_lower].into(),
            upper: [a_upper, t0_upper, tau_upper, b_upper].into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
#[non_exhaustive]
pub enum LinexpLnPrior {
    Fixed(Box<LnPrior<NPARAMS>>),
}

impl LinexpLnPrior {
    pub fn fixed(ln_prior: LnPrior<NPARAMS>) -> Self {
        Self::Fixed(ln_prior.into())
    }

    pub fn ln_prior_from_ts<T: Float>(&self, _ts: &mut TimeSeries<T>) -> LnPrior<NPARAMS> {
        match self {
            Self::Fixed(ln_prior) => ln_prior.as_ref().clone(),
        }
    }
}

impl From<LnPrior<NPARAMS>> for LinexpLnPrior {
    fn from(item: LnPrior<NPARAMS>) -> Self {
        Self::fixed(item)
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
    use crate::CeresCurveFit;
    #[cfg(feature = "gsl")]
    use crate::LmsderCurveFit;
    use crate::TimeSeries;
    use crate::nl_fit::LnPrior1D;
    use crate::tests::*;

    use approx::assert_relative_eq;
    use hyperdual::Hyperdual;

    check_feature!(LinexpFit);

    check_fit_model_derivatives!(LinexpFit);

    feature_test!(
        linexp_fit_plateau,
        [LinexpFit::default()],
        [0.0, 6.25, 2.5, 0.0, 0.0], // initial model parameters and zero chi2
        linspace(0.0, 10.0, 11),
        [0.0; 11],
    );

    fn linexp_fit_noisy(eval: LinexpFit) {
        const N: usize = 50;

        let mut rng = StdRng::seed_from_u64(42);

        let param_true = [1000.0, -15.0, 20.0, 15.0];

        let t = linspace(-10.0, 100.0, N);
        let model: Vec<_> = t
            .iter()
            .map(|&x| LinexpFit::model(x, &param_true))
            .collect();
        let m: Vec<_> = model
            .iter()
            .map(|&y| {
                let std = f64::sqrt(y);
                let err = std * rng.sample::<f64, _>(StandardNormal);
                y + err
            })
            .collect();
        let w: Vec<_> = model.iter().copied().map(f64::recip).collect();
        println!("t = {t:?}\nmodel = {model:?}\nm = {m:?}\nw = {w:?}");
        let mut ts = TimeSeries::new(&t, &m, &w);

        // curve_fit(lambda t, a, t0, fall, b : b + a * ((t - t0) / fall) * np.exp(-(t - t0) / fall), xdata=t, ydata=m, sigma=np.array(w)**-0.5, p0=[800, 10, 15, 30])
        let desired = [986.62990444, -15.1956711, 20.05763093, 15.54839175];

        let values = eval.eval(&mut ts).unwrap();
        assert_relative_eq!(&values[..NPARAMS], &param_true[..], max_relative = 0.07);
        assert_relative_eq!(&values[..NPARAMS], &desired[..], max_relative = 0.04);
    }

    #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
    #[test]
    fn linexp_fit_noisy_ceres() {
        linexp_fit_noisy(LinexpFit::new(
            CeresCurveFit::default().into(),
            LnPrior::none(),
            LinexpInitsBounds::Default,
        ));
    }

    #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
    #[test]
    fn linexp_fit_noizy_mcmc_plus_ceres() {
        let ceres = CeresCurveFit::default();
        let mcmc = McmcCurveFit::new(512, Some(ceres.into()));
        linexp_fit_noisy(LinexpFit::new(
            mcmc.into(),
            LnPrior::none(),
            LinexpInitsBounds::Default,
        ));
    }

    #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
    #[test]
    fn linexp_fit_noizy_mcmc_plus_ceres_and_bounds() {
        let ceres = CeresCurveFit::default();
        let mcmc = McmcCurveFit::new(10, Some(ceres.into()));
        linexp_fit_noisy(LinexpFit::new(
            mcmc.into(),
            LnPrior::none(),
            LinexpInitsBounds::option_arrays(
                [None; 4],
                [None; 4],
                [None, Some(50.0), Some(50.0), None],
            ),
        ));
    }

    #[cfg(feature = "gsl")]
    #[test]
    fn linexp_fit_noisy_lmsder() {
        linexp_fit_noisy(LinexpFit::new(
            LmsderCurveFit::new(9).into(),
            LnPrior::none(),
            LinexpInitsBounds::Default,
        ));
    }

    #[cfg(feature = "gsl")]
    #[test]
    fn linexp_fit_noizy_mcmc_plus_lmsder() {
        let lmsder = LmsderCurveFit::new(1);
        let mcmc = McmcCurveFit::new(512, Some(lmsder.into()));
        linexp_fit_noisy(LinexpFit::new(
            mcmc.into(),
            LnPrior::none(),
            LinexpInitsBounds::Default,
        ));
    }

    #[test]
    fn linexp_fit_noizy_mcmc_with_prior() {
        let prior = LnPrior::ind_components([
            LnPrior1D::normal(900.0, 100.0),
            LnPrior1D::uniform(-50.0, 50.0),
            LnPrior1D::log_normal(f64::ln(20.0), 0.2),
            LnPrior1D::normal(15.0, 20.0),
        ]);
        let mcmc = McmcCurveFit::new(1024, None);
        linexp_fit_noisy(LinexpFit::new(
            mcmc.into(),
            prior,
            LinexpInitsBounds::Default,
        ));
    }

    /// https://github.com/light-curve/light-curve-feature/issues/138
    #[test]
    fn linexp_left_bound_larger_right_bound() {
        let mut ts = light_curve_feature_test_util::issue_light_curve_mag::<f64, _>(
            "light-curve-feature-138/1.csv",
        )
        .into_triple(None)
        .into();
        let linexp = LinexpFit::new(
            McmcCurveFit::new(1 << 14, None).into(),
            LnPrior::none(),
            LinexpInitsBounds::Default,
        );
        let _result = linexp.eval(&mut ts).unwrap();
    }

    #[cfg(feature = "nuts")]
    #[test]
    fn linexp_fit_noisy_nuts() {
        use crate::NutsCurveFit;
        const N: usize = 50;

        let mut rng = StdRng::seed_from_u64(42);

        let param_true = [1000.0, -15.0, 20.0, 15.0];

        let t = linspace(-10.0, 100.0, N);
        let model: Vec<_> = t
            .iter()
            .map(|&x| LinexpFit::model(x, &param_true))
            .collect();
        let m: Vec<_> = model
            .iter()
            .map(|&y| {
                let std = f64::sqrt(y);
                let err = std * rng.sample::<f64, _>(StandardNormal);
                y + err
            })
            .collect();
        let w: Vec<_> = model.iter().copied().map(f64::recip).collect();
        let mut ts = TimeSeries::new(&t, &m, &w);

        let eval = LinexpFit::new(
            NutsCurveFit::new(200, 200, None).into(),
            LnPrior::none(),
            LinexpInitsBounds::Default,
        );
        let values = eval.eval(&mut ts).unwrap();
        // NUTS is stochastic and may need more relaxed tolerance
        assert_relative_eq!(&values[..NPARAMS], &param_true[..], max_relative = 0.08);
    }

    #[cfg(all(feature = "nuts", any(feature = "ceres-source", feature = "ceres-system")))]
    #[test]
    fn linexp_fit_noisy_nuts_plus_ceres() {
        use crate::NutsCurveFit;
        let ceres = CeresCurveFit::default();
        let nuts = NutsCurveFit::new(50, 50, Some(ceres.into()));
        linexp_fit_noisy(LinexpFit::new(
            nuts.into(),
            LnPrior::none(),
            LinexpInitsBounds::Default,
        ));
    }
}
