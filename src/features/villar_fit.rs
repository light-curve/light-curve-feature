use crate::evaluator::*;
use crate::nl_fit::{
    data::NormalizedData, evaluator::*, CurveFitAlgorithm, CurveFitResult, CurveFitTrait,
    LikeFloat, LnPrior, LnPrior1D, McmcCurveFit,
};

use conv::ConvUtil;

const NPARAMS: usize = 7;

macro_const! {
    const DOC: &str = r#"
Villar function fit

Seven fit parameters and goodness of fit (reduced $\chi^2$) of the Villar function developed for
supernovae classification:

<span>
$$
f(t) = c + \frac{A}{ 1 + \exp{\frac{-(t - t_0)}{\tau_\mathrm{rise}}}}  \left\{ \begin{array}{ll} 1 - \frac{\nu (t - t_0)}{\gamma}, &t < t_0 + \gamma \\ (1 - \nu) \exp{\frac{-(t-t_0-\gamma)}{\tau_\mathrm{fall}}}, &t \geq t_0 + \gamma \end{array} \right.
$$
</span>
where $A, \gamma, \tau_\mathrm{rise}, \tau_\mathrm{fall} > 0$, $\nu \in [0; 1)$.

Here we introduce a new dimensionless parameter $\nu$ instead of the plateau slope $\beta$ from the
orioginal paper: $\nu \equiv -\beta \gamma / A$.

Note, that the Villar function is developed to be used with fluxes, not magnitudes.

- Depends on: **time**, **magnitude**, **magnitude error**
- Minimum number of observations: **8**
- Number of features: **8**

Villar et al. 2019 [DOI:10.3847/1538-4357/ab418c](https://doi.org/10.3847/1538-4357/ab418c)
"#;
}

#[doc = DOC!()]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
pub struct VillarFit {
    algorithm: CurveFitAlgorithm,
    ln_prior: VillarLnPrior,
    inits_bounds: VillarInitsBounds,
}

impl VillarFit {
    /// New [VillarFit] instance
    ///
    /// `algorithm` specifies which optimization method is used, it is an instance of the
    /// [CurveFitAlgorithm], currently supported algorithms are [MCMC](McmcCurveFit) and
    /// [LMSDER](crate::nl_fit::LmsderCurveFit) (a Levenberg–Marquard algorithm modification,
    /// requires `gsl` Cargo feature).
    ///
    /// `ln_prior` is an instance of [LnPrior] and specifies the natural logarithm of the prior to
    /// use. Some curve-fit algorithms doesn't support this and ignores the prior
    pub fn new<VLP>(
        algorithm: CurveFitAlgorithm,
        ln_prior: VLP,
        inits_bounds: VillarInitsBounds,
    ) -> Self
    where
        VLP: Into<VillarLnPrior>,
    {
        Self {
            algorithm,
            ln_prior: ln_prior.into(),
            inits_bounds,
        }
    }

    /// Default [McmcCurveFit] for [VillarFit]
    #[inline]
    pub fn default_algorithm() -> CurveFitAlgorithm {
        McmcCurveFit::new(
            McmcCurveFit::default_niterations(),
            McmcCurveFit::default_fine_tuning_algorithm(),
        )
        .into()
    }

    /// Default [VillarLnPrior] for [VillarFit]
    #[inline]
    pub fn default_ln_prior() -> VillarLnPrior {
        LnPrior::none().into()
    }

    #[inline]
    pub fn default_inits_bounds() -> VillarInitsBounds {
        VillarInitsBounds::Default
    }

    pub fn doc() -> &'static str {
        DOC
    }

    fn nu_to_b<U>(nu: U) -> U
    where
        U: LikeFloat,
    {
        U::half() * (U::ln_1p(nu) - U::ln(U::one() - nu))
    }

    fn b_to_nu<U>(b: U) -> U
    where
        U: LikeFloat,
    {
        U::two() * U::logistic(U::two() * b.abs()) - U::one()
    }
}

impl Default for VillarFit {
    fn default() -> Self {
        Self::new(
            Self::default_algorithm(),
            Self::default_ln_prior(),
            Self::default_inits_bounds(),
        )
    }
}

lazy_info!(
    VILLAR_FIT_INFO,
    VillarFit,
    size: NPARAMS + 1,
    min_ts_length: NPARAMS + 1,
    t_required: true,
    m_required: true,
    w_required: true,
    sorting_required: true, // improve reproducibility
);

impl<T, U> FitModelTrait<T, U, NPARAMS> for VillarFit
where
    T: Float + Into<U>,
    U: LikeFloat,
{
    fn model(t: T, param: &[U; NPARAMS]) -> U {
        let t: U = t.into();
        let x = Params {
            internal: param,
            external: Self::internal_to_dimensionless(param),
        };
        x.c() + x.a() * x.rise(t) * x.plateau(t) * x.fall(t)
    }
}

impl<T> FitFunctionTrait<T, NPARAMS> for VillarFit where T: Float {}

impl<T> FitDerivalivesTrait<T, NPARAMS> for VillarFit
where
    T: Float,
{
    fn derivatives(t: T, param: &[T; NPARAMS], jac: &mut [T; NPARAMS]) {
        let x = Params {
            internal: param,
            external: Self::internal_to_dimensionless(param),
        };
        let dt = x.dt(t);
        let t1 = x.t1();
        let nu = x.nu();
        let plateau = x.plateau(t);
        let rise = x.rise(t);
        let fall = x.fall(t);
        let is_rise = t <= t1;
        let f_minus_c = x.a() * plateau * rise * fall;

        // A
        jac[0] = x.sgn_a_internal() * plateau * rise * fall;
        // c
        jac[1] = T::one();
        // t0
        jac[2] = x.a()
            * rise
            * fall
            * (-(T::one() - rise) * plateau / x.tau_rise()
                + if is_rise {
                    nu / x.gamma()
                } else {
                    plateau / x.tau_fall()
                });
        // tau_rise
        jac[3] =
            -x.sgn_tau_rise_internal() * f_minus_c * (T::one() - rise) * dt / x.tau_rise().powi(2);
        // tau_fall
        jac[4] = if is_rise {
            T::zero()
        } else {
            x.sgn_tau_fall_internal() * f_minus_c * (dt - x.gamma()) / x.tau_fall().powi(2)
        };
        // b = 1/2 * ln [(1 - nu) / (1 + nu)]
        jac[5] = -x.sgn_b()
            * (T::one() - nu.powi(2))
            * x.a()
            * rise
            * fall
            * if is_rise { dt / x.gamma() } else { T::one() };
        // gamma
        jac[6] = x.sgn_gamma_internal()
            * if is_rise {
                x.a() * rise * fall * nu * dt / x.gamma().powi(2)
            } else {
                f_minus_c / x.tau_fall()
            };
    }
}

impl<T> FitInitsBoundsTrait<T, NPARAMS> for VillarFit
where
    T: Float,
{
    fn init_and_bounds_from_ts(&self, ts: &mut TimeSeries<T>) -> FitInitsBoundsArrays<NPARAMS> {
        match &self.inits_bounds {
            VillarInitsBounds::Default => VillarInitsBounds::default_from_ts(ts),
            VillarInitsBounds::Arrays(arrays) => arrays.as_ref().clone(),
            VillarInitsBounds::OptionArrays(opt_arr) => {
                opt_arr.unwrap_with(&VillarInitsBounds::default_from_ts(ts))
            }
        }
    }
}

impl FitParametersOriginalDimLessTrait<NPARAMS> for VillarFit {
    fn orig_to_dimensionless(
        norm_data: &NormalizedData<f64>,
        orig: &[f64; NPARAMS],
    ) -> [f64; NPARAMS] {
        [
            norm_data.m_to_norm_scale(orig[0]), // A amplitude
            norm_data.m_to_norm(orig[1]),       // c baseline
            norm_data.t_to_norm(orig[2]),       // t_0 reference time
            norm_data.t_to_norm_scale(orig[3]), // tau_rise rise time
            norm_data.t_to_norm_scale(orig[4]), // tau_fall fall time
            orig[5],                            // nu = (beta gamma / A), dimensionless
            norm_data.t_to_norm_scale(orig[6]), // gamma plateau duration
        ]
    }

    fn dimensionless_to_orig(
        norm_data: &NormalizedData<f64>,
        norm: &[f64; NPARAMS],
    ) -> [f64; NPARAMS] {
        [
            norm_data.m_to_orig_scale(norm[0]), // A amplitude
            norm_data.m_to_orig(norm[1]),       // c baseline
            norm_data.t_to_orig(norm[2]),       // t_0 reference time
            norm_data.t_to_orig_scale(norm[3]), // tau_rise rise time
            norm_data.t_to_orig_scale(norm[4]), // tau_fall fall time
            norm[5], // nu = (beta gamma / A) relative plateau amplitude, dimensionless
            norm_data.t_to_orig_scale(norm[6]), // gamma plateau duration
        ]
    }
}

impl<U> FitParametersInternalDimlessTrait<U, NPARAMS> for VillarFit
where
    U: LikeFloat,
{
    fn dimensionless_to_internal(params: &[U; NPARAMS]) -> [U; NPARAMS] {
        [
            params[0],
            params[1],
            params[2],
            params[3],
            params[4],
            Self::nu_to_b(params[5]),
            params[6],
        ]
    }

    fn internal_to_dimensionless(params: &[U; NPARAMS]) -> [U; NPARAMS] {
        [
            params[0].abs(),
            params[1],
            params[2],
            params[3].abs(),
            params[4].abs(),
            Self::b_to_nu(params[5]),
            params[6].abs(),
        ]
    }
}

impl FitParametersInternalExternalTrait<NPARAMS> for VillarFit {}

impl FitFeatureEvaluatorGettersTrait<NPARAMS> for VillarFit {
    fn get_algorithm(&self) -> &CurveFitAlgorithm {
        &self.algorithm
    }

    fn ln_prior_from_ts<T: Float>(&self, ts: &mut TimeSeries<T>) -> LnPrior<NPARAMS> {
        self.ln_prior.ln_prior_from_ts(ts)
    }
}

impl FeatureNamesDescriptionsTrait for VillarFit {
    fn get_names(&self) -> Vec<&str> {
        vec![
            "villar_fit_amplitude",
            "villar_fit_baseline",
            "villar_fit_reference_time",
            "villar_fit_rise_time",
            "villar_fit_fall_time",
            "villar_fit_plateau_rel_amplitude",
            "villar_fit_plateau_duration",
            "villar_fit_reduced_chi2",
        ]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![
            "half amplitude of the Villar function (A)",
            "baseline of the Villar function (c)",
            "reference time of the Villar function (t_0)",
            "rise time of the Villar function (tau_rise)",
            "decline time of the Villar function (tau_fall)",
            "relative plateau amplitude of the Villar function (nu = beta gamma / A)",
            "plateau duration of the Villar function (gamma)",
            "Villar fit quality (reduced chi2)",
        ]
    }
}

impl<T> FeatureEvaluator<T> for VillarFit
where
    T: Float,
{
    fit_eval!();
}

struct Params<'a, T> {
    internal: &'a [T; NPARAMS],
    external: [T; NPARAMS],
}

impl<'a, T> Params<'a, T>
where
    T: LikeFloat,
{
    #[inline]
    fn a(&self) -> T {
        self.external[0]
    }

    #[inline]
    fn sgn_a_internal(&self) -> T {
        self.internal[0].signum()
    }

    #[inline]
    fn c(&self) -> T {
        self.external[1]
    }

    #[inline]
    fn t0(&self) -> T {
        self.external[2]
    }

    #[inline]
    fn tau_rise(&self) -> T {
        self.external[3]
    }

    #[inline]
    fn sgn_tau_rise_internal(&self) -> T {
        self.internal[3].signum()
    }

    #[inline]
    fn tau_fall(&self) -> T {
        self.external[4]
    }

    #[inline]
    fn sgn_tau_fall_internal(&self) -> T {
        self.internal[4].signum()
    }

    /// $\nu = 2 logistic(2|b|) - 1$
    /// Properties:
    /// dnu/db = 4 logistic(2|b|) (1 - logistic(2|b|)) sgn(b) = (1 - nu^2) sgn(b)
    /// $\nu = 0$ <=> $b = 0$, for this point $d\nu/db = 1$
    /// $\nu = 1$ <=> $|b| = inf$
    #[inline]
    fn nu(&self) -> T {
        self.external[5]
    }

    #[inline]
    fn sgn_b(&self) -> T {
        self.internal[5].signum()
    }

    #[inline]
    fn gamma(&self) -> T {
        self.external[6]
    }

    #[inline]
    fn sgn_gamma_internal(&self) -> T {
        self.internal[6].signum()
    }

    #[inline]
    fn t1(&self) -> T {
        self.t0() + self.gamma()
    }

    #[inline]
    fn dt(&self, t: T) -> T {
        t - self.t0()
    }

    #[inline]
    fn rise(&self, t: T) -> T {
        T::logistic(self.dt(t) / self.tau_rise())
    }

    #[inline]
    fn plateau(&self, t: T) -> T {
        T::one() - self.nu() * T::min(self.dt(t) / self.gamma(), T::one())
    }

    #[inline]
    fn fall(&self, t: T) -> T {
        let t1 = self.t1();
        if t <= t1 {
            T::one()
        } else {
            T::exp(-(t - t1) / self.tau_fall())
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[non_exhaustive]
pub enum VillarInitsBounds {
    Default,
    Arrays(Box<FitInitsBoundsArrays<NPARAMS>>),
    OptionArrays(Box<OptionFitInitsBoundsArrays<NPARAMS>>),
}

impl Default for VillarInitsBounds {
    fn default() -> Self {
        Self::Default
    }
}

impl VillarInitsBounds {
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

        let a_init = 0.5 * m_amplitude;
        let (a_lower, a_upper) = (0.0, 100.0 * m_amplitude);

        let c_init = m_min;
        let (c_lower, c_upper) = (m_min - 100.0 * m_amplitude, m_max + 100.0 * m_amplitude);

        // t0 is not a peak time, but something before the peak
        let t0_init = t_peak;
        let (t0_lower, t0_upper) = (t_min - 20.0 * t_amplitude, t_max + 10.0 * t_amplitude);

        let tau_rise_init = 0.5 * t_amplitude;
        let (tau_rise_lower, tau_rise_upper) = (0.0, 10.0 * t_amplitude);

        let tau_fall_init = 0.5 * t_amplitude;
        let (tau_fall_lower, tau_fall_upper) = (0.0, 10.0 * t_amplitude);

        let nu_init = 0.0;
        let (nu_lower, nu_upper) = (0.0, 1.0);

        let gamma_init = 0.1 * t_amplitude;
        let (gamma_lower, gamma_upper) = (0.0, 10.0 * t_amplitude);

        FitInitsBoundsArrays {
            init: [
                a_init,
                c_init,
                t0_init,
                tau_rise_init,
                tau_fall_init,
                nu_init,
                gamma_init,
            ]
            .into(),
            lower: [
                a_lower,
                c_lower,
                t0_lower,
                tau_rise_lower,
                tau_fall_lower,
                nu_lower,
                gamma_lower,
            ]
            .into(),
            upper: [
                a_upper,
                c_upper,
                t0_upper,
                tau_rise_upper,
                tau_fall_upper,
                nu_upper,
                gamma_upper,
            ]
            .into(),
        }
    }
}

/// Logarithm of priors for [VillarFit] parameters
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum VillarLnPrior {
    Fixed(Box<LnPrior<NPARAMS>>),
    /// Adopted from Hosseinzadeh, et al. 2020, table 2
    ///
    /// `time_units_in_day` specifies the units of time you use in yout `TimeSeries` object, it
    /// should be `1` for days, `86400` for seconds, etc. `min_amplitude` is a lower bound for
    /// the log-uniform prior of amplitude, the original paper used unity
    Hosseinzadeh2020 {
        time_units_in_day: f64,
        min_amplitude: f64,
    },
}

impl VillarLnPrior {
    pub fn fixed(ln_prior: LnPrior<NPARAMS>) -> Self {
        Self::Fixed(ln_prior.into())
    }

    pub fn hosseinzadeh2020(time_units_in_day: f64, min_flux: f64) -> Self {
        Self::Hosseinzadeh2020 {
            time_units_in_day,
            min_amplitude: min_flux,
        }
    }

    pub fn ln_prior_from_ts<T: Float>(&self, ts: &mut TimeSeries<T>) -> LnPrior<NPARAMS> {
        match self {
            Self::Fixed(ln_prior) => ln_prior.as_ref().clone(),
            Self::Hosseinzadeh2020 {
                time_units_in_day: day,
                min_amplitude,
            } => {
                let t_peak: f64 = ts.get_t_max_m().value_into().unwrap();
                let m_min: f64 = ts.m.get_min().value_into().unwrap();
                let m_max: f64 = ts.m.get_max().value_into().unwrap();
                let m_amplitude = m_max - m_min;

                LnPrior::ind_components([
                    LnPrior1D::log_uniform(*min_amplitude, 100.0 * m_amplitude), // amplitude
                    LnPrior1D::none(), // offset, not used in the original paper
                    LnPrior1D::uniform(t_peak - 50.0 * day, t_peak + 300.0 * day), // reference time
                    LnPrior1D::uniform(0.01 * day, 50.0 * day), // tau_rise
                    LnPrior1D::uniform(1.0 * day, 300.0 * day), // tau_fall
                    LnPrior1D::none(), // relative plateau amplitude, original paper used slope in day^-1
                    LnPrior1D::mix(vec![
                        (2.0, LnPrior1D::normal(5.0 * day, 5.0 * day)),
                        (1.0, LnPrior1D::normal(60.0 * day, 30.0 * day)),
                    ]), // plateau duration
                ])
            }
        }
    }
}

impl From<LnPrior<NPARAMS>> for VillarLnPrior {
    fn from(item: LnPrior<NPARAMS>) -> Self {
        Self::fixed(item)
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::nl_fit::LnPrior1D;
    use crate::tests::*;
    use crate::LmsderCurveFit;
    use crate::TimeSeries;

    use approx::assert_relative_eq;
    use hyperdual::Hyperdual;

    check_feature!(VillarFit);

    feature_test!(
        villar_fit_plateau,
        [VillarFit::default()],
        [0.0, 0.0, 10.0, 5.0, 5.0, 0.0, 1.0, 0.0], // initial model parameters and zero chi2
        linspace(0.0, 10.0, 11),
        [0.0; 11],
    );

    fn villar_fit_noisy(eval: VillarFit) {
        const N: usize = 50;

        let mut rng = StdRng::seed_from_u64(0);

        let param_true = [1e4, 1e3, 20.0, 5.0, 30.0, 0.3, 30.0];

        let t = linspace(0.0, 100.0, N);
        let model: Vec<_> = t.iter().map(|&x| VillarFit::f(x, &param_true)).collect();
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

        // curve_fit(lambda t, a, c, t0, tau_rise, tau_fall, nu, gamma: c + a * (1 - nu * np.where(t - t0 < gamma, (t - t0) / gamma, 1)) / (1 + np.exp(-(t-t0) / tau_rise)) * np.where(t > t0 + gamma, np.exp(-(t-t0-gamma) / tau_fall), 1.0), xdata=t, ydata=m, sigma=np.array(w)**-0.5, p0=[1e4, 1e3, 20, 5, 30, 0.3, 30])
        let desired = [
            1.00899754e+04,
            1.00689083e+03,
            2.02385802e+01,
            5.04283765e+00,
            3.00679883e+01,
            3.00400783e-01,
            2.94149214e+01,
        ];

        let values = eval.eval(&mut ts).unwrap();

        println!(
            "t = {:?}\nmodel = {:?}\nm = {:?}\nw = {:?}\nreduced_chi2 = {}",
            t, model, m, w, values[NPARAMS]
        );

        assert_relative_eq!(&values[..NPARAMS], &desired[..], max_relative = 0.01);
    }

    #[test]
    fn villar_fit_noisy_lmsder() {
        villar_fit_noisy(VillarFit::new(
            LmsderCurveFit::new(6).into(),
            LnPrior::none(),
            VillarInitsBounds::Default,
        ));
    }

    #[test]
    fn villar_fit_noizy_mcmc_plus_lmsder() {
        let lmsder = LmsderCurveFit::new(1);
        let mcmc = McmcCurveFit::new(2048, Some(lmsder.into()));
        villar_fit_noisy(VillarFit::new(
            mcmc.into(),
            LnPrior::none(),
            VillarInitsBounds::Default,
        ));
    }

    #[test]
    fn villar_fit_noizy_mcmc_with_prior() {
        let prior = LnPrior::ind_components([
            LnPrior1D::normal(1e4, 1e4),
            LnPrior1D::normal(1e3, 1e3),
            LnPrior1D::uniform(5.0, 30.0),
            LnPrior1D::log_normal(f64::ln(5.0), 1.0),
            LnPrior1D::log_normal(f64::ln(30.0), 1.0),
            LnPrior1D::uniform(0.0, 0.5),
            LnPrior1D::log_normal(f64::ln(30.0), 1.0),
        ]);
        let lmsder = LmsderCurveFit::new(1);
        let mcmc = McmcCurveFit::new(1024, Some(lmsder.into()));
        villar_fit_noisy(VillarFit::new(
            mcmc.into(),
            prior,
            VillarInitsBounds::Default,
        ));
    }

    #[test]
    fn villar_fit_derivatives() {
        const REPEAT: usize = 10;

        let mut rng = StdRng::seed_from_u64(0);
        for _ in 0..REPEAT {
            let t = 10.0 * rng.gen::<f64>();

            let param = {
                let mut param = [0.0; NPARAMS];
                for x in param.iter_mut() {
                    *x = rng.gen::<f64>() - 0.5;
                }
                param
            };
            let actual = {
                let mut jac = [0.0; NPARAMS];
                VillarFit::derivatives(t, &param, &mut jac);
                jac
            };

            let desired: Vec<_> = {
                let hyper_param = {
                    let mut hyper = [Hyperdual::<f64, { NPARAMS + 1 }>::from_real(0.0); NPARAMS];
                    for (i, (x, h)) in param.iter().zip(hyper.iter_mut()).enumerate() {
                        h[0] = *x;
                        h[i + 1] = 1.0;
                    }
                    hyper
                };
                let result = VillarFit::model(t, &hyper_param);
                (1..=NPARAMS).map(|i| result[i]).collect()
            };

            assert_relative_eq!(&actual[..], &desired[..], epsilon = 1e-9);
        }
    }
}
