use crate::evaluator::*;
use crate::nl_fit::{
    data::NormalizedData, evaluator::*, CurveFitAlgorithm, CurveFitResult, CurveFitTrait,
    LikeFloat, LnPrior, McmcCurveFit,
};

use conv::ConvUtil;

const NPARAMS: usize = 4;

macro_const! {
    const DOC: &str = r#"
Linexp function fit

Four fit parameters and goodness of fit (reduced $\chi^2$) of the Linexp function developed for
core-collapsed supernovae:

$$
f(t) = A(t-t_0) \times \mathrm{e}^{-\tau_\mathrm{fall} \times (t-t_0)} + B.
$$

Note, that the Linexp function is developed to be used with fluxes, not magnitudes. 

- Depends on: **time**, **flux**, **flux error**
- Minimum number of observations: **6**
- Number of features: **5**

"#;
}

#[doc = DOC!()]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
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

    pub fn doc() -> &'static str {
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

impl<'a, T> Params<'a, T>
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
    fn tau_fall(&self) -> T {
        self.external[2]
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
        let minus_dt = x.t0() - t;
        x.b() + (x.a() * minus_dt) * U::exp(-x.tau_fall() * minus_dt)
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
        let minus_dt = x.t0() - t;
        let exp_fall = T::exp(-x.tau_fall() * minus_dt);

        // a
        jac[0] = x.sgn_a() * minus_dt * exp_fall;
        // t0
        jac[1] = x.a() * exp_fall * (x.tau_fall() * minus_dt - T::one());
        // tau_fall
        jac[2] = x.a() * t * minus_dt * exp_fall;
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
            norm_data.t_to_norm_scale(orig[2]), // tau_fall fall slope
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
            norm_data.t_to_orig_scale(norm[2]), // tau_fall fall slope
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
        [
            params[0].abs(),
            params[1],
            params[2].abs(),
            params[3],
        ]
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
            "linexp_fit_fall_slope",
            "linexp_fit_baseline",
            "linexp_fit_reduced_chi2",
        ]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![
            "Amplitude of the Linexp function (A)",
            "reference time of the Linexp fit (t0)",
            "fall slope of the Linexp function (tau_fall)",
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

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[non_exhaustive]
pub enum LinexpInitsBounds {
    Default,
    Arrays(Box<FitInitsBoundsArrays<NPARAMS>>),
    OptionArrays(Box<OptionFitInitsBoundsArrays<NPARAMS>>),
}

impl Default for LinexpInitsBounds {
    fn default() -> Self {
        Self::Default
    }
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
        let t_peak: f64 = ts.get_t_max_m().value_into().unwrap();
        let m_min: f64 = ts.m.get_min().value_into().unwrap();
        let m_max: f64 = ts.m.get_max().value_into().unwrap();
        let m_amplitude = m_max - m_min;

        let a_init = 0.015 * m_amplitude;
        let (a_lower, a_upper) = (0.0, 2.0 * m_amplitude);

        let t0_init = t_peak - 15.0;
        let (t0_lower, t0_upper) = (t_peak - 300.0, t_peak + 300.0);

        let fall_init = 0.005;
        let (fall_lower, fall_upper) = (0.0, 2.0);
        
        let b_init = m_min;
        let (b_lower, b_upper) = (m_min - 100.0 * m_amplitude, m_max + 100.0 * m_amplitude);

        FitInitsBoundsArrays {
            init: [a_init, t0_init, fall_init, b_init].into(),
            lower: [a_lower, t0_lower, fall_lower, b_lower].into(),
            upper: [a_upper, t0_upper, fall_upper, b_upper].into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
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

/*

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::nl_fit::LnPrior1D;
    use crate::tests::*;
    #[cfg(any(feature = "ceres-source", feature = "ceres-system"))]
    use crate::CeresCurveFit;
    #[cfg(feature = "gsl")]
    use crate::LmsderCurveFit;
    use crate::TimeSeries;

    use approx::assert_relative_eq;
    use hyperdual::Hyperdual;

    check_feature!(LinexpFit);

    feature_test!(
        linexp_fit_plateau,
        [LinexpFit::default()],
        [0.015, 15.0, 0.005, 0.0, 0.0], // initial model parameters and zero chi2
        linspace(0.0, 10.0, 11),
        [0.0; 11],
    );

    fn linexp_fit_noisy(eval: LinexpFit) {
        const N: usize = 50;

        let mut rng = StdRng::seed_from_u64(0);

        let param_true = [0.03, 0.0, 0.007, 0.0];
        let noise_scale = 0.05;

        let t = linspace(-10.0, 600.0, N);
        let model: Vec<_> = t.iter().map(|&x| LinexpFit::model(x, &param_true)).collect();
        let m: Vec<_> = model
            .iter()
            .map(|&y| {
                let std = noise_scale * f64::abs(y);
                let err = std * rng.sample::<f64, _>(StandardNormal);
                y + err
            })
            .collect();
        let w: Vec<_> = model.iter().copied().map(f64::recip).collect();
        println!("{:?}\n{:?}\n{:?}\n{:?}", t, model, m, w);
        let mut ts = TimeSeries::new(&t, &m, &w);

        // curve_fit(lambda t, a, t0, fall, b : b + a * (t - t0) * np.exp(-fall * (t - t0)), xdata=t, ydata=m, sigma=0.05*abs(y), p0=[0.015, 15, .005, 0])
        let desired = [
            -0.00700205,
            0.02974836,
            0.03040189,
            0.0,
        ];

        let values = eval.eval(&mut ts).unwrap();
        assert_relative_eq!(&values[..5], &desired[..], max_relative = 0.01);
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
                [None; 5],
                [None; 5],
                [None, None, Some(50.0), Some(50.0), Some(50.0)],
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
            LnPrior1D::normal(1e4, 2e3),
            LnPrior1D::normal(1e3, 2e2),
            LnPrior1D::uniform(25.0, 35.0),
            LnPrior1D::log_normal(f64::ln(10.0), 0.2),
            LnPrior1D::log_normal(f64::ln(30.0), 0.2),
        ]);
        let mcmc = McmcCurveFit::new(1024, None);
        linexp_fit_noisy(LinexpFit::new(mcmc.into(), prior, LinexpInitsBounds::Default));
    }

    #[test]
    fn linexp_fit_derivatives() {
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
                LinexpFit::derivatives(t, &param, &mut jac);
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
                let result = LinexpFit::model(t, &hyper_param);
                (1..=NPARAMS).map(|i| result[i]).collect()
            };

            assert_relative_eq!(&actual[..], &desired[..], epsilon = 1e-9);
        }
    }

    /// https://github.com/light-curve/light-curve-feature/issues/29
    #[test]
    fn linexp_fit_different_flux_scale() {
        const MAG_ERR: f64 = 0.01;

        let (t, m, _) = light_curve_feature_test_util::issue_light_curve_mag(
            "light-curve-feature-29/1.csv",
            None,
        );

        let f0 = [3.63e-20f64, 1.0];
        let m_models: Vec<_> = f0
            .iter()
            .map(|&f0| {
                let flux: Vec<_> = m.iter().map(|&y| f0 * f64::powf(10.0, -0.4 * y)).collect();
                let w: Vec<_> = flux
                    .iter()
                    .map(|y| (0.4 * f64::ln(10.0) * y * MAG_ERR).powi(-2)) // 1 / sigma^2
                    .collect();

                let mut ts = TimeSeries::new(t.view(), &flux, &w);
                let linexp = LinexpFit::new(
                    McmcCurveFit::new(1_000, None).into(),
                    LnPrior::none(),
                    LinexpInitsBounds::option_arrays(
                        [None; 5],
                        [
                            None,
                            Some(f0 * f64::powf(10.0, -0.4 * 22.0)),
                            None,
                            None,
                            None,
                        ],
                        [None; 5],
                    ),
                );
                let result = linexp.eval(&mut ts).unwrap();

                let t_model = Array1::ace(t[0] - 1.0, t[t.len() - 1] + 1.0, 100);
                let flux_model =
                    t_model.mapv(|x| LinexpFit::model(x, &result[..NPARAMS].try_into().unwrap()));
                flux_model.mapv(|y| -2.5 * f64::log10(y / f0))
            })
            .collect();
        assert_relative_eq!(
            m_models[0].as_slice().unwrap(),
            m_models[1].as_slice().unwrap(),
            epsilon = 1e-9,
        );
    }

    /// https://github.com/light-curve/light-curve-feature/issues/51
    #[test]
    fn linexp_fit_nan_lnprior() {
        let mut ts = light_curve_feature_test_util::issue_light_curve_flux::<f64, _>(
            "light-curve-feature-51/1.csv",
            None,
        )
        .into();
        let linexp = LinexpFit::new(
            McmcCurveFit::new(1 << 14, None).into(),
            LnPrior::none(),
            LinexpInitsBounds::Default,
        );
        let _result = linexp.eval(&mut ts).unwrap();
    }
}
*/

