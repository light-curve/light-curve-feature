use crate::evaluator::*;
use crate::nl_fit::{
    data::NormalizedData, evaluator::*, CurveFitAlgorithm, CurveFitResult, CurveFitTrait,
    LikeFloat, LnPrior, McmcCurveFit,
};

use conv::ConvUtil;

const NPARAMS: usize = 5;

macro_const! {
    const DOC: &str = r#"
Bazin function fit

Five fit parameters and goodness of fit (reduced $\chi^2$) of the Bazin function developed for
core-collapsed supernovae:

$$
f(t) = A \frac{ \mathrm{e}^{ -(t-t_0)/\tau_\mathrm{fall} } }{ 1 + \mathrm{e}^{ -(t - t_0) / \tau_\mathrm{rise} } } + B.
$$

Note, that the Bazin function is developed to be used with fluxes, not magnitudes. Also note a typo
in the Eq. (1) of the original paper, the minus sign is missed in the "rise" exponent.

- Depends on: **time**, **magnitude**, **magnitude error**
- Minimum number of observations: **6**
- Number of features: **6**

Bazin et al. 2009 [DOI:10.1051/0004-6361/200911847](https://doi.org/10.1051/0004-6361/200911847)
"#;
}

#[doc = DOC!()]
#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
pub struct BazinFit {
    algorithm: CurveFitAlgorithm,
    ln_prior: BazinLnPrior,
    inits_bounds: BazinInitsBounds,
}

impl BazinFit {
    /// New [BazinFit] instance
    ///
    /// `algorithm` specifies which optimization method is used, it is an instance of the
    /// [CurveFitAlgorithm], currently supported algorithms are [MCMC](McmcCurveFit) and
    /// [LMSDER](crate::nl_fit::LmsderCurveFit) (a Levenberg–Marquard algorithm modification,
    /// requires `gsl` Cargo feature).
    ///
    /// `ln_prior` is an instance of [BazinLnPrior] and specifies the natural logarithm of the prior
    /// to use. Some curve-fit algorithms doesn't support this and ignores the prior
    pub fn new<BLP>(
        algorithm: CurveFitAlgorithm,
        ln_prior: BLP,
        inits_bounds: BazinInitsBounds,
    ) -> Self
    where
        BLP: Into<BazinLnPrior>,
    {
        Self {
            algorithm,
            ln_prior: ln_prior.into(),
            inits_bounds,
        }
    }

    /// Default [McmcCurveFit] for [BazinFit]
    #[inline]
    pub fn default_algorithm() -> CurveFitAlgorithm {
        McmcCurveFit::new(
            McmcCurveFit::default_niterations(),
            McmcCurveFit::default_fine_tuning_algorithm(),
        )
        .into()
    }

    /// Default [LnPrior] for [BazinFit]
    #[inline]
    pub fn default_ln_prior() -> BazinLnPrior {
        LnPrior::none().into()
    }

    #[inline]
    pub fn default_inits_bounds() -> BazinInitsBounds {
        BazinInitsBounds::Default
    }

    pub fn doc() -> &'static str {
        DOC
    }
}

impl Default for BazinFit {
    fn default() -> Self {
        Self::new(
            Self::default_algorithm(),
            Self::default_ln_prior(),
            Self::default_inits_bounds(),
        )
    }
}

lazy_info!(
    BAZIN_FIT_INFO,
    BazinFit,
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
    fn b(&self) -> T {
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
    fn sgn_tau_rise(&self) -> T {
        self.internal[3].signum()
    }

    #[inline]
    fn tau_fall(&self) -> T {
        self.external[4]
    }

    #[inline]
    fn sgn_tau_fall(&self) -> T {
        self.internal[4].signum()
    }
}

impl<T, U> FitModelTrait<T, U, NPARAMS> for BazinFit
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
        x.b()
            + x.a() * U::exp(minus_dt / x.tau_fall()) / (U::exp(minus_dt / x.tau_rise()) + U::one())
    }
}

impl<T> FitFunctionTrait<T, NPARAMS> for BazinFit where T: Float {}

impl<T> FitDerivalivesTrait<T, NPARAMS> for BazinFit
where
    T: Float,
{
    fn derivatives(t: T, param: &[T; NPARAMS], jac: &mut [T; NPARAMS]) {
        let x = Params {
            internal: param,
            external: Self::internal_to_dimensionless(param),
        };
        let minus_dt = x.t0() - t;
        let exp_rise = T::exp(minus_dt / x.tau_rise());
        let frac = T::exp(minus_dt / x.tau_fall()) / (T::one() + exp_rise);
        let exp_1p_exp_rise = (T::one() + exp_rise.recip()).recip();
        // a
        jac[0] = x.sgn_a() * frac;
        // b
        jac[1] = T::one();
        // t0
        jac[2] = x.a() * frac * (x.tau_fall().recip() - exp_1p_exp_rise / x.tau_rise());
        // tau_rise
        jac[3] =
            x.sgn_tau_rise() * x.a() * minus_dt * frac / x.tau_rise().powi(2) * exp_1p_exp_rise;
        // tau_fall
        jac[4] = -x.sgn_tau_fall() * x.a() * minus_dt * frac / x.tau_fall().powi(2);
    }
}

impl<T> FitInitsBoundsTrait<T, NPARAMS> for BazinFit
where
    T: Float,
{
    fn init_and_bounds_from_ts(&self, ts: &mut TimeSeries<T>) -> FitInitsBoundsArrays<NPARAMS> {
        match &self.inits_bounds {
            BazinInitsBounds::Default => BazinInitsBounds::default_from_ts(ts),
            BazinInitsBounds::Arrays(arrays) => arrays.as_ref().clone(),
            BazinInitsBounds::OptionArrays(opt_arr) => {
                opt_arr.unwrap_with(&BazinInitsBounds::default_from_ts(ts))
            }
        }
    }
}

impl FitParametersOriginalDimLessTrait<NPARAMS> for BazinFit {
    fn orig_to_dimensionless(
        norm_data: &NormalizedData<f64>,
        orig: &[f64; NPARAMS],
    ) -> [f64; NPARAMS] {
        [
            norm_data.m_to_norm_scale(orig[0]), // A amplitude
            norm_data.m_to_norm(orig[1]),       // c baseline
            norm_data.t_to_norm(orig[2]),       // t_0 reference_time
            norm_data.t_to_norm_scale(orig[3]), // tau_rise rise time
            norm_data.t_to_norm_scale(orig[4]), // tau_fall fall time
        ]
    }

    fn dimensionless_to_orig(
        norm_data: &NormalizedData<f64>,
        norm: &[f64; NPARAMS],
    ) -> [f64; NPARAMS] {
        [
            norm_data.m_to_orig_scale(norm[0]), // A amplitude
            norm_data.m_to_orig(norm[1]),       // c baseline
            norm_data.t_to_orig(norm[2]),       // t_0 reference_time
            norm_data.t_to_orig_scale(norm[3]), // tau_rise rise time
            norm_data.t_to_orig_scale(norm[4]), // tau_fall fall time
        ]
    }
}

impl<U> FitParametersInternalDimlessTrait<U, NPARAMS> for BazinFit
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
            params[2],
            params[3].abs(),
            params[4].abs(),
        ]
    }
}

impl FitParametersInternalExternalTrait<NPARAMS> for BazinFit {}

impl FitFeatureEvaluatorGettersTrait<NPARAMS> for BazinFit {
    fn get_algorithm(&self) -> &CurveFitAlgorithm {
        &self.algorithm
    }

    fn ln_prior_from_ts<T: Float>(&self, ts: &mut TimeSeries<T>) -> LnPrior<NPARAMS> {
        self.ln_prior.ln_prior_from_ts(ts)
    }
}

impl FeatureNamesDescriptionsTrait for BazinFit {
    fn get_names(&self) -> Vec<&str> {
        vec![
            "bazin_fit_amplitude",
            "bazin_fit_baseline",
            "bazin_fit_reference_time",
            "bazin_fit_rise_time",
            "bazin_fit_fall_time",
            "bazin_fit_reduced_chi2",
        ]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec![
            "half amplitude of the Bazin function (A)",
            "baseline of the Bazin function (B)",
            "reference time of the Bazin fit (t0)",
            "rise time of the Bazin function (tau_rise)",
            "fall time of the Bazin function (tau_fall)",
            "Bazin fit quality (reduced chi2)",
        ]
    }
}

impl<T> FeatureEvaluator<T> for BazinFit
where
    T: Float,
{
    fit_eval!();
}

#[derive(Clone, Debug, Deserialize, Serialize, JsonSchema)]
#[non_exhaustive]
pub enum BazinInitsBounds {
    Default,
    Arrays(Box<FitInitsBoundsArrays<NPARAMS>>),
    OptionArrays(Box<OptionFitInitsBoundsArrays<NPARAMS>>),
}

impl Default for BazinInitsBounds {
    fn default() -> Self {
        Self::Default
    }
}

impl BazinInitsBounds {
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

        let t0_init = t_peak;
        let (t0_lower, t0_upper) = (t_min - 10.0 * t_amplitude, t_max + 10.0 * t_amplitude);

        let rise_init = 0.5 * t_amplitude;
        let (rise_lower, rise_upper) = (0.0, 10.0 * t_amplitude);

        let fall_init = 0.5 * t_amplitude;
        let (fall_lower, fall_upper) = (0.0, 10.0 * t_amplitude);

        FitInitsBoundsArrays {
            init: [a_init, c_init, t0_init, rise_init, fall_init].into(),
            lower: [a_lower, c_lower, t0_lower, rise_lower, fall_lower].into(),
            upper: [a_upper, c_upper, t0_upper, rise_upper, fall_upper].into(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
#[non_exhaustive]
pub enum BazinLnPrior {
    Fixed(Box<LnPrior<NPARAMS>>),
}

impl BazinLnPrior {
    pub fn fixed(ln_prior: LnPrior<NPARAMS>) -> Self {
        Self::Fixed(ln_prior.into())
    }

    pub fn ln_prior_from_ts<T: Float>(&self, _ts: &mut TimeSeries<T>) -> LnPrior<NPARAMS> {
        match self {
            Self::Fixed(ln_prior) => ln_prior.as_ref().clone(),
        }
    }
}

impl From<LnPrior<NPARAMS>> for BazinLnPrior {
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
    #[cfg(feature = "gsl")]
    use crate::LmsderCurveFit;
    use crate::TimeSeries;

    use approx::assert_relative_eq;
    use hyperdual::Hyperdual;

    check_feature!(BazinFit);

    feature_test!(
        bazin_fit_plateau,
        [BazinFit::default()],
        [0.0, 0.0, 10.0, 5.0, 5.0, 0.0], // initial model parameters and zero chi2
        linspace(0.0, 10.0, 11),
        [0.0; 11],
    );

    fn bazin_fit_noisy(eval: BazinFit) {
        const N: usize = 50;

        let mut rng = StdRng::seed_from_u64(0);

        let param_true = [1e4, 1e3, 30.0, 10.0, 30.0];

        let t = linspace(0.0, 100.0, N);
        let model: Vec<_> = t.iter().map(|&x| BazinFit::model(x, &param_true)).collect();
        let m: Vec<_> = model
            .iter()
            .map(|&y| {
                let std = f64::sqrt(y);
                let err = std * rng.sample::<f64, _>(StandardNormal);
                y + err
            })
            .collect();
        let w: Vec<_> = model.iter().copied().map(f64::recip).collect();
        println!("{:?}\n{:?}\n{:?}\n{:?}", t, model, m, w);
        let mut ts = TimeSeries::new(&t, &m, &w);

        // curve_fit(lambda t, a, b, t0, rise, fall: b + a * np.exp(-(t-t0)/fall) / (1 + np.exp(-(t-t0) / rise)), xdata=t, ydata=m, sigma=np.array(w)**-0.5, p0=[1e4, 1e3, 30, 10, 30])
        let desired = [
            9.89658673e+03,
            1.11312724e+03,
            3.06401284e+01,
            9.75027284e+00,
            2.86714363e+01,
        ];

        let values = eval.eval(&mut ts).unwrap();
        assert_relative_eq!(&values[..5], &desired[..], max_relative = 0.01);
    }

    #[cfg(feature = "gsl")]
    #[test]
    fn bazin_fit_noisy_lmsder() {
        bazin_fit_noisy(BazinFit::new(
            LmsderCurveFit::new(9).into(),
            LnPrior::none(),
            BazinInitsBounds::Default,
        ));
    }

    #[cfg(feature = "gsl")]
    #[test]
    fn bazin_fit_noizy_mcmc_plus_lmsder() {
        let lmsder = LmsderCurveFit::new(1);
        let mcmc = McmcCurveFit::new(512, Some(lmsder.into()));
        bazin_fit_noisy(BazinFit::new(
            mcmc.into(),
            LnPrior::none(),
            BazinInitsBounds::Default,
        ));
    }

    #[test]
    fn bazin_fit_noizy_mcmc_with_prior() {
        let prior = LnPrior::ind_components([
            LnPrior1D::normal(1e4, 2e3),
            LnPrior1D::normal(1e3, 2e2),
            LnPrior1D::uniform(25.0, 35.0),
            LnPrior1D::log_normal(f64::ln(10.0), 0.2),
            LnPrior1D::log_normal(f64::ln(30.0), 0.2),
        ]);
        let mcmc = McmcCurveFit::new(1024, None);
        bazin_fit_noisy(BazinFit::new(mcmc.into(), prior, BazinInitsBounds::Default));
    }

    #[test]
    fn bazin_fit_derivatives() {
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
                BazinFit::derivatives(t, &param, &mut jac);
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
                let result = BazinFit::model(t, &hyper_param);
                (1..=NPARAMS).map(|i| result[i]).collect()
            };

            assert_relative_eq!(&actual[..], &desired[..], epsilon = 1e-9);
        }
    }

    /// https://github.com/light-curve/light-curve-feature/issues/29
    #[test]
    fn bazin_fit_different_flux_scale() {
        const MAG_ERR: f64 = 0.01;

        let t = [
            5369.72858, 5369.77391, 5369.81188, 5369.85266, 5376.65853, 5376.69891, 5376.73714,
            5376.77595, 5376.8141, 5376.85206, 5377.54863, 5377.59134, 5377.62954, 5377.70322,
            5377.74667, 5377.78939, 5377.8271, 5377.86528, 5378.54904, 5378.6337, 5378.72268,
            5378.76098, 5378.79951, 5378.83939, 5378.87711, 5379.51614, 5379.58578, 5379.67183,
            5379.74434, 5379.782, 5379.81992, 5379.86049, 5380.52762, 5380.56812, 5380.65217,
            5380.72337, 5380.76161, 5380.79947, 5380.8374, 5381.53714, 5381.57471, 5381.61486,
            5381.67421, 5381.71252, 5381.75066, 5381.79758, 5381.83571, 5382.52592, 5382.5635,
            5382.61292, 5382.65083, 5382.69261, 5382.73245, 5382.77025, 5382.82095, 5383.77655,
            5384.80008, 5385.47067, 5385.50989, 5385.8104, 5385.84797, 5386.47831, 5386.56051,
            5386.60757, 5386.64396, 5386.74341, 5386.78168, 5386.82469, 5387.50393, 5387.61293,
            5387.6477, 5387.68716, 5387.72579, 5387.76655, 5387.80776, 5388.46773, 5388.50533,
            5388.56524, 5388.67765, 5388.76433, 5388.80758, 5388.84645, 5389.49504, 5389.53318,
            5389.61851, 5389.72375, 5389.78412, 5389.82465, 5390.4695, 5390.52432, 5390.56634,
            5390.65192, 5390.69816, 5390.74762, 5390.78557, 5390.82325, 5391.5183, 5391.55708,
            5391.61665, 5391.72592, 5391.8092, 5391.84666, 5392.49444, 5392.53316, 5392.5946,
            5392.70472, 5392.78668, 5392.82431, 5393.63967, 5393.67939, 5393.71757, 5393.76293,
            5393.80104, 5395.80807,
        ];
        let m = [
            21.401, 20.299, 20.245, 21.335, 18.103, 18.1, 18.077, 18.088, 18.109, 18.167, 18.132,
            18.167, 18.144, 18.074, 18.187, 18.124, 18.113, 18.075, 18.191, 18.171, 18.182, 18.161,
            18.181, 18.098, 18.18, 18.164, 18.191, 18.181, 18.189, 18.197, 18.166, 18.08, 18.205,
            18.161, 18.257, 18.168, 18.232, 18.264, 18.112, 18.254, 18.126, 18.156, 18.296, 18.142,
            18.048, 18.233, 18.223, 18.256, 18.223, 18.034, 18.238, 18.243, 18.087, 18.031, 18.32,
            18.12, 18.279, 18.225, 18.234, 18.175, 18.134, 18.3, 18.257, 18.314, 18.113, 18.333,
            18.18, 18.259, 18.342, 18.204, 18.304, 18.324, 18.33, 18.25, 18.312, 18.41, 18.388,
            18.357, 18.428, 18.412, 18.452, 18.319, 18.44, 18.431, 18.465, 18.418, 18.491, 18.294,
            18.58, 18.402, 18.483, 18.494, 18.531, 18.53, 18.449, 18.498, 18.575, 18.674, 18.548,
            18.715, 18.562, 18.566, 18.784, 18.642, 18.614, 18.708, 18.637, 18.642, 18.785, 18.825,
            18.598, 18.763, 18.858, 18.974,
        ];

        let f0 = [3.63e-20f64, 1.0];
        let m_models: Vec<_> = f0
            .iter()
            .map(|&f0| {
                let flux: Vec<_> = m.iter().map(|&y| f0 * f64::powf(10.0, -0.4 * y)).collect();
                let w: Vec<_> = flux
                    .iter()
                    .map(|y| (0.4 * f64::ln(10.0) * y * MAG_ERR).powi(-2)) // 1 / sigma^2
                    .collect();

                let mut ts = TimeSeries::new(&t, &flux, &w);
                let bazin = BazinFit::new(
                    McmcCurveFit::new(1_000, None).into(),
                    LnPrior::none(),
                    BazinInitsBounds::option_arrays(
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
                let result = bazin.eval(&mut ts).unwrap();

                let t_model = Array1::linspace(t[0] - 1.0, t[t.len() - 1] + 1.0, 100);
                let flux_model =
                    t_model.mapv(|x| BazinFit::model(x, &result[..NPARAMS].try_into().unwrap()));
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
    fn bazin_fit_nan_lnprior() {
        let t = [
            1011.352, 1022.3424, 1025.351, 1038.2935, 1051.2896, 1061.3154, 1062.2382, 1080.213,
            1081.1609, 1092.1885, 1104.1208, 1106.1193, 1120.1038, 1123.0575, 1133.093, 1148.9738,
            1152.044, 1159.9653, 1377.3401,
        ];
        let flux = [
            -1.554567, -8.362201, 14.185513, -4.4885917, -6.841251, 0.06717205, -1.3147569,
            12.001787, 17.649708, -5.276116, -3.7433321, -3.4837098, -1.3123958, 3.7815158,
            0.9699717, 29.324707, 27.131853, 51.47246, 1.0797225,
        ];
        let fluxerr = [
            4.8021693, 5.1533613, 9.4972925, 6.1327066, 6.664459, 9.421627, 8.472085, 6.631271,
            7.9540033, 6.8943286, 13.816224, 11.715339, 6.2831564, 8.967609, 7.8787127, 7.501725,
            6.7277308, 15.790535, 7.6157956,
        ];
        let w: Vec<_> = fluxerr.into_iter().map(|x: f32| x.powi(-2)).collect();
        let mut ts = TimeSeries::new(&t, &flux, &w);
        let bazin = BazinFit::new(
            McmcCurveFit::new(1 << 14, None).into(),
            LnPrior::none(),
            BazinInitsBounds::Default,
        );
        let _result = bazin.eval(&mut ts).unwrap();
    }
}
