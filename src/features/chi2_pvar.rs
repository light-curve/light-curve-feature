use crate::evaluator::*;

macro_const! {
    const DOC: &'static str = r"
Probability of variability based on $\chi^2$ distribution

$$
p_{\rm var} \equiv Q\!\left(\frac{N-1}{2},\,\frac{\chi^2}{2}\right),
$$
where $Q(a, x)$ is the regularized upper incomplete gamma function,
$N$ is the number of observations, and
$$
\chi^2 \equiv \sum_i\left(\frac{m_i - \bar{m}}{\delta_i}\right)^2
$$
is the chi-squared statistic with $N-1$ degrees of freedom.
The weighted mean $\bar{m}$ is computed as described in [WeightedMean].

$p_{\rm var}$ is the probability that the observed scatter exceeds what is
expected from measurement uncertainties alone, i.e.\ the probability that
the source is intrinsically variable.

- Depends on: **magnitude**, **magnitude error**
- Minimum number of observations: **2**
- Number of features: **1**

[Kim et al. 2014](https://ui.adsabs.harvard.edu/abs/2014A%26A...566A..43K/abstract)
";
}

#[doc = DOC!()]
#[derive(Clone, Default, Debug, Serialize, Deserialize, JsonSchema, PartialEq, Eq, Hash)]
pub struct Chi2Pvar {}

lazy_info!(
    CHI2_PVAR_INFO,
    Chi2Pvar,
    size: 1,
    min_ts_length: 2,
    t_required: false,
    m_required: true,
    w_required: true,
    sorting_required: false,
    variability_required: false,
);

impl Chi2Pvar {
    pub fn new() -> Self {
        Self {}
    }

    pub const fn doc() -> &'static str {
        DOC
    }
}

impl FeatureNamesDescriptionsTrait for Chi2Pvar {
    fn get_names(&self) -> Vec<&str> {
        vec!["chi2_pvar"]
    }

    fn get_descriptions(&self) -> Vec<&str> {
        vec!["probability of variability from the chi-squared test"]
    }
}

impl<T> FeatureEvaluator<T> for Chi2Pvar
where
    T: Float,
{
    fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        use special::Gamma;
        let chi2: f64 = ts.get_m_chi2().value_into().unwrap();
        let dof: f64 = (ts.lenu() - 1) as f64;
        let pvar = 1.0_f64 - (chi2 / 2.0).inc_gamma(dof / 2.0);
        Ok(vec![T::approx_from(pvar.clamp(0.0, 1.0)).unwrap()])
    }
}

#[cfg(test)]
#[allow(clippy::unreadable_literal)]
#[allow(clippy::excessive_precision)]
mod tests {
    use super::*;
    use crate::tests::*;

    check_feature!(Chi2Pvar);

    // Expected value computed with:
    //   import numpy as np
    //   from scipy.special import gammaincc
    //   m = np.array([1., 2., 1., 0., -1., 0., 1., 2., -2., 0.])
    //   w = np.array([1., 2., 1., 2.,  1., 2., 1., 2.,  1., 2.])
    //   wmean = np.average(m, weights=w)
    //   chi2 = np.sum((m - wmean)**2 * w)   # 19.7333...
    //   dof = len(m) - 1                     # 9
    //   gammaincc(dof / 2, chi2 / 2)         # 0.019631336718999857
    feature_test!(
        chi2_pvar,
        [Chi2Pvar::default()],
        [0.019631336718999857_f64],
        [0.0_f64; 10], // isn't used
        [1.0, 2.0, 1.0, 0.0, -1.0, 0.0, 1.0, 2.0, -2.0, 0.0],
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
    );
}
