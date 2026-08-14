/// Helper for static EvaluatorInfo creation
macro_rules! lazy_info {
    (
        $name: ident,
        size: $size: expr,
        min_ts_length: $len: expr,
        t_required: $t: expr,
        m_required: $m: expr,
        w_required: $w: expr,
        sorting_required: $sort: expr,
        variability_required: $var: expr,
    ) => {
        lazy_static! {
            static ref $name: EvaluatorInfo = EvaluatorInfo {
                size: $size,
                min_ts_length: $len,
                t_required: $t,
                m_required: $m,
                w_required: $w,
                sorting_required: $sort,
                variability_required: $var,
            };
        }
    };
    (
        $name: ident,
        $feature: ty,
        size: $size: expr,
        min_ts_length: $len: expr,
        t_required: $t: expr,
        m_required: $m: expr,
        w_required: $w: expr,
        sorting_required: $sort: expr,
        variability_required: $var: expr,
    ) => {
        lazy_info!(
            $name,
            size: $size,
            min_ts_length: $len,
            t_required: $t,
            m_required: $m,
            w_required: $w,
            sorting_required: $sort,
            variability_required: $var,
        );

        impl EvaluatorInfoTrait for $feature {
            fn get_info(&self) -> &EvaluatorInfo {
                &$name
            }
        }
    };
    (
        $name: ident,
        $feature: ty,
        T,
        size: $size: expr,
        min_ts_length: $len: expr,
        t_required: $t: expr,
        m_required: $m: expr,
        w_required: $w: expr,
        sorting_required: $sort: expr,
        variability_required: $var: expr,
    ) => {
        lazy_info!(
            $name,
            size: $size,
            min_ts_length: $len,
            t_required: $t,
            m_required: $m,
            w_required: $w,
            sorting_required: $sort,
            variability_required: $var,
        );

        impl<T: Float> EvaluatorInfoTrait for $feature {
            fn get_info(&self) -> &EvaluatorInfo {
                &$name
            }
        }
    };
}

/// Helper for FeatureEvaluator implementations using time-series transformation.
/// You must implement:
/// - `transform_ts(&self, ts: &mut TimeSeries<T>) -> Result<impl OwnedArrays<T>, EvaluatorError>`
macro_rules! transformer_eval {
    () => {
        fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
            let arrays = self.transform_ts(ts)?;
            let mut new_ts = arrays.ts();
            self.feature_extractor.eval(&mut new_ts)
        }

        fn eval_or_fill(&self, ts: &mut TimeSeries<T>, fill_value: T) -> Vec<T> {
            let arrays = match self.transform_ts(ts) {
                Ok(x) => x,
                Err(_) => return vec![fill_value; self.size_hint()],
            };
            let mut new_ts = arrays.ts();
            self.feature_extractor.eval_or_fill(&mut new_ts, fill_value)
        }
    };
}

/// Helper implementing JsonSchema crate
macro_rules! json_schema {
    ($parameters: ty, $is_referenceable: expr) => {
        fn is_referenceable() -> bool {
            $is_referenceable
        }

        fn schema_name() -> String {
            <$parameters>::schema_name()
        }

        fn json_schema(r#gen: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
            <$parameters>::json_schema(r#gen)
        }
    };
}

/// Helper implementing *Fit feature evaluators
/// You must:
/// - implement all traits of [nl_fit::evaluator]
/// - satisfy all [FeatureEvaluator] trait constraints
/// - declare `const NPARAMS: usize` in your code
///
/// `$n` must be that same `NPARAMS` value: `curve_fit`'s solver-facing API is generic over
/// `Fn(f64, &[f64]) -> f64`-shaped closures (GSL/Ceres/emcee/nuts-rs don't know about
/// `MAX_NPARAMS`), while `Self::model`/`Self::derivatives` take fixed-size `&[T; $n]` so their
/// own indexing is bounds-check-free. This macro is the one place that bridges the two: it
/// builds thin wrapper closures that convert the solver's runtime slice to a fixed array before
/// calling into the feature's own math.
macro_rules! fit_eval {
    ($n: expr) => {
        fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
            let norm_data = NormalizedData::<f64>::from_ts(ts);

            let (x0, lower, upper) = {
                let FitInitsBoundsArrays { init, lower, upper } = self.init_and_bounds_from_ts(ts);
                let x0 = Self::convert_to_internal(&norm_data, &init);
                let lower = Self::convert_to_internal(&norm_data, &lower);
                let upper = Self::convert_to_internal(&norm_data, &upper);
                (x0, lower, upper)
            };

            let model_closure = |t: f64, param: &[f64]| -> f64 {
                let arr: &[f64; $n] = param
                    .try_into()
                    .expect("curve_fit always calls model with exactly NPARAMS values");
                Self::model(t, arr)
            };
            let derivatives_closure = |t: f64, param: &[f64], jac: &mut [f64]| {
                let arr: &[f64; $n] = param
                    .try_into()
                    .expect("curve_fit always calls derivatives with exactly NPARAMS values");
                let mut jac_arr = [0.0_f64; $n];
                Self::derivatives(t, arr, &mut jac_arr);
                jac.copy_from_slice(&jac_arr);
            };

            let result = {
                let CurveFitResult {
                    x, reduced_chi2, ..
                } = self.get_algorithm().curve_fit(
                    norm_data.data.clone(),
                    &x0,
                    (&lower, &upper),
                    model_closure,
                    derivatives_closure,
                    self.ln_prior_from_ts(ts)
                        .with_fit_parameters_transformation::<Self, $n>(&norm_data),
                );
                let result = Self::convert_to_external(&norm_data, &x);
                result
                    .into_iter()
                    .chain(std::iter::once(reduced_chi2))
                    .map(|x| {
                        x.approx_as::<T>().unwrap_or_else(|_| {
                            if x.is_sign_negative() {
                                T::min_value()
                            } else {
                                T::max_value()
                            }
                        })
                    })
                    .collect()
            };

            Ok(result)
        }
    };
}
