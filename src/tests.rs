pub use crate::evaluator::*;
pub use crate::extractor::FeatureExtractor;
pub use crate::feature::Feature;
pub use crate::float_trait::Float;
pub use crate::time_series::TimeSeries;

pub use light_curve_common::{all_close, linspace};
pub use ndarray::Array1;
pub use rand::prelude::*;
pub use rand_distr::StandardNormal;

#[macro_export]
macro_rules! feature_test {
    ($name: ident, $fe: tt, $desired: expr_2021, $y: expr_2021 $(,)?) => {
        feature_test!($name, $fe, $desired, $y, $y);
    };
    ($name: ident, $fe: tt, $desired: expr_2021, $x: expr_2021, $y: expr_2021 $(,)?) => {
        feature_test!($name, $fe, $desired, $x, $y, vec![1.0; $x.len()]);
    };
    ($name: ident, $fe: tt, $desired: expr_2021, $x: expr_2021, $y: expr_2021, $w: expr_2021 $(,)?) => {
        feature_test!($name, $fe, $desired, $x, $y, $w, 1e-6);
    };
    ($name: ident, $fe: tt, $desired: expr_2021, $x: expr_2021, $y: expr_2021, $w: expr_2021, $tol: expr_2021 $(,)?) => {
        #[test]
        fn $name() {
            let fe = FeatureExtractor::new(vec!$fe);
            let desired = $desired;
            let x = $x;
            let y = $y;
            let w = $w;
            let mut ts = TimeSeries::new(&x, &y, &w);
            let actual = fe.eval(&mut ts).unwrap();
            all_close(&desired[..], &actual[..], $tol);

            let names = fe.get_names();
            let descs = fe.get_descriptions();
            assert_eq!(fe.size_hint(), actual.len(), "size_hint() returns wrong size");
            assert_eq!(actual.len(), names.len(),
                "Length of values and names should be the same");
            assert_eq!(actual.len(), descs.len(),
                "Length of values and descriptions should be the same");
        }
    };
}

#[macro_export]
macro_rules! eval_info_test {
    ($name: ident, $eval: expr_2021 $(,)?) => {
        #[test]
        fn $name() {
            eval_info_tests($eval.into(), true, true, true, true, true);
        }
    };
}

pub fn eval_info_tests(
    eval: Feature<f64>,
    test_ts_length: bool,
    test_t_required: bool,
    test_m_required: bool,
    test_w_required: bool,
    test_sorting_required: bool,
) {
    const N: usize = 128;

    let mut rng = StdRng::seed_from_u64(0);

    let t = randvec::<f64>(&mut rng, N);
    let t_sorted = sorted(&t);
    assert_ne!(t, t_sorted);
    let m = randvec::<f64>(&mut rng, N);
    let w = positive_randvec::<f64>(&mut rng, N);

    let size_hint = eval.size_hint();
    assert_eq!(
        eval.get_names().len(),
        size_hint,
        "names vector has a wrong size"
    );
    assert_eq!(
        eval.get_descriptions().len(),
        size_hint,
        "description vector has a wrong size"
    );
    let check_size =
        |v: &Vec<f64>| assert_eq!(size_hint, v.len(), "size_hint() returns wrong value");

    let baseline = eval.eval(&mut TimeSeries::new(&t_sorted, &m, &w)).unwrap();
    check_size(&baseline);

    if test_ts_length {
        for n in 0..10 {
            eval_info_ts_length_test(&eval, &t_sorted, &m, &w, n)
                .as_ref()
                .map(check_size);
        }
    }

    if test_t_required {
        check_size(&eval_info_t_required_test(
            &eval, &baseline, &t_sorted, &m, &w, &mut rng,
        ));
    }

    if test_m_required {
        check_size(&eval_info_m_required_test(
            &eval, &baseline, &t_sorted, &m, &w, &mut rng,
        ));
    }

    if test_w_required {
        check_size(&eval_info_w_required_test(
            &eval, &baseline, &t_sorted, &m, &w, &mut rng,
        ));
    }

    if test_sorting_required {
        eval_info_sorting_required_test(&eval, &baseline, &t, &m, &w)
            .as_ref()
            .map(check_size);
    }
}

fn eval_info_ts_length_test(
    eval: &Feature<f64>,
    t_sorted: &[f64],
    m: &[f64],
    w: &[f64],
    n: usize,
) -> Option<Vec<f64>> {
    let min_ts_length = eval.min_ts_length();
    let mut ts = TimeSeries::new(&t_sorted[..n], &m[..n], &w[..n]);
    let result = eval.eval(&mut ts);
    assert_eq!(
        n >= min_ts_length,
        result.is_ok(),
        "min_ts_length() returns wrong value, \
                    time-series length: {}, \
                    min_ts_length(): {}, \
                    eval(ts).is_ok(): {}",
        n,
        min_ts_length,
        result.is_ok(),
    );
    result.ok()
}

fn eval_info_t_required_test(
    eval: &Feature<f64>,
    baseline: &[f64],
    t_sorted: &[f64],
    m: &[f64],
    w: &[f64],
    rng: &mut StdRng,
) -> Vec<f64> {
    let t2_sorted = sorted(&randvec::<f64>(rng, t_sorted.len()));
    assert_ne!(t_sorted, t2_sorted);

    let mut ts = TimeSeries::new(&t2_sorted, m, w);

    let v = eval.eval(&mut ts).unwrap();
    let neq_baseline = !simeq(&v, baseline, 1e-12);
    assert_eq!(
        neq_baseline,
        eval.is_t_required(),
        "is_t_required() returns wrong value, \
                    v != baseline: {} ({:?} <=> {:?}), \
                    is_t_required(): {}",
        neq_baseline,
        v,
        baseline,
        eval.is_t_required(),
    );
    v
}

fn eval_info_m_required_test(
    eval: &Feature<f64>,
    baseline: &[f64],
    t_sorted: &[f64],
    m: &[f64],
    w: &[f64],
    rng: &mut StdRng,
) -> Vec<f64> {
    let m2 = randvec::<f64>(rng, m.len());
    assert_ne!(m, m2);

    let mut ts = TimeSeries::new(t_sorted, m2, w);

    let v = eval.eval(&mut ts).unwrap();
    let neq_baseline = !simeq(&v, baseline, 1e-12);
    assert_eq!(
        neq_baseline,
        eval.is_m_required(),
        "is_m_required() returns wrong value, \
                    v != baseline: {} ({:?} <=> {:?}), \
                    is_m_required(): {}",
        neq_baseline,
        v,
        baseline,
        eval.is_m_required(),
    );
    v
}

fn eval_info_w_required_test(
    eval: &Feature<f64>,
    baseline: &[f64],
    t_sorted: &[f64],
    m: &[f64],
    w: &[f64],
    rng: &mut StdRng,
) -> Vec<f64> {
    let w2 = positive_randvec::<f64>(rng, w.len());
    assert_ne!(w, w2);

    let mut ts = TimeSeries::new(t_sorted, m, &w2);
    let v = eval.eval(&mut ts).unwrap();
    let neq_baseline = !simeq(&v, baseline, 1e-12);
    assert_eq!(
        neq_baseline,
        eval.is_w_required(),
        "is_w_required() returns wrong value, \
                    v != baseline: {}, \
                    is_w_required(): {}",
        neq_baseline,
        eval.is_w_required(),
    );
    v
}

fn eval_info_sorting_required_test(
    eval: &Feature<f64>,
    baseline: &[f64],
    t: &[f64],
    m: &[f64],
    w: &[f64],
) -> Option<Vec<f64>> {
    let m_ordered = sorted_by(m, t);
    assert_ne!(m_ordered, m);
    let w_ordered = sorted_by(w, t);
    assert_ne!(w_ordered, w);

    let is_sorting_required = eval.is_sorting_required();

    // FeatureEvaluator is allowed to panic for unsorted input if it requires sorted input
    let v = match (
        std::panic::catch_unwind(|| eval.eval(&mut TimeSeries::new(t, &m_ordered, &w_ordered))),
        is_sorting_required,
    ) {
        (Ok(result), _) => result.unwrap(),
        (Err(_), true) => return None,
        (Err(err), false) => panic!("{err:?}"),
    };

    let neq_baseline = !simeq(&v, baseline, 1e-12);
    assert_eq!(
        neq_baseline, is_sorting_required,
        "is_sorting_required() returns wrong value, \
                    unsorted result: {v:?}, \
                    sorted result: {baseline:?}, \
                    is_sorting_required: {is_sorting_required}",
    );
    Some(v)
}

#[macro_export]
macro_rules! serialization_name_test {
    ($feature_type: ty, $feature_expr: expr_2021) => {
        #[test]
        fn serialization_name() {
            let feature = $feature_expr;
            let actual_name = serde_type_name::type_name(&feature).unwrap();

            let str_type = stringify!($feature_type);
            let desired_name = match str_type.split_once('<') {
                Some((name, _)) => name,
                None => str_type,
            };

            assert_eq!(actual_name, desired_name);
        }
    };
    ($feature_type: ty) => {
        serialization_name_test!($feature_type, <$feature_type>::default());
    };
}

#[macro_export]
macro_rules! serde_json_test {
    ($name: ident, $feature_type: ty, $feature_expr: expr_2021 $(,)?) => {
        #[test]
        fn $name() {
            const N: usize = 128;
            let mut rng = StdRng::seed_from_u64(0);

            let t = sorted(&randvec::<f64>(&mut rng, N));
            let m = randvec::<f64>(&mut rng, N);
            let w = positive_randvec::<f64>(&mut rng, N);

            let eval = $feature_expr;
            let eval_serde: $feature_type =
                serde_json::from_str(&serde_json::to_string(&eval).unwrap()).unwrap();
            assert_eq!(
                eval.eval(&mut TimeSeries::new(&t, &m, &w)),
                eval_serde.eval(&mut TimeSeries::new(&t, &m, &w))
            );

            let feature: Feature<_> = eval.into();
            let feature_serde: Feature<_> =
                serde_json::from_str(&serde_json::to_string(&feature).unwrap()).unwrap();
            assert_eq!(
                feature.eval(&mut TimeSeries::new(&t, &m, &w)),
                feature_serde.eval(&mut TimeSeries::new(&t, &m, &w))
            );
        }
    };
}

#[macro_export]
macro_rules! check_doc_static_method {
    ($name: ident, $feature: ty) => {
        #[test]
        fn $name() {
            const DOC: &'static str = <$feature>::doc();
            assert!(DOC.contains("Depends on: "));
            assert!(DOC.contains("Minimum number of observations: "));
            assert!(DOC.contains("Number of features: "));
        }
    };
}

#[macro_export]
macro_rules! check_finite {
    ($name: ident, $feature_expr: expr_2021 $(,)?) => {
        #[test]
        fn $name() {
            let eval = $feature_expr;
            for (path, triple) in light_curve_feature_test_util::ISSUE_LIGHT_CURVES_ALL_F64
                .iter()
                .map(|(name, mclc)| (name, mclc.clone().into_triple(None)))
                .chain(
                    light_curve_feature_test_util::SNIA_LIGHT_CURVES_FLUX_F64
                        .iter()
                        .map(|(name, mclc)| (name, mclc.clone().into_triple(Some("g"))))
                        .take(10),
                )
                .chain(
                    light_curve_feature_test_util::RRLYR_LIGHT_CURVES_MAG_F64
                        .iter()
                        .map(|(name, mclc)| (name, mclc.clone().into_triple(Some("r"))))
                        .take(10),
                )
            {
                let mut ts: TimeSeries<f64> = triple.into();
                let result = eval.eval(&mut ts);
                assert!(result.is_ok(), "{}", path);
                for (value, feature_name) in result.unwrap().into_iter().zip(eval.get_names()) {
                    assert!(
                        value.is_finite(),
                        "{}: {} is not finite",
                        path,
                        feature_name
                    );
                }
            }
        }
    };
}

#[macro_export]
macro_rules! check_partial_eq {
    ($name: ident, $feature_type: ty, $feature_expr: expr_2021 $(,)?) => {
        #[test]
        fn $name() {
            // Test that two instances with the same parameters are equal
            let feature1 = $feature_expr;
            let feature2 = $feature_expr;
            assert_eq!(
                feature1, feature2,
                "Two instances with same parameters should be equal"
            );

            // Test reflexivity: a == a
            assert_eq!(feature1, feature1, "PartialEq should be reflexive");

            // Test symmetry: if a == b then b == a
            assert_eq!(feature2, feature1, "PartialEq should be symmetric");
        }
    };
    ($feature_type: ty) => {
        check_partial_eq!(partial_eq, $feature_type, <$feature_type>::default());
    };
}

#[macro_export]
macro_rules! check_feature {
    ($feature: ty) => {
        eval_info_test!(info_default, <$feature>::default());
        serialization_name_test!($feature);
        serde_json_test!(ser_json_de, $feature, <$feature>::default());
        check_doc_static_method!(doc_static_method, $feature);
        check_finite!(check_values_finite, <$feature>::default());
        check_partial_eq!($feature);
    };
}

#[macro_export]
macro_rules! check_fit_model_derivatives {
    ($feature: ty) => {
        #[test]
        fn fit_derivatives() {
            const REPEAT: usize = 10;

            let mut rng = StdRng::seed_from_u64(0);
            for _ in 0..REPEAT {
                let t = 10.0 * rng.random::<f64>();

                let param = {
                    let mut param = [0.0; NPARAMS];
                    for x in param.iter_mut() {
                        *x = rng.random::<f64>() - 0.5;
                    }
                    param
                };
                let actual = {
                    let mut jac = [0.0; NPARAMS];
                    <$feature>::derivatives(t, &param, &mut jac);
                    jac
                };

                let desired: Vec<_> = {
                    let hyper_param = {
                        let mut hyper =
                            [Hyperdual::<f64, { NPARAMS + 1 }>::from_real(0.0); NPARAMS];
                        for (i, (x, h)) in param.iter().zip(hyper.iter_mut()).enumerate() {
                            h[0] = *x;
                            h[i + 1] = 1.0;
                        }
                        hyper
                    };
                    let result = <$feature>::model(t, &hyper_param);
                    (1..=NPARAMS).map(|i| result[i]).collect()
                };

                assert_relative_eq!(&actual[..], &desired[..], epsilon = 1e-9);
            }
        }
    };
}

#[macro_export]
macro_rules! transformer_check_doc_static_method {
    ($name: ident, $transformer: ty) => {
        #[test]
        fn $name() {
            let doc = <$transformer>::doc();
            assert!(doc.len() > 10);
        }
    };
}

#[macro_export]
macro_rules! transformer_check_size_hint {
    ($name: ident, $transformer: expr_2021, $type: ty) => {
        #[test]
        fn $name() {
            let transformer = $transformer;
            const SIZE_RANGE: std::ops::RangeInclusive<usize> = 0..=20;
            let valid_sizes_count = SIZE_RANGE
                .filter(|&feature_size| transformer.is_size_valid(feature_size))
                .map(|feature_size| {
                    let transformer_size_hint = transformer.size_hint(feature_size);

                    // Check output size
                    let input = vec![1.0; feature_size];
                    let output = transformer.transform(input);
                    assert_eq!(output.len(), transformer_size_hint);

                    // Check names size
                    let input_names = vec!["XXX"; feature_size];
                    let output_names = transformer.names(&input_names);
                    assert_eq!(output_names.len(), transformer_size_hint);

                    // Check descriptions size
                    let input_descriptions = vec!["XXX ###"; feature_size];
                    let output_descriptions = transformer.descriptions(&input_descriptions);
                    assert_eq!(output_descriptions.len(), transformer_size_hint);
                })
                .count();
            assert!(
                valid_sizes_count > 0,
                "No valid sizes for transformer {} in range {:?}",
                stringify!($type),
                SIZE_RANGE
            );
        }
    };
}

#[macro_export]
macro_rules! check_transformer {
    ($transformer: ty) => {
        transformer_check_doc_static_method!(check_doc_static_method, $transformer);
        transformer_check_size_hint!(check_size_hint, <$transformer>::default(), $transformer);
    };
}

pub fn simeq<T: Float>(a: &[T], b: &[T], eps: T) -> bool {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b.iter())
        .all(|(&x, &y)| (x - y).abs() < eps + T::max(x.abs(), y.abs()) * eps)
}

pub fn randvec<T>(rng: &mut StdRng, n: usize) -> Vec<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    (0..n)
        .map(|_| {
            let x: T = rng.sample(StandardNormal);
            x
        })
        .collect()
}

pub fn positive_randvec<T>(rng: &mut StdRng, n: usize) -> Vec<T>
where
    T: Float,
    StandardNormal: Distribution<T>,
{
    let mut v = randvec(rng, n);
    v.iter_mut().for_each(|x| *x = x.abs());
    v
}

pub fn sorted<T>(v: &[T]) -> Vec<T>
where
    T: Float,
{
    let mut v = v.to_vec();
    v[..].sort_by(|a, b| a.partial_cmp(b).unwrap());
    v
}

pub fn sorted_by<T: Float>(to_sort: &[T], key: &[T]) -> Vec<T> {
    assert_eq!(to_sort.len(), key.len());
    let mut idx: Vec<_> = (0..to_sort.len()).collect();
    idx[..].sort_by(|&a, &b| key[a].partial_cmp(&key[b]).unwrap());
    idx.iter().map(|&i| to_sort[i]).collect()
}

// Some tests validating tests

#[test]
fn test_data_non_empty() {
    assert!(!light_curve_feature_test_util::ISSUE_LIGHT_CURVES_FLUX_F64.is_empty());
    assert!(!light_curve_feature_test_util::ISSUE_LIGHT_CURVES_MAG_F64.is_empty());
    assert!(!light_curve_feature_test_util::ISSUE_LIGHT_CURVES_ALL_F64.is_empty());
    assert!(!light_curve_feature_test_util::SNIA_LIGHT_CURVES_FLUX_F64.is_empty());
    assert!(!light_curve_feature_test_util::ALL_LIGHT_CURVES_F64.is_empty());
}
