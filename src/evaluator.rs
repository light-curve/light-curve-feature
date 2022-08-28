pub use crate::data::TimeSeries;
pub use crate::error::EvaluatorError;
pub use crate::float_trait::Float;

pub use conv::errors::GeneralError;
use enum_dispatch::enum_dispatch;
pub use lazy_static::lazy_static;
pub use macro_const::macro_const;
use ndarray::Array1;
pub use schemars::JsonSchema;
use serde::de::DeserializeOwned;
pub use serde::{Deserialize, Serialize};
pub use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
pub struct EvaluatorInfo {
    pub size: usize,
    pub min_ts_length: usize,
    pub t_required: bool,
    pub m_required: bool,
    pub w_required: bool,
    pub sorting_required: bool,
    pub variability_required: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct EvaluatorProperties {
    pub info: EvaluatorInfo,
    pub names: Vec<String>,
    pub descriptions: Vec<String>,
}

// pub trait EvaluatorPropertiesTrait {
//     fn get_properties(&self) -> &EvaluatorProperties;
// }

#[enum_dispatch]
pub trait EvaluatorInfoTrait {
    /// Get feature evaluator meta-information
    fn get_info(&self) -> &EvaluatorInfo;

    /// Size of vectors returned by [eval()](FeatureEvaluator::eval),
    /// [get_names()](FeatureEvaluator::get_names) and
    /// [get_descriptions()](FeatureEvaluator::get_descriptions)
    fn size_hint(&self) -> usize {
        self.get_info().size
    }

    /// Minimum time series length required to successfully evaluate feature
    fn min_ts_length(&self) -> usize {
        self.get_info().min_ts_length
    }

    /// If time array used by the feature
    fn is_t_required(&self) -> bool {
        self.get_info().t_required
    }

    /// If magnitude array is used by the feature
    fn is_m_required(&self) -> bool {
        self.get_info().m_required
    }

    /// If weight array is used by the feature
    fn is_w_required(&self) -> bool {
        self.get_info().w_required
    }

    /// If feature requires time-sorting on the input [TimeSeries]
    fn is_sorting_required(&self) -> bool {
        self.get_info().sorting_required
    }

    /// If feature requires magnitude array elements to be different
    fn is_variability_required(&self) -> bool {
        self.get_info().variability_required
    }
}

// impl<P> EvaluatorInfoTrait for P
// where
//     P: EvaluatorPropertiesTrait,
// {
//     fn get_info(&self) -> &EvaluatorInfo {
//         &self.get_properties().info
//     }
// }

#[enum_dispatch]
pub trait FeatureNamesDescriptionsTrait {
    /// Vector of feature names. The length and feature order corresponds to
    /// [eval()](FeatureEvaluator::eval) output
    fn get_names(&self) -> Vec<&str>;

    /// Vector of feature descriptions. The length and feature order corresponds to
    /// [eval()](FeatureEvaluator::eval) output
    fn get_descriptions(&self) -> Vec<&str>;
}

// impl<P> FeatureNamesDescriptionsTrait for P
// where
//     P: EvaluatorPropertiesTrait,
// {
//     fn get_names(&self) -> Vec<&str> {
//         self.get_properties()
//             .names
//             .iter()
//             .map(|name| name.as_str())
//             .collect()
//     }
//
//     fn get_descriptions(&self) -> Vec<&str> {
//         self.get_properties()
//             .descriptions
//             .iter()
//             .map(|descr| descr.as_str())
//             .collect()
//     }
// }

/// The trait each feature should implement
#[enum_dispatch]
pub trait FeatureEvaluator<T: Float>:
    FeatureNamesDescriptionsTrait
    + EvaluatorInfoTrait
    + Send
    + Clone
    + Debug
    + Serialize
    + DeserializeOwned
    + JsonSchema
{
    /// Version of [FeatureEvaluator::eval] which can panic for incorrect input
    fn eval_no_ts_check(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError>;

    /// Vector of feature values or `EvaluatorError`
    fn eval(&self, ts: &mut TimeSeries<T>) -> Result<Vec<T>, EvaluatorError> {
        self.check_ts(ts)?;
        self.eval_no_ts_check(ts)
    }

    /// Returns vector of feature values and fill invalid components with given value
    fn eval_or_fill(&self, ts: &mut TimeSeries<T>, fill_value: T) -> Vec<T> {
        match self.eval(ts) {
            Ok(v) => v,
            Err(_) => vec![fill_value; self.size_hint()],
        }
    }

    fn check_ts(&self, ts: &mut TimeSeries<T>) -> Result<(), EvaluatorError> {
        self.check_ts_length(ts)?;
        self.check_ts_variability(ts)
    }

    /// Checks if [TimeSeries] has enough points to evaluate the feature
    fn check_ts_length(&self, ts: &TimeSeries<T>) -> Result<(), EvaluatorError> {
        let length = ts.lenu();
        if length < self.min_ts_length() {
            Err(EvaluatorError::ShortTimeSeries {
                actual: length,
                minimum: self.min_ts_length(),
            })
        } else {
            Ok(())
        }
    }

    /// Checks if [TimeSeries] meets variability requirement
    fn check_ts_variability(&self, ts: &mut TimeSeries<T>) -> Result<(), EvaluatorError> {
        if self.is_variability_required() && ts.is_plateau() {
            Err(EvaluatorError::FlatTimeSeries)
        } else {
            Ok(())
        }
    }
}

pub trait OwnedArrays<T>
where
    T: Float,
{
    fn ts(self) -> TimeSeries<'static, T>;
}

pub struct TmArrays<T> {
    pub t: Array1<T>,
    pub m: Array1<T>,
}

impl<T> OwnedArrays<T> for TmArrays<T>
where
    T: Float,
{
    fn ts(self) -> TimeSeries<'static, T> {
        TimeSeries::new_without_weight(self.t, self.m)
    }
}

pub struct TmwArrays<T> {
    pub t: Array1<T>,
    pub m: Array1<T>,
    pub w: Array1<T>,
}

impl<T> OwnedArrays<T> for TmwArrays<T>
where
    T: Float,
{
    fn ts(self) -> TimeSeries<'static, T> {
        TimeSeries::new(self.t, self.m, self.w)
    }
}
