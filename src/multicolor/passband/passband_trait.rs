use schemars::JsonSchema;
use serde::Serialize;
use std::fmt::Debug;

pub trait PassbandTrait: Debug + Clone + Send + Sync + Ord + Serialize + JsonSchema {
    fn name(&self) -> &str;
}
