pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub trait PassbandTrait: Debug + Clone + Send + Sync + Ord + Serialize + JsonSchema {
    fn name(&self) -> &str;
}
