use crate::PassbandTrait;

pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};
use std::fmt::Debug;

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct DumpPassband {}

impl PassbandTrait for DumpPassband {
    fn name(&self) -> &str {
        ""
    }
}
