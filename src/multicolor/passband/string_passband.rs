use crate::PassbandTrait;

pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};

/// A passband identified by a string name.
///
/// Useful when passband information is available only as a label (e.g. "r", "g", "i").
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct StringPassband(pub String);

impl PassbandTrait for StringPassband {
    fn name(&self) -> &str {
        &self.0
    }
}

impl From<&str> for StringPassband {
    fn from(s: &str) -> Self {
        Self(s.to_owned())
    }
}

impl From<String> for StringPassband {
    fn from(s: String) -> Self {
        Self(s)
    }
}
