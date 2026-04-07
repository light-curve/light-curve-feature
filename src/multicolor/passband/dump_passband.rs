use crate::PassbandTrait;

pub use schemars::JsonSchema;
pub use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A passband for the cases where we don't care about the actual passband.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, JsonSchema)]
pub struct DumpPassband {}

impl PassbandTrait for DumpPassband {
    fn name(&self) -> &str {
        ""
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dump_passband() {
        let passband = DumpPassband {};
        assert_eq!(passband.name(), "");
    }
}
