use schemars::JsonSchema;
use serde::Serialize;
use std::fmt::Debug;

pub trait PassbandTrait: Debug + Clone + Send + Sync + Ord + Serialize + JsonSchema {
    fn name(&self) -> &str;

    /// Effective wavelength of the passband, in cm, if known.
    ///
    /// Defaults to `None` so existing implementors (e.g. [StringPassband](super::StringPassband),
    /// which only carries a label) are unaffected. Wavelength-aware features such as `RainbowFit`
    /// require it and reject passbands that return `None` at evaluation time, the same way any
    /// other feature rejects a passband it can't use. [MonochromePassband](super::MonochromePassband)
    /// overrides this to return its wavelength.
    fn wavelength(&self) -> Option<f64> {
        None
    }
}
