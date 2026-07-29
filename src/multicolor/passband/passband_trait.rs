use schemars::JsonSchema;
use serde::Serialize;
use std::fmt::Debug;

pub trait PassbandTrait: Debug + Clone + Send + Sync + Ord + Serialize + JsonSchema {
    fn name(&self) -> &str;

    /// Effective wavelength of the passband, in cm, if known.
    ///
    /// Defaults to `None` so existing implementors (e.g. [StringPassband](super::StringPassband))
    /// are unaffected; [MonochromePassband](super::MonochromePassband) overrides it.
    /// Wavelength-aware features like `RainbowFit` reject passbands that return `None`.
    fn wavelength(&self) -> Option<f64> {
        None
    }
}
