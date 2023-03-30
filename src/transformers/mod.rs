use transformer::*;
pub use transformer::{Transformer, TransformerPropsTrait, TransformerTrait};

use paste::paste;

pub mod bazin_fit;
pub mod clipped_lg;
pub mod composed;
pub mod transformer;
pub mod villar_fit;

macro_rules! transformer {
    ($module:ident, $structure:ident, $func:expr, $names:expr, $descriptions:expr, $doc:literal $(,)?) => {
        pub mod $module {
            use super::*;

            #[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
            pub struct $structure {}

            impl $structure {
                pub fn new() -> Self {
                    Self {}
                }

                pub const fn doc() -> &'static str {
                    $doc
                }
            }

            impl Default for $structure {
                fn default() -> Self {
                    Self::new()
                }
            }

            impl TransformerPropsTrait for $structure {
                #[inline]
                fn is_size_valid(&self, _size: usize) -> bool {
                    true
                }

                #[inline]
                fn size_hint(&self, size: usize) -> usize {
                    size
                }

                fn names(&self, names: &[&str]) -> Vec<String> {
                    let func = $names;
                    func(names)
                }

                fn descriptions(&self, desc: &[&str]) -> Vec<String> {
                    let func = $descriptions;
                    func(desc)
                }
            }

            impl<T: Float> TransformerTrait<T> for $structure {
                #[inline]
                fn transform(&self, v: Vec<T>) -> Vec<T> {
                    let func = $func;
                    func(v)
                }
            }

            #[cfg(test)]
            mod tests {
                use super::*;

                check_transformer!($structure);
            }
        }
    };
}

// into_iter().map().collect() should create no allocations
macro_rules! transformer_from_per_element_fn {
    ($module:ident, $func:expr, $prefix_name:literal, $prefix_description:literal, $doc:literal $(,)?) => {
        paste! {
            transformer!(
                $module,
                [<$module:camel Transformer>],
                |v: Vec<T>| v.into_iter().map($func).collect::<Vec<T>>(),
                |names: &[&str]| {
                    names
                        .iter()
                        .map(|name| format!("{}_{}", $prefix_name, name))
                        .collect::<Vec<_>>()
                },
                |desc: &[&str]| {
                    desc.iter()
                        .map(|name| format!("{} of {}", $prefix_description, name))
                        .collect::<Vec<_>>()
                },
                $doc
            );
        }
    };
}

transformer!(
    identity,
    IdentityTransformer,
    |x| x,
    |names: &[&str]| names.iter().map(|x| x.to_string()).collect(),
    |desc: &[&str]| desc.iter().map(|x| x.to_string()).collect(),
    "Identity feature transformer",
);

transformer_from_per_element_fn!(
    arcsinh,
    T::asinh,
    "arcsinh",
    "hyperbolic arcsine",
    "Hyperbolic arcsine feature transformer",
);
transformer_from_per_element_fn!(
    ln1p,
    T::ln_1p,
    "ln1p",
    "natural logarithm of unity plus",
    "ln(1+x) feature transformer",
);
transformer_from_per_element_fn!(
    lg,
    T::log10,
    "lg",
    "decimal logarithm",
    "Decimal logarithm feature transformer"
);
transformer_from_per_element_fn!(
    sqrt,
    T::sqrt,
    "sqrt",
    "square root",
    "Square root feature transformer"
);
