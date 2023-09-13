use crate::transformers::transformer::*;

use thiserror::Error;

macro_const! {
    const DOC: &str = r#"
Transformer composed from a list of transformers

The transformers are stacked in the order they are given in the list, with number of
features per trasnformer specified.
"#;
}

#[derive(Error, Debug)]
pub enum ComposedTransformerConstructionError {
    #[error("Size mismatch between transformer size requirements and given feature size")]
    SizeMismatch,
}

#[doc = DOC!()]
#[derive(Clone, Debug, Serialize, Deserialize, JsonSchema)]
pub struct ComposedTransformer<Tr> {
    transformers: Vec<(Tr, usize)>,
    input_size: usize,
    size_hint: usize,
}

impl<Tr> ComposedTransformer<Tr>
where
    Tr: TransformerPropsTrait,
{
    /// Create a new composed transformer from a list of transformers and the number of features
    /// they take as input.
    pub fn new(
        transformers: impl Into<Vec<(Tr, usize)>>,
    ) -> Result<Self, ComposedTransformerConstructionError> {
        let transformers = transformers.into();
        let mut input_size = 0;
        let mut size_hint = 0;
        for (tr, size) in &transformers {
            if !tr.is_size_valid(*size) {
                return Err(ComposedTransformerConstructionError::SizeMismatch);
            }
            input_size += *size;
            size_hint += tr.size_hint(*size);
        }
        Ok(Self {
            transformers,
            input_size,
            size_hint,
        })
    }

    /// Create a new composed transformer from a list of transformers assumed that all of them
    /// may take a single feature as input.
    pub fn from_transformers(
        transformers: impl IntoIterator<Item = Tr>,
    ) -> Result<Self, ComposedTransformerConstructionError> {
        let transformers = transformers
            .into_iter()
            .map(|tr| (tr, 1))
            .collect::<Vec<_>>();
        Self::new(transformers)
    }
}

impl<Tr> ComposedTransformer<Tr> {
    pub const fn doc() -> &'static str {
        DOC
    }
}

impl<Tr> TransformerPropsTrait for ComposedTransformer<Tr>
where
    Tr: TransformerPropsTrait,
{
    fn is_size_valid(&self, input_size: usize) -> bool {
        self.input_size == input_size
    }

    fn size_hint(&self, _input_size: usize) -> usize {
        self.size_hint
    }

    fn names(&self, input_names: &[&str]) -> Vec<String> {
        let mut names_iter = input_names.iter();
        self.transformers
            .iter()
            .flat_map(|(tr, size)| {
                let names_batch = names_iter.by_ref().take(*size).copied().collect::<Vec<_>>();
                tr.names(&names_batch[..]).into_iter()
            })
            .collect()
    }

    fn descriptions(&self, input_descriptions: &[&str]) -> Vec<String> {
        let mut desc_iter = input_descriptions.iter();
        self.transformers
            .iter()
            .flat_map(|(tr, size)| {
                let desc_batch = desc_iter.by_ref().take(*size).copied().collect::<Vec<_>>();
                tr.descriptions(&desc_batch[..]).into_iter()
            })
            .collect()
    }
}

impl<T, Tr> TransformerTrait<T> for ComposedTransformer<Tr>
where
    T: Float,
    Tr: TransformerTrait<T>,
{
    fn transform(&self, input: Vec<T>) -> Vec<T> {
        let mut input_iter = input.into_iter();
        self.transformers
            .iter()
            .flat_map(|(tr, size)| {
                let input_batch = input_iter.by_ref().take(*size).collect::<Vec<_>>();
                tr.transform(input_batch)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transformers::bazin_fit::BazinFitTransformer;
    use crate::transformers::clipped_lg::ClippedLgTransformer;
    use crate::transformers::identity::IdentityTransformer;
    use crate::transformers::linexp_fit::LinexpFitTransformer;

    transformer_check_doc_static_method!(
        check_doc_static_method,
        ComposedTransformer::<Transformer<f32>>
    );
    transformer_check_size_hint!(
        check_size_hint,
        ComposedTransformer::<Transformer<f32>>::new([
            (IdentityTransformer::new().into(), 3),
            (ClippedLgTransformer::default().into(), 2),
            (BazinFitTransformer::default().into(), 6),
            (LinexpFitTransformer::default().into(), 5),
        ])
        .unwrap(),
        ComposedTransformer::<Transformer<f32>>
    );

    #[test]
    fn test_from_transformers() {
        let tr = ComposedTransformer::from_transformers([
            IdentityTransformer::new(),
            IdentityTransformer::new(),
        ])
        .unwrap();
        assert_eq!(tr.transformers.len(), 2);
        assert_eq!(tr.input_size, 2);
        assert_eq!(tr.size_hint, 2);
        assert_eq!(tr.transformers[0].1, 1);
        assert_eq!(tr.transformers[1].1, 1);
    }

    #[test]
    fn test_from_transformers_size_mismatch() {
        let result = ComposedTransformer::<Transformer<f32>>::from_transformers([
            IdentityTransformer::new().into(),
            BazinFitTransformer::default().into(), // requires more than one feature
            LinexpFitTransformer::default().into(), // requires more than one feature
        ]);
        assert!(result.is_err());
    }

    #[test]
    fn test_new() {
        let tr = ComposedTransformer::<Transformer<f32>>::new([
            (IdentityTransformer::new().into(), 2),
            (BazinFitTransformer::default().into(), 6),
            (LinexpFitTransformer::default().into(), 5),
        ])
        .unwrap();
        assert_eq!(tr.transformers.len(), 3);
        assert_eq!(tr.input_size, 13);
        assert_eq!(tr.size_hint, 11);
        assert_eq!(tr.transformers[0].1, 2);
        assert_eq!(tr.transformers[1].1, 6);
        assert_eq!(tr.transformers[2].1, 5);
    }

    #[test]
    fn test_new_size_mismatch() {
        let result = ComposedTransformer::<Transformer<f32>>::new([
            (IdentityTransformer::new().into(), 3),
            (BazinFitTransformer::default().into(), 3), // requires six features
            (LinexpFitTransformer::default().into(), 3), // requires five features
        ]);
        assert!(result.is_err());
    }
}
