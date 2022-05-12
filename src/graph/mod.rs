use crate::{Float, TimeSeries};

use enum_dispatch::enum_dispatch;
use ndarray_stats::QuantileExt;
use petgraph::{graphmap::GraphMap, Directed};
use std::fmt::Debug;
use std::hash::Hash;

pub struct FeatureGraph {
    graph: GraphMap<Node, f64, Directed>,
}

impl FeatureGraph {
    pub fn new<I>(features: I) -> Self
    where
        I: IntoIterator<Item = Feature>,
    {
        let mut graph = GraphMap::new();
        for feature in features {
            for (extractor, weight) in feature.parents() {
                graph.add_edge(feature.into(), extractor.into(), weight);
                // TODO: extractor's children
                // while let Node::Extractor(parent) = extractor.parent() {
                //     graph.add_edge(extractor.into(), parent.into(), 1.0); // TODO: weight
                //     extractor = parent;
                // }
            }
        }
        Self { graph }
    }
}

#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub enum Node {
    Root(Root),
    Extractor(Extractor),
    Feature(Feature),
}

impl From<Root> for Node {
    fn from(v: Root) -> Self {
        Self::Root(v)
    }
}

impl From<Extractor> for Node {
    fn from(v: Extractor) -> Self {
        Self::Extractor(v)
    }
}

impl From<Feature> for Node {
    fn from(v: Feature) -> Self {
        Self::Feature(v)
    }
}

pub trait ParentNodeTrait<'o, T> {
    type Output;
}

#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct Root {}

impl<'o, T: Float> ParentNodeTrait<'o, T> for Root {
    type Output = TimeSeries<'o, T>;
}

pub trait ExtractorTrait<'i, 'o, T>:
    ParentNodeTrait<'o, T> + Debug + Copy + Hash + Ord + Into<Node>
{
    type Parent: ParentNodeTrait<'i, T>;

    fn parent(&self) -> Self::Parent;

    fn children(&self) -> Vec<Node> {
        vec![]
    }

    fn extract(
        &self,
        ts: <<Self as ExtractorTrait<'i, 'o, T>>::Parent as ParentNodeTrait<'i, T>>::Output,
    ) -> Self::Output;
}

#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
#[non_exhaustive]
pub enum Extractor {
    MinMUnsorted(MinMUnsorted),
}

impl From<MinMUnsorted> for Extractor {
    fn from(v: MinMUnsorted) -> Self {
        Self::MinMUnsorted(v)
    }
}

#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct MinMUnsorted {}

impl<T> ParentNodeTrait<'static, T> for MinMUnsorted {
    type Output = T;
}

impl<'i, T: Float> ExtractorTrait<'i, 'static, T> for MinMUnsorted {
    type Parent = Root;

    fn parent(&self) -> Self::Parent {
        Root {}
    }

    fn extract(&self, ts: TimeSeries<'i, T>) -> Self::Output {
        *ts.m.sample.min().unwrap()
    }
}

impl From<MinMUnsorted> for Node {
    fn from(v: MinMUnsorted) -> Self {
        let extractor: Extractor = v.into();
        extractor.into()
    }
}

#[enum_dispatch]
pub trait FeatureTrait: Debug + Copy + Hash + Ord + Into<Node> {
    fn parents(&self) -> Vec<(Extractor, f64)>;
}

#[enum_dispatch(FeatureTrait)]
#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
#[non_exhaustive]
pub enum Feature {
    MinM,
}

#[derive(Debug, Copy, Clone, Hash, Ord, PartialOrd, Eq, PartialEq)]
pub struct MinM {}

impl FeatureTrait for MinM {
    fn parents(&self) -> Vec<(Extractor, f64)> {
        vec![(MinMUnsorted {}.into(), 1.0)]
    }
}

impl From<MinM> for Node {
    fn from(v: MinM) -> Self {
        let feature: Feature = v.into();
        feature.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use petgraph::dot::{Config, Dot};

    #[test]
    fn graph() {
        let graph = FeatureGraph::new([MinM {}.into()]);
        println!("{:?}", Dot::with_config(&graph.graph, &[]));
    }
}
