use ndarray::{ArrayRef, CowArray, Ix1};

pub type ArrayRef1<T> = ArrayRef<T, Ix1>;
pub type CowArray1<'a, T> = CowArray<'a, T, Ix1>;
