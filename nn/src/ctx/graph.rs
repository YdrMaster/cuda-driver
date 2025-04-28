use super::TensorMeta;
use crate::Arg;

pub struct Node {
    pub name: String,
    pub op: String,
    pub arg: Option<Arg>,
}

pub struct Edge<T> {
    pub meta: TensorMeta,
    pub weight_info: Option<WeightInfo<T>>,
}

pub struct WeightInfo<T> {
    pub name: String,
    pub item: T,
}
