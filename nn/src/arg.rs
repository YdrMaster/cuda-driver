use crate::Dim;
use std::collections::HashMap;

/// 神经网络标量参数
#[derive(Clone)]
pub enum Arg {
    Dim(Dim),
    Float(f64),
    Arr(Box<[Arg]>),
    Dict(HashMap<String, Arg>),
}

impl From<Dim> for Arg {
    fn from(value: Dim) -> Self {
        Self::Dim(value)
    }
}

impl From<f64> for Arg {
    fn from(value: f64) -> Self {
        Self::Float(value)
    }
}

impl From<Box<[Arg]>> for Arg {
    fn from(value: Box<[Arg]>) -> Self {
        Self::Arr(value)
    }
}

impl From<HashMap<String, Arg>> for Arg {
    fn from(value: HashMap<String, Arg>) -> Self {
        Self::Dict(value)
    }
}

impl Arg {
    pub fn int(value: usize) -> Self {
        Self::Dim(Dim::Constant(value))
    }

    pub fn float(value: f64) -> Self {
        Self::Float(value)
    }

    pub fn arr(value: impl IntoIterator<Item = Arg>) -> Self {
        Self::Arr(value.into_iter().collect())
    }

    pub fn dict(value: impl Into<HashMap<String, Arg>>) -> Self {
        Self::Dict(value.into())
    }
}
