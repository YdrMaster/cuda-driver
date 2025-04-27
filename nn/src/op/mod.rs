use crate::{Arg, TensorMeta};

/// 计算图层算子，只考虑形状推导
pub trait Operator {
    fn infer(&self, inputs: &[TensorMeta], arg: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError>;
}

#[derive(Clone, Copy, Debug)]
pub enum OpError {
    NotExist,
    DataTypeError,
    DataTypeMismatch,
    ShapeError,
    ShapeMismatch,
}
