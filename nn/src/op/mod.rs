use crate::{Arg, TensorMeta};

pub mod activation;
pub mod attention;
pub mod embedding;
pub mod linear;
pub mod normalization;
pub mod rope;
pub mod split;

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
    ArgError,
}

pub mod macros {
    macro_rules! destruct {
        ([$( $name:ident ),+] = $iter:expr) => {
            let mut iter = $iter.into_iter();
            $( let $name = iter.next().ok_or(OpError::ShapeError)?; )+
            if iter.next().is_some() {
                return Err(OpError::ShapeError);
            }
        };
    }

    macro_rules! dims {
        ($pat:pat = $tensor:expr) => {
            let $pat = &*$tensor.shape() else {
                return Err(OpError::ShapeError);
            };
        };
    }

    pub(crate) use {destruct, dims};
}
