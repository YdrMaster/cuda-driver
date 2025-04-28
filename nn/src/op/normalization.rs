use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct RmsNorm;
pub struct LayerNorm;

impl Operator for RmsNorm {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let _epsilon = args.ok_or(OpError::ArgError)?;
        // epsilon是浮点数

        match inputs {
            [x, scale] => {
                dims!([n, d] = x);
                dims!([d_] = scale);

                // TODO 检查最后一维是否等于scale的维度
                // assert_eq!(shape[shape.len() - 1], d);

                // 输出形状与输入相同
                Ok(vec![TensorMeta::new(x.dt, [n.clone(), d.clone()])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}

impl Operator for LayerNorm {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let _epsilon = args.ok_or(OpError::ArgError)?;
        // epsilon是浮点数

        match inputs {
            [x, scale, bias] => {
                dims!([n, d] = x);
                dims!([d_1] = scale);
                dims!([d_2] = bias);

                // TODO 检查维度
                // assert_eq!(shape[shape.len() - 1], d);
                // assert_eq!(d, d_);

                // 输出形状与输入相同
                Ok(vec![TensorMeta::new(x.dt, [n.clone(), d.clone()])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
