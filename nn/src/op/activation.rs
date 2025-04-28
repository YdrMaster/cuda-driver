use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct SwiGLU;
pub struct GeLU;

impl Operator for SwiGLU {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        destruct!([gate, up] = inputs);
        dims!([n, d] = gate);
        dims!([n_, d_] = up);

        // TODO 需要判断相等

        Ok(vec![TensorMeta::new(up.dt, [n.clone(), d.clone()])])
    }
}

impl Operator for GeLU {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        destruct!([x] = inputs);
        dims!([n, d] = x);

        Ok(vec![TensorMeta::new(x.dt, [n.clone(), d.clone()])])
    }
}
