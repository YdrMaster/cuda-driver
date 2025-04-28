use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct Linear;

impl Operator for Linear {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }
        match inputs {
            [x, w] => {
                dims!([n, d] = x);
                dims!([m, d_] = w);
                Ok(vec![TensorMeta::new(x.dt, [n.clone(), m.clone()])])
            }
            [x, w, b] => {
                dims!([n, d] = x);
                dims!([m, d_] = w);
                dims!([m_] = b);

                // TODO 需要判断相等

                Ok(vec![TensorMeta::new(x.dt, [n.clone(), m.clone()])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
