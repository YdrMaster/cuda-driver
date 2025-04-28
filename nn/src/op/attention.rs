use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct Attention;

impl Operator for Attention {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let dh = if let Some(Arg::Dim(dh)) = args {
            dh
        } else {
            return Err(OpError::ArgError);
        };

        match inputs {
            [q, k, v] => {
                dims!([n, dq] = q);
                dims!([n_1, dk] = k);
                dims!([n_2, dv] = v);

                // TODO 需要判断相等
                // assert_eq!(dk, dv);
                // assert_eq!(n, n_1, n_2);

                Ok(vec![TensorMeta::new(q.dt, [n.clone(), dh.clone()])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
