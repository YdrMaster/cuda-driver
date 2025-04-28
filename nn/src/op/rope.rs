use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct Rope;

impl Operator for Rope {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }

        match inputs {
            [x, pos, sin, cos] => {
                dims!([n, d] = x);

                // dims!([s_] = pos);
                // dims!([nctx, hd_half] = sin);
                // dims!([nctx_, hd_half_] = cos);

                // TODO 需要判断相等
                // assert_eq!(s, s_);
                // assert_eq!(hd / 2, hd_half);
                // assert_eq!(nctx, nctx_);
                // assert_eq!(hd_half, hd_half_);

                // 输出形状与输入相同
                Ok(vec![TensorMeta::new(x.dt, [n.clone(), d.clone()])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
