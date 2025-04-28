use super::{OpError, Operator, macros::*};
use crate::{Arg, TensorMeta};

pub struct Embedding;

impl Operator for Embedding {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        if args.is_some() {
            return Err(OpError::ArgError);
        }
        match inputs {
            [wte, tokens] => {
                dims!([_, d] = wte);
                dims!([n] = tokens);
                Ok(vec![TensorMeta::new(wte.dt, [n.clone(), d.clone()])])
            }
            [wte, tokens, wpe, pos] => {
                dims!([_, d] = wte);
                dims!([n] = tokens);
                dims!([_, d_] = wpe);
                dims!([n_] = pos);

                // TODO 需要判断相等
                // assert_eq!(d, d_);
                // assert_eq!(n, n_);
                Ok(vec![TensorMeta::new(wte.dt, [n.clone(), d.clone()])])
            }
            _ => Err(OpError::ShapeError),
        }
    }
}
