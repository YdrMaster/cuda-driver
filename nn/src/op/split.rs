use super::{OpError, Operator, macros::*};
use crate::{Arg, Dim, TensorMeta};
use std::collections::HashMap;

pub struct Split;

impl Operator for Split {
    fn infer(&self, inputs: &[TensorMeta], args: Option<&Arg>) -> Result<Vec<TensorMeta>, OpError> {
        let (axis, parts) = if let Arg::Dict(args) = args.ok_or(OpError::ArgError)? {
            let axis = args.get("axis").ok_or(OpError::ArgError)?;
            let parts = args.get("parts").ok_or(OpError::ArgError)?;
            let axis = if let Arg::Dim(Dim::Constant(axis)) = axis {
                axis
            } else {
                return Err(OpError::ArgError);
            };
            let parts = if let Arg::Arr(parts) = parts {
                parts
                    .iter()
                    .map(|p| {
                        if let Arg::Dim(dim) = p {
                            Ok(dim.clone())
                        } else {
                            Err(OpError::ArgError)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()?
            } else {
                return Err(OpError::ArgError);
            };
            (axis, parts)
        } else {
            return Err(OpError::ArgError);
        };

        destruct!([x] = inputs);

        let shape = x.shape();

        if *axis >= shape.len() {
            return Err(OpError::ShapeError);
        }

        let sum = parts
            .iter()
            .fold(Dim::Constant(0), |acc, p| acc + p.clone());

        let c = shape[*axis].clone() / sum;
        //TODO 需要检查parts的和是否等于shape[axis]
        Ok(parts
            .into_iter()
            .map(|p| {
                let mut shape = shape.to_vec();
                shape[*axis] = p * c.clone();
                TensorMeta::new(x.dt, shape)
            })
            .collect::<Vec<_>>())
    }
}
