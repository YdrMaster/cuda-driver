use super::{NNCtx, NNError, NuralNetwork, Tensor, macros::*};
use crate::Arg;

#[derive(Clone, Copy)]
pub enum Activation {
    SwiGLU,
    GeLU,
}

impl<T> NuralNetwork<T> for Activation {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: NNCtx<T>,
    ) -> Result<(NNCtx<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x] = inputs);
        dims!([_, d] = x);

        let outputs = match self {
            Self::SwiGLU => {
                let d = d.clone() / 2;
                destruct!(
                    [gate, up] = ctx.call(
                        "split-gate-up",
                        "split",
                        Some(Arg::dict([
                            ("axis".into(), Arg::int(1)),
                            ("parts".into(), Arg::arr([d.clone(), d].map(Arg::from))),
                        ])),
                        [x],
                    )?
                );
                ctx.call("", "swiglu", None, [gate, up])
            }
            Self::GeLU => {
                // format
                ctx.call("", "gelu", None, [x])
            }
        };

        Ok((ctx, outputs?))
    }
}
