use super::{Context, Linear, NNError, NuralNetwork, Tensor, macros::*};
use crate::{Arg, Dim};

pub struct Attention<T> {
    pub nh: Dim,
    pub nkvh: Dim,
    pub qkv: Linear<T>,
    pub rope: Option<RoPE<T>>,
    pub output: Linear<T>,
}

pub struct RoPE<T> {
    pub nctx: Dim,
    pub sin: T,
    pub cos: T,
}

impl<T> NuralNetwork<T> for Attention<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x, pos] = inputs);

        let Self {
            nh,
            nkvh,
            qkv,
            rope,
            output,
        } = self;
        let residual = x.clone();

        destruct!([x] = ctx.trap("attn-qkv", qkv, [x])?);
        dims!([_, dqkv] = x);
        let dt = x.dt();
        let dh = dqkv.clone() / (nh.clone() + nkvh.clone() + nkvh.clone());

        destruct!(
            [q, k, v] = ctx.call(
                "split-qkv",
                "split",
                Some(Arg::dict([
                    ("axis".into(), Arg::int(1)),
                    (
                        "parts".into(),
                        Arg::arr([nh, nkvh.clone(), nkvh].map(Arg::from))
                    )
                ])),
                [x],
            )?
        );

        let [q, k] = match rope {
            Some(RoPE { nctx, sin, cos }) => {
                let shape = [nctx, dh / 2];
                let sin = ctx.weight("rope.sin", dt, shape.clone(), sin);
                let cos = ctx.weight("rope.cos", dt, shape.clone(), cos);
                destruct!(
                    [q_] = ctx.call(
                        "attn-q-rope",
                        "rope",
                        None,
                        [q, pos.clone(), sin.clone(), cos.clone()]
                    )?
                );
                destruct!([k_] = ctx.call("attn-k-rope", "rope", None, [k, pos, sin, cos])?);
                [q_, k_]
            }
            None => [q, k],
        };

        destruct!([o] = ctx.call("", "attention", None, [q, k, v])?);

        let outputs = ctx.trap("attn-output", output, [o, residual]);

        Ok((ctx, outputs?))
    }
}
