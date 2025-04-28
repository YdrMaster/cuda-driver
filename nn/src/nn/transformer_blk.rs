use super::{Attention, Context, Mlp, NNError, Normalization, NuralNetwork, Tensor};
use crate::macros::destruct;
pub struct TransformerBlk<T> {
    pub attn_norm: Normalization<T>,
    pub attn: Attention<T>,
    pub ffn_norm: Normalization<T>,
    pub ffn: Mlp<T>,
}

impl<T> NuralNetwork<T> for TransformerBlk<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self {
            attn_norm,
            attn,
            ffn_norm,
            ffn,
        } = self;
        destruct!([x, pos] = inputs);
        let tensors = ctx.trap("attn-norm", attn_norm, [x])?;
        destruct!([x] = tensors);
        let tensors = ctx.trap("attn", attn, [x, pos])?;
        destruct!([x] = tensors);
        let tensors = ctx.trap("ffn-norm", ffn_norm, [x])?;
        destruct!([x] = tensors);
        let tensors = ctx.trap("ffn", ffn, [x])?;

        Ok((ctx, tensors))
    }
}
