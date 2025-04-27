use super::{Embedding, NNCtx, NNError, NuralNetwork, Tensor, TransformerBlk, macros::destruct};

pub struct LLaMA<T> {
    pub embedding: Embedding<T>,
    pub blks: Box<[TransformerBlk<T>]>,
}

impl<T> NuralNetwork<T> for LLaMA<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: NNCtx<T>,
    ) -> Result<(NNCtx<T>, Vec<Tensor<T>>), NNError> {
        let Self { embedding, blks } = self;

        destruct!([tokens, pos] = inputs);

        destruct!([x] = ctx.trap("embedding", embedding, [tokens])?);

        let x = blks.into_iter().enumerate().try_fold(x, |x, (i, blk)| {
            destruct!([x] = ctx.trap(format!("blk{i}"), blk, [x, pos.clone()])?);
            Ok(x)
        })?;

        Ok((ctx, vec![x]))
    }
}
