use super::{Activation, Context, Linear, NNError, NuralNetwork, Tensor, macros::destruct};

pub struct Mlp<T> {
    pub up: Linear<T>,
    pub act: Activation,
    pub down: Linear<T>,
}

impl<T> NuralNetwork<T> for Mlp<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        let Self { up, act, down } = self;

        destruct!([x] = inputs);
        let residual = x.clone();
        destruct!([x] = ctx.trap("ffn-up", up, [x])?);
        destruct!([x] = ctx.trap("activation", act, [x])?);
        destruct!([x] = ctx.trap("ffn-down", down, [x, residual])?);

        Ok((ctx, vec![x]))
    }
}
