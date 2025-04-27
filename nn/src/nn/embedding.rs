use super::{NNCtx, NNError, NuralNetwork, Tensor};
use crate::Dim;
use digit_layout::DigitLayout;

pub struct Embedding<T> {
    pub dt: DigitLayout,
    pub d: Dim,
    pub wte: Table<T>,
    pub wpe: Option<Table<T>>,
}

pub struct Table<T> {
    pub row: Dim,
    pub weight: T,
}

impl<T> NuralNetwork<T> for Embedding<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: NNCtx<T>,
    ) -> Result<(NNCtx<T>, Vec<Tensor<T>>), NNError> {
        let Self { dt, d, wte, wpe } = self;
        let mut inputs = inputs.into_iter();

        let Table { row, weight } = wte;
        let wte = ctx.weight("wte", dt, [row, d.clone()], weight);
        let tokens = inputs.next().unwrap();

        let outputs = match wpe {
            Some(wpe) => {
                let Table { row, weight } = wpe;
                let wpe = ctx.weight("wpe", dt, [row, d], weight);
                let pos = inputs.next().unwrap();
                ctx.call("", "embedding", None, [wte, tokens, wpe, pos])
            }
            None => {
                // format
                ctx.call("", "embedding", None, [wte, tokens])
            }
        };

        Ok((ctx, outputs?))
    }
}
