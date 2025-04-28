use super::{Context, NNError, NuralNetwork, Tensor, macros::destruct};
use crate::Dim;
use digit_layout::DigitLayout;

pub struct Linear<T> {
    pub dt: DigitLayout,
    pub shape: [Dim; 2],
    pub weight: T,
    pub bias: Option<(DigitLayout, T)>,
}

impl<T> NuralNetwork<T> for Linear<T> {
    fn launch(
        self,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        mut ctx: Context<T>,
    ) -> Result<(Context<T>, Vec<Tensor<T>>), NNError> {
        destruct!([x] = inputs);

        let Self {
            dt,
            shape,
            weight,
            bias,
        } = self;
        let [r, c] = shape;
        let w = ctx.weight("weight", dt, [r, c.clone()], weight);

        let outputs = match bias {
            Some((dt, bias)) => {
                let b = ctx.weight("bias", dt, [c], bias);
                ctx.call("", "linear", None, [x, w, b])
            }
            None => {
                // format
                ctx.call("", "linear", None, [x, w])
            }
        };

        Ok((ctx, outputs?))
    }
}
