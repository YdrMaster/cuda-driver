use super::{Linear, LinearResidual, NeuralNetwork, SwiGLU};

pub struct Ffn<T> {
    pub up: Linear<T>,
    pub act: SwiGLU<T>,
    pub down: LinearResidual<T>,
}

impl<T> NeuralNetwork<T> for Ffn<T> {}

impl<T> Ffn<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Ffn<U> {
        Ffn {
            up: self.up.map(&mut f),
            act: self.act.map(&mut f),
            down: self.down.map(&mut f),
        }
    }

    pub fn as_ref(&self) -> Ffn<&T> {
        Ffn {
            up: self.up.as_ref(),
            act: self.act.as_ref(),
            down: self.down.as_ref(),
        }
    }
}
