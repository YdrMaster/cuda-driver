use super::{Ffn, NeuralNetwork, RmsNorm, SelfAttn};

pub struct TransformerBlk<T> {
    pub attn_norm: RmsNorm<T>,
    pub attn: SelfAttn<T>,
    pub ffn_norm: RmsNorm<T>,
    pub ffn: Ffn<T>,
}

impl<T> NeuralNetwork<T> for TransformerBlk<T> {}

impl<T> TransformerBlk<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> TransformerBlk<U> {
        TransformerBlk {
            attn_norm: self.attn_norm.map(&mut f),
            attn: self.attn.map(&mut f),
            ffn_norm: self.ffn_norm.map(&mut f),
            ffn: self.ffn.map(&mut f),
        }
    }

    pub fn as_ref(&self) -> TransformerBlk<&T> {
        TransformerBlk {
            attn_norm: self.attn_norm.as_ref(),
            attn: self.attn.as_ref(),
            ffn_norm: self.ffn_norm.as_ref(),
            ffn: self.ffn.as_ref(),
        }
    }
}
