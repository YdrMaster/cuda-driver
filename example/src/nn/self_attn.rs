use super::{Attention, Linear, LinearResidual, NeuralNetwork, RoPE};

pub struct SelfAttn<T> {
    pub qkv: Linear<T>,
    pub q_rope: RoPE<T>,
    pub k_rope: RoPE<T>,
    pub attn: Attention<T>,
    pub output: LinearResidual<T>,
}

impl<T> NeuralNetwork<T> for SelfAttn<T> {}

impl<T> SelfAttn<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> SelfAttn<U> {
        SelfAttn {
            qkv: self.qkv.map(&mut f),
            q_rope: self.q_rope.map(&mut f),
            k_rope: self.k_rope.map(&mut f),
            attn: self.attn.map(&mut f),
            output: self.output.map(&mut f),
        }
    }

    pub fn as_ref(&self) -> SelfAttn<&T> {
        SelfAttn {
            qkv: self.qkv.as_ref(),
            q_rope: self.q_rope.as_ref(),
            k_rope: self.k_rope.as_ref(),
            attn: self.attn.as_ref(),
            output: self.output.as_ref(),
        }
    }
}
