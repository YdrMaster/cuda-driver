use super::{Embedding, Linear, NeuralNetwork, RmsNorm, TransformerBlk};

pub struct Llama<T> {
    pub embedding: Embedding<T>,
    pub blks: Box<[TransformerBlk<T>]>,
    pub output_norm: RmsNorm<T>,
    pub lm_head: Linear<T>,
}

impl<T> NeuralNetwork<T> for Llama<T> {}

impl<T> Llama<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Llama<U> {
        Llama {
            embedding: self.embedding.map(&mut f),
            blks: self.blks.into_iter().map(|blk| blk.map(&mut f)).collect(),
            output_norm: self.output_norm.map(&mut f),
            lm_head: self.lm_head.map(&mut f),
        }
    }

    pub fn as_ref(&self) -> Llama<&T> {
        Llama {
            embedding: self.embedding.as_ref(),
            blks: self.blks.iter().map(|blk| blk.as_ref()).collect(),
            output_norm: self.output_norm.as_ref(),
            lm_head: self.lm_head.as_ref(),
        }
    }
}
