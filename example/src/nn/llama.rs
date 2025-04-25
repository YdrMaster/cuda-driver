use crate::loader::MemCalculator;

use super::{embedding, linear, normalization, transformer};
use std::ops::Range;
use tensor::{Tensor, digit_layout::DigitLayout};

pub struct Weight<T> {
    pub embedding: embedding::Weight<T>,
    pub blks: Box<[transformer::Weight<T>]>,
    pub output_norm: normalization::Weight<T>,
    pub lm_head: linear::Weight<T>,
}

impl<T> Weight<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Weight<U> {
        Weight {
            embedding: self.embedding.map(&mut f),
            blks: self.blks.into_iter().map(|blk| blk.map(&mut f)).collect(),
            output_norm: self.output_norm.map(&mut f),
            lm_head: self.lm_head.map(&mut f),
        }
    }

    pub fn as_ref(&self) -> Weight<&T> {
        Weight {
            embedding: self.embedding.as_ref(),
            blks: self.blks.iter().map(|blk| blk.as_ref()).collect(),
            output_norm: self.output_norm.as_ref(),
            lm_head: self.lm_head.as_ref(),
        }
    }
}

pub struct Meta {
    pub t_tok: DigitLayout,
    pub t_pos: DigitLayout,
    pub t_embd: DigitLayout,
    pub d: usize,
    pub nh: usize,
    pub nkvh: usize,
    pub dh: usize,
    pub di: usize,
}

impl Meta {
    pub fn tokens(&self, n: usize) -> Tensor<usize, 2> {
        let &Self { t_tok, .. } = self;
        Tensor::from_dim_slice(t_tok, &[n])
    }

    pub fn pos(&self, n: usize) -> Tensor<usize, 2> {
        let &Self { t_pos, .. } = self;
        Tensor::from_dim_slice(t_pos, &[n])
    }

    pub fn x(&self, n: usize) -> Tensor<usize, 2> {
        let &Self { t_embd, d, .. } = self;
        Tensor::from_dim_slice(t_embd, &[n, d])
    }

    pub fn attn(&self, n: usize) -> Tensor<usize, 2> {
        let &Self {
            t_embd,
            nh,
            nkvh,
            dh,
            ..
        } = self;
        Tensor::from_dim_slice(t_embd, &[n, (nh + nkvh + nkvh) * dh])
    }

    pub fn ffn(&self, n: usize) -> Tensor<usize, 2> {
        let &Self { t_embd, di, .. } = self;
        Tensor::from_dim_slice(t_embd, &[n, di])
    }

    pub fn workspace(&self, n: usize, align: usize) -> Workspace {
        let pos = self.pos(n).take();
        let x = self.x(n).take();
        let tokens = self.tokens(n).take();
        let attn = self.attn(n).take();
        let ffn = self.ffn(n).take();
        let others = [tokens, attn, ffn].into_iter().max().unwrap();
        let mut calculator = MemCalculator::new(align);
        let x0 = calculator.push(x);
        let x1 = calculator.push(x);
        let pos = calculator.push(pos);
        let others = calculator.push(others);
        Workspace {
            tokens: others.start..others.start + tokens,
            pos,
            x: x0,
            x_: x1,
            attn: others.start..others.start + attn,
            ffn: others.start..others.start + ffn,
            size: calculator.size(),
        }
    }
}

pub struct Workspace {
    pub tokens: Range<usize>,
    pub pos: Range<usize>,
    pub x: Range<usize>,
    pub x_: Range<usize>,
    pub attn: Range<usize>,
    pub ffn: Range<usize>,
    pub size: usize,
}
