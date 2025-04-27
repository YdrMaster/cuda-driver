use super::{Modules, embedding, linear, normalization, transformer};
use crate::loader::MemCalculator;
use cuda::{DevByte, Graph, VirMem};
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
    pub nblk: usize,
    pub d: usize,
    pub nh: usize,
    pub nkvh: usize,
    pub dh: usize,
    pub di: usize,
}

impl Meta {
    pub fn tokens<const N: usize>(&self, n: usize) -> Tensor<usize, N> {
        let &Self { t_tok, .. } = self;
        Tensor::from_dim_slice(t_tok, [n])
    }

    pub fn pos<const N: usize>(&self, n: usize) -> Tensor<usize, N> {
        let &Self { t_pos, .. } = self;
        Tensor::from_dim_slice(t_pos, [n])
    }

    pub fn x<const N: usize>(&self, n: usize) -> Tensor<usize, N> {
        let &Self { t_embd, d, .. } = self;
        Tensor::from_dim_slice(t_embd, [n, d])
    }

    pub fn attn<const N: usize>(&self, n: usize) -> Tensor<usize, N> {
        let &Self {
            t_embd,
            nh,
            nkvh,
            dh,
            ..
        } = self;
        Tensor::from_dim_slice(t_embd, [n, (nh + nkvh + nkvh) * dh])
    }

    pub fn ffn<const N: usize>(&self, n: usize) -> Tensor<usize, N> {
        let &Self { t_embd, di, .. } = self;
        Tensor::from_dim_slice(t_embd, [n, di])
    }

    pub fn kv_cache<const N: usize>(&self, n: usize) -> Tensor<usize, N> {
        let &Self {
            t_embd,
            nblk,
            nkvh,
            dh,
            ..
        } = self;
        Tensor::from_dim_slice(t_embd, [n, nblk, 2, nkvh, dh])
    }

    pub fn workspace<const N: usize>(&self, n: usize, align: usize) -> Workspace<usize, N> {
        let pos = self.pos(n);
        let x = self.x(n);
        let tokens = self.tokens(n);
        let attn = self.attn(n);
        let ffn = self.ffn(n);
        let others = [tokens.get(), attn.get(), ffn.get()]
            .into_iter()
            .cloned()
            .max()
            .unwrap();

        let mut calculator = MemCalculator::new(align);
        let x0 = calculator.push(*x.get());
        let x1 = calculator.push(*x.get());
        let pos_ = calculator.push(*pos.get());
        let others = calculator.push(others);

        Workspace {
            tokens: tokens.map(|len| others.start..others.start + len),
            pos: pos.map(|_| pos_),
            x: x.clone().map(|_| x0),
            x_: x.map(|_| x1),
            attn: attn.map(|len| others.start..others.start + len),
            ffn: ffn.map(|len| others.start..others.start + len),
            item: calculator.size(),
        }
    }
}

#[derive(Clone)]
pub struct Workspace<T, const N: usize> {
    pub tokens: Tensor<Range<usize>, N>,
    pub pos: Tensor<Range<usize>, N>,
    pub x: Tensor<Range<usize>, N>,
    pub x_: Tensor<Range<usize>, N>,
    pub attn: Tensor<Range<usize>, N>,
    pub ffn: Tensor<Range<usize>, N>,
    pub item: T,
}

impl<T, const N: usize> Workspace<T, N> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Workspace<U, N> {
        Workspace {
            tokens: self.tokens,
            pos: self.pos,
            x: self.x,
            x_: self.x_,
            attn: self.attn,
            ffn: self.ffn,
            item: f(self.item),
        }
    }
}

impl<const N: usize> Weight<Tensor<&[DevByte], N>> {
    pub fn add_to_graph(
        &self,
        graph: &Graph,
        modules: &mut Modules,
        workspace: &Workspace<VirMem, N>,
    ) {
        let Workspace {
            tokens,
            x,
            item: workspace,
            ..
        } = workspace;
        let tokens = tokens.clone().map(|range| &workspace[range]);
        let x = x.clone().map(|range| &workspace[range]);
        self.embedding.add_to_graph(graph, modules, x, tokens);
    }
}
