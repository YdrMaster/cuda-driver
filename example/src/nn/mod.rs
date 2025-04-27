use cuda::{CurrentCtx, Module, Ptx};
use std::collections::HashMap;
use tensor::digit_layout::{DigitLayout, types};

pub mod embedding;
pub mod ffn;
pub mod linear;
pub mod llama;
pub mod normalization;
pub mod rope;
pub mod self_attn;
pub mod transformer;

pub struct Modules<'ctx> {
    ctx: &'ctx CurrentCtx,
    modules: HashMap<Box<[ModuleKey]>, Module<'ctx>>,
}

impl<'ctx> Modules<'ctx> {
    pub fn new(ctx: &'ctx CurrentCtx) -> Self {
        Self {
            ctx,
            modules: HashMap::new(),
        }
    }

    pub fn compile(&mut self, key: Box<[ModuleKey]>, code: impl FnOnce() -> String) -> &Module {
        self.modules.entry(key).or_insert_with(|| {
            let (ptx, log) = Ptx::compile(code(), self.ctx.dev().compute_capability());
            let Ok(ptx) = ptx else { panic!("{log}") };
            self.ctx.load(&ptx)
        })
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModuleKey {
    Text(&'static str),
    Type(DigitLayout),
    Size(usize),
}

fn cuda_type(ty: DigitLayout) -> &'static str {
    match ty {
        types::U32 => "unsigned int",
        types::F32 => "float",
        _ => todo!(),
    }
}

fn move_type(unit: usize) -> &'static str {
    match unit {
        1 => "char",
        2 => "short",
        4 => "float",
        8 => "float2",
        16 => "float4",
        32 => "double4",
        _ => todo!(),
    }
}
