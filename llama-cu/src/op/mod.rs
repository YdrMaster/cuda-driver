mod embedding;
mod linear;
mod rms_norm;

use cuda::{CurrentCtx, Graph, GraphNode, Module, Ptx, VirByte};
use nn::Tensor;
use std::{collections::HashMap, usize};
use tensor::digit_layout::{DigitLayout, types};

pub use embedding::Embedding;
pub use linear::Linear;
pub use rms_norm::RmsNorm;

pub trait Operator {
    fn add_to_graph<'a, const N: usize>(
        graph: &'a Graph,
        deps: &[GraphNode<'a>],
        modules: &mut Modules,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
    ) -> Vec<GraphNode<'a>>;
}

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

mod macros {
    macro_rules! destruct {
        ([$( $name:ident ),+] = $iter:expr) => {
            let mut iter = $iter.into_iter();
            $( let $name = iter.next().unwrap(); )+
            assert!(iter.next().is_none());
        };
    }

    macro_rules! dims {
        ($pat:pat = $tensor:expr) => {
            let &$pat = &*$tensor.shape() else {
                panic!("Ndim mismatch ( = {})", $tensor.shape().len())
            };
        };
    }

    pub(crate) use {destruct, dims};
}
