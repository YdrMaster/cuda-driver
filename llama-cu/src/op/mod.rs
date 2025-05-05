mod embedding;
mod linear;
mod rms_norm;
mod rope;
mod swiglu;

use cublas::Cublas;
use cuda::{CurrentCtx, Module, Ptx, Stream, VirByte};
use nn::Tensor;
use std::{collections::HashMap, usize};
use tensor::digit_layout::{DigitLayout, types};

pub use embedding::Embedding;
pub use linear::Linear;
pub use rms_norm::RmsNorm;
pub use rope::Rope;
pub use swiglu::Swiglu;

pub trait Operator {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    );
}

pub struct Handle<'ctx> {
    ctx: &'ctx CurrentCtx,
    cublas: Cublas<'ctx>,
    modules: HashMap<Box<[ModuleKey]>, Module<'ctx>>,
}

impl<'ctx> Handle<'ctx> {
    pub fn new(ctx: &'ctx CurrentCtx) -> Self {
        Self {
            ctx,
            cublas: Cublas::new(ctx),
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
        types::F16 => "half",
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

    macro_rules! strides {
        ($pat:pat = $tensor:expr) => {
            let &$pat = &*$tensor.strides() else {
                panic!("Ndim mismatch ( = {})", $tensor.strides().len())
            };
        };
    }

    pub(crate) use {destruct, dims, strides};
}
