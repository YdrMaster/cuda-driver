use crate::{
    macros::destruct,
    op::{self, Handle, Operator},
};
use cuda::{CurrentCtx, VirByte};
use nn::{Node, Tensor};
use regex::Regex;
use std::sync::LazyLock;

pub(crate) enum Exec<'ctx> {
    Graph(cuda::GraphExec<'ctx>),
    Attention(Box<Attention>),
}

pub(crate) struct Attention {
    pub iblk: usize,
    pub q: Tensor<*const VirByte, 2>,
    pub k: Tensor<*const VirByte, 2>,
    pub v: Tensor<*const VirByte, 2>,
    pub o: Tensor<*const VirByte, 2>,
}

pub fn merge_cuda_graph(
    ctx: &CurrentCtx,
    exec: impl IntoIterator<Item = nn::Exec<*const VirByte>>,
) -> (Handle, Box<[Exec]>) {
    let mut handle = op::Handle::new(ctx);
    let mut stream = None;
    let mut exec_ = Vec::new();
    for nn::Exec {
        node,
        inputs,
        outputs,
    } in exec
    {
        let Node { name, op, arg } = node;
        macro_rules! add_to_graph {
            ($op:ident) => {
                op::$op::launch(
                    &mut handle,
                    arg,
                    inputs,
                    outputs,
                    stream.get_or_insert_with(|| ctx.stream().capture()),
                )
            };
        }
        match &*op {
            "embedding" => add_to_graph!(Embedding),
            "rms-norm" => add_to_graph!(RmsNorm),
            "linear" => add_to_graph!(Linear),
            "rope" => add_to_graph!(Rope),
            "swiglu" => add_to_graph!(Swiglu),
            "empty" => {}
            "attention" => {
                static REGEX: LazyLock<Regex> =
                    LazyLock::new(|| Regex::new(r"^Ω\.blk(\d+)\.attn:attention$").unwrap());

                if let Some(stream) = stream.take() {
                    exec_.push(Exec::Graph(ctx.instantiate(&stream.end())))
                }

                destruct!([q, k, v] = inputs);
                destruct!([o] = outputs);
                let Some(nn::Arg::Int(dh)) = arg else {
                    panic!()
                };
                let dh = dh as usize;

                let transform = |t: Tensor<*const VirByte, 2>| {
                    t.transform(|layout| {
                        layout
                            .tile_be(1, &[layout.shape()[1] / dh, dh])
                            .transpose(&[1, 0])
                    })
                };

                let iblk = {
                    let (_, [iblk]) = REGEX.captures(&name).unwrap().extract();
                    iblk.parse().unwrap()
                };
                let q = transform(q);
                let k = transform(k);
                let v = transform(v);
                let o = transform(o);

                exec_.push(Exec::Attention(Box::new(Attention { iblk, q, k, v, o })))
            }
            _ => {
                print!("todo! {op} ({arg:?})");
                for t in inputs {
                    print!(" {}{:?}", t.dt(), t.shape())
                }
                print!(" ->");
                for t in outputs {
                    print!(" {}{:?}", t.dt(), t.shape())
                }
                println!();
                break;
            }
        }
    }
    if let Some(stream) = stream.take() {
        exec_.push(Exec::Graph(ctx.instantiate(&stream.end())))
    }
    (handle, exec_.into())
}
