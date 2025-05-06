use crate::op::{self, Handle, Operator};
use cuda::{CurrentCtx, VirByte};
use nn::{Node, Tensor};

pub(crate) enum Exec<'ctx> {
    Graph(cuda::GraphExec<'ctx>),
    Attention {
        dh: usize,
        q: Tensor<*const VirByte, 2>,
        k: Tensor<*const VirByte, 2>,
        v: Tensor<*const VirByte, 2>,
        o: Tensor<*const VirByte, 2>,
    },
}

pub fn merge_cuda_graph<'ctx>(
    ctx: &'ctx CurrentCtx,
    exec: impl IntoIterator<Item = nn::Exec<*const VirByte>>,
) -> (Handle<'ctx>, Box<[Exec<'ctx>]>) {
    let mut handle = op::Handle::new(ctx);
    let mut stream = None;
    let mut exec_ = Vec::new();
    for nn::Exec {
        node,
        inputs,
        outputs,
    } in exec
    {
        let Node { op, arg, .. } = node;
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
                if let Some(stream) = stream.take() {
                    exec_.push(Exec::Graph(ctx.instantiate(&stream.end())))
                }

                let Some(nn::Arg::Int(dh)) = arg else {
                    panic!()
                };
                let mut inputs = inputs.into_iter();
                let mut outputs = outputs.into_iter();
                exec_.push(Exec::Attention {
                    dh: dh as _,
                    q: inputs.next().unwrap(),
                    k: inputs.next().unwrap(),
                    v: inputs.next().unwrap(),
                    o: outputs.next().unwrap(),
                });
                assert!(inputs.next().is_none());
                assert!(outputs.next().is_none());
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
