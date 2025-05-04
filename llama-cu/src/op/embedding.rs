use super::{ModuleKey, Modules, Operator, cuda_type, macros::*, move_type};
use cuda::{Graph, GraphNode, VirByte, params};
use std::ffi::c_uint;
use tensor::{Tensor, digit_layout::DigitLayout};

pub struct Embedding;

impl Operator for Embedding {
    fn add_to_graph<'a, const N: usize>(
        graph: &'a Graph,
        deps: &[GraphNode<'a>],
        modules: &mut Modules,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
    ) -> Vec<GraphNode<'a>> {
        let None = arg else { panic!() };
        destruct!([token_embd, tokens] = inputs);
        destruct!([x] = outputs);
        let tval = x.dt();
        let tidx = tokens.dt();
        assert_eq!(token_embd.dt(), tval);

        dims!([n] = tokens);
        dims!([n_, d] = x);
        dims!([_, d_] = token_embd);
        assert_eq!(n, n_);
        assert_eq!(d, d_);

        let line = d * tval.nbytes();
        let unit = (0..=5)
            .rev()
            .map(|i| 1 << i)
            .find(|unit| line % unit == 0)
            .unwrap();

        let key = [
            ModuleKey::Text("embedding"),
            ModuleKey::Type(tidx),
            ModuleKey::Size(unit),
        ]
        .into_iter();
        let module = modules.compile(key.collect(), || code(unit, tidx));
        let kernel = module.get_kernel(c"embedding");
        let params = params![x.get(), token_embd.get(), tokens.get()];

        vec![
            graph
                .add_kernel_call(
                    &kernel,
                    (n as c_uint, (line / unit) as c_uint, 0),
                    &params.to_ptrs(),
                    deps,
                )
                .into(),
        ]
    }
}

fn code(unit: usize, tidx: DigitLayout) -> String {
    const CODE: &str = include_str!("embedding.cuh");
    let tval = move_type(unit);
    let tidx = cuda_type(tidx);
    format!(
        r#"{CODE}

extern "C" __global__ void embedding(
    {tval} *__restrict__ out,
    {tval} const *__restrict__ table,
    {tidx} const *__restrict__ index
) {{
    kernel(out, table, index);
}}"#
    )
}
