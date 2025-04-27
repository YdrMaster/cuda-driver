use super::Modules;
use crate::nn::{ModuleKey, cuda_type, move_type};
use cuda::{DevByte, Graph, VirByte, params};
use std::ffi::c_uint;
use tensor::{Tensor, digit_layout::DigitLayout};

pub struct Weight<T> {
    pub token_embd: T,
}

impl<T> Weight<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Weight<U> {
        Weight {
            token_embd: f(self.token_embd),
        }
    }

    pub fn as_ref(&self) -> Weight<&T> {
        Weight {
            token_embd: &self.token_embd,
        }
    }
}

impl<const N: usize> Weight<Tensor<&[DevByte], N>> {
    pub fn add_to_graph(
        &self,
        graph: &Graph,
        modules: &mut Modules,
        x: Tensor<&[VirByte], N>,
        tokens: Tensor<&[VirByte], N>,
    ) {
        let tval = x.dt();
        let tidx = tokens.dt();
        assert_eq!(self.token_embd.dt(), tval);

        let &[n] = tokens.shape() else { unreachable!() };
        let &[n_, d] = x.shape() else { unreachable!() };
        let &[_, d_] = self.token_embd.shape() else {
            unreachable!()
        };
        assert_eq!(n, n_);
        assert_eq!(d, d_);

        let line = d * tval.nbytes();
        let unit = (0..=5)
            .rev()
            .map(|i| 1 << i)
            .find(|unit| line % unit == 0)
            .unwrap();

        let key = vec![
            ModuleKey::Text("embedding"),
            ModuleKey::Type(tidx),
            ModuleKey::Size(unit),
        ]
        .into_boxed_slice();
        let module = modules.compile(key, || code(unit, tidx));
        let kernel = module.get_kernel(c"embedding");
        let out = x.get().as_ptr();
        let table = self.token_embd.get().as_ptr();
        let index = tokens.get().as_ptr();
        let params = params![out, table, index];

        graph.add_kernel_call(
            &kernel,
            (n as c_uint, (line / unit) as c_uint, 0),
            params.as_ptr() as _,
            &[],
        );
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
