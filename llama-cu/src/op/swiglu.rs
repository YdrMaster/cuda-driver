use super::{ModuleKey, Modules, Operator, cuda_type, macros::*};
use cuda::{Graph, GraphNode, VirByte, params};

use std::ffi::c_uint;
use tensor::{Tensor, digit_layout::DigitLayout};

pub struct Swiglu;

impl Operator for Swiglu {
    fn add_to_graph<'a, const N: usize>(
        graph: &'a Graph,
        deps: &[GraphNode<'a>],
        modules: &mut Modules,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
    ) -> Vec<GraphNode<'a>> {
        if arg.is_some() {
            panic!("Swiglu不需要额外参数");
        }

        destruct!([gate, up] = inputs);
        destruct!([out] = outputs);

        // 检查维度
        dims!([n, d] = gate);
        dims!([n2, d2] = up);
        dims!([n3, d3] = out);

        assert_eq!(n, n2);
        assert_eq!(n, n3);
        assert_eq!(d, d2);
        assert_eq!(d, d3);

        // 检查类型
        let dt = gate.dt();
        assert_eq!(up.dt(), dt);
        assert_eq!(out.dt(), dt);

        // 获取stride
        strides!([s_n_gate, s_d_gate] = gate);
        strides!([s_n_up, s_d_up] = up);
        strides!([s_n_out, s_d_out] = out);

        // 确保stride符合期望
        let unit = dt.nbytes() as isize;
        assert_eq!(s_d_gate, unit);
        assert_eq!(s_d_up, unit);
        assert_eq!(s_d_out, unit);

        let stride_token_gate = (s_n_gate / unit) as i32;
        let stride_token_up = (s_n_up / unit) as i32;
        let stride_token_out = (s_n_out / unit) as i32;

        // 获取代码
        let code = code(dt);

        // 获取最大线程数
        let max_threads_block = modules.ctx.dev().block_limit().max_threads;

        // 编译内核
        let key = [ModuleKey::Text("swiglu"), ModuleKey::Type(dt)].into_iter();
        let module = modules.compile(key.collect(), || code);
        let kernel = module.get_kernel(c"swiglu");

        // 准备参数
        let params = params![
            out.get(),
            stride_token_out,
            gate.get(),
            stride_token_gate,
            up.get(),
            stride_token_up
        ];

        // 计算线程块配置
        let block = gcd(max_threads_block, d);

        // 启动内核
        vec![
            graph
                .add_kernel_call(
                    &kernel,
                    (n as c_uint, (d / block) as c_uint, block),
                    &params.to_ptrs(),
                    deps,
                )
                .into(),
        ]
    }
}

// 计算最大公约数，用于确定最佳的块大小
fn gcd(mut a: usize, mut b: usize) -> usize {
    while b != 0 {
        let rem = a % b;
        a = b;
        b = rem;
    }
    a
}

fn code(dt: DigitLayout) -> String {
    const CODE: &str = include_str!("swiglu.cuh");
    let dt = cuda_type(dt);

    format!(
        r#"{CODE}

extern "C" __global__ void swiglu(
    {dt} *__restrict__ out,
    int const stride_out,
    {dt} const *__restrict__ gate,
    int const stride_gate,
    {dt} const *__restrict__ up,
    int const stride_up
){{
    swiglu<{dt}>(out, stride_out, gate, stride_gate, up, stride_up);
}}"#
    )
}
