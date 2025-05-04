use super::{ModuleKey, Modules, Operator, cuda_type, macros::*};
use cuda::{Graph, GraphNode, VirByte, params};
use std::ffi::c_uint;
use tensor::{
    Tensor,
    digit_layout::{DigitLayout, types},
};

pub struct Rope;

impl Operator for Rope {
    fn add_to_graph<'a, const N: usize>(
        graph: &'a Graph,
        deps: &[GraphNode<'a>],
        modules: &mut Modules,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
    ) -> Vec<GraphNode<'a>> {
        if arg.is_some() {
            panic!("Rope不需要额外参数");
        }

        destruct!([x, pos, sin, cos] = inputs);
        destruct!([y] = outputs);

        //检查dim
        dims!([n, dh_mut_dhead] = x);
        dims!([n2] = pos);
        dims!([nctx, dh_2] = sin);
        dims!([nctx2, dh_2_] = cos);
        dims!([n3, dh_mut_dhead_] = y);

        assert_eq!(n, n2);
        assert_eq!(n, n3);
        assert_eq!(dh_mut_dhead, dh_mut_dhead_);
        assert_eq!(dh_2, dh_2_);
        assert_eq!(nctx, nctx2);

        let dh = dh_2 * 2;
        let d_head = dh_mut_dhead / dh;
        assert_eq!(dh_mut_dhead % dh, 0);

        //检查type
        let dt_t = x.dt();
        let dt_p = pos.dt();

        assert_eq!(y.dt(), dt_t);
        assert_eq!(sin.dt(), types::F32);
        assert_eq!(cos.dt(), types::F32);

        //获取stride
        strides!([s_n_y, s_dh_mut_dhead_y] = y);
        strides!([s_n_x, s_dh_mut_dhead_x] = x);
        let stride_token_y = (s_n_y / dt_t.nbytes() as isize) as i32;
        let stride_head_y = (s_dh_mut_dhead_y / dt_t.nbytes() as isize * dh as isize) as i32;

        let stride_token_x = (s_n_x / dt_t.nbytes() as isize) as i32;
        let stride_head_x = (s_dh_mut_dhead_x / dt_t.nbytes() as isize * dh as isize) as i32;

        // 计算线程块配置，参考mod.rs中的逻辑
        let dh_div_2 = dh / 2;
        // 先获取最大线程数，避免后面借用冲突
        let max_threads_block = modules.ctx.dev().block_limit().max_threads;

        // 确保线程块大小不超过设备限制
        assert!(max_threads_block >= dh_div_2);

        // 计算合适的nh_l和nh_h
        let max_nh_l = (max_threads_block / dh_div_2).min(d_head);
        let nh_l = (1..=max_nh_l)
            .rev()
            .find(|nhl| d_head % nhl == 0)
            .unwrap_or(1);
        let nh_h = d_head / nh_l;

        let code = code(dt_p, dt_t);

        let key = [
            ModuleKey::Text("rope"),
            ModuleKey::Type(dt_t),
            ModuleKey::Type(dt_p),
        ]
        .into_iter();
        let module = modules.compile(key.collect(), || code);
        let kernel = module.get_kernel(c"rope");

        let params = params![
            y.get(),
            stride_token_y,
            stride_head_y,
            x.get(),
            stride_token_x,
            stride_head_x,
            pos.get(),
            sin.get(),
            cos.get()
        ];

        // 参考mod.rs中的kernel配置方式
        // gridDim = (n, nh_h)
        // blockDim = (dh_div_2, nh_l)
        vec![
            graph
                .add_kernel_call(
                    &kernel,
                    (
                        (n as c_uint, nh_h as c_uint),
                        (nh_l as c_uint, dh_div_2 as c_uint),
                        0,
                    ),
                    &params.to_ptrs(),
                    deps,
                )
                .into(),
        ]
    }
}

fn code(tp: DigitLayout, ta: DigitLayout) -> String {
    const CODE: &str = include_str!("rope.cuh");
    let ta = cuda_type(ta);
    let tp = cuda_type(tp);

    // rope操作直接使用padding模板函数
    let body = format!(
        "padding<{tp}, {ta}>(y, stride_token_y, stride_head_y, x, stride_token_x, stride_head_x, pos, sin_table, cos_table)"
    );

    let code = format!(
        r#"{CODE}

extern "C" __global__ void rope(
    {ta} *__restrict__ y,
    int const stride_token_y,
    int const stride_head_y,
    {ta} const *__restrict__ x,
    int const stride_token_x,
    int const stride_head_x,
    {tp} const *__restrict__ pos,
    float const *__restrict__ sin_table,
    float const *__restrict__ cos_table
){{
    {body};
}}"#
    );
    code
}
