use super::{Handle, ModuleKey, Operator, cuda_type, macros::*, offset_ptr};
use cuda::{Device, Stream, VirByte, params};
use nn::Arg;
use std::ffi::{c_int, c_uint};
use tensor::{Tensor, digit_layout::DigitLayout};

pub struct RmsNorm;

impl Operator for RmsNorm {
    fn launch<'a, const N: usize>(
        handle: &mut Handle,
        arg: Option<nn::Arg>,
        inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        stream: &Stream,
    ) {
        destruct!([x, w] = inputs);
        destruct!([y] = outputs);
        let Some(Arg::Float(epsilon)) = arg else {
            panic!()
        };
        let ta = x.dt();
        let tw = w.dt();
        assert_eq!(y.dt(), ta);

        dims!([n, d] = x);
        dims!([n_, d_] = y);
        dims!([d__] = w);
        assert_eq!(n_, n);
        assert_eq!(d_, d);
        assert_eq!(d__, d);

        let (code, block_dim) = code(&handle.ctx.dev(), ta, tw, d);
        let key = [
            ModuleKey::Text("rms-norm"),
            ModuleKey::Type(ta),
            ModuleKey::Type(tw),
            ModuleKey::Size(d),
        ]
        .into_iter();
        let module = handle.compile(key.collect(), || code);
        let kernel = module.get_kernel(c"rms_norm");

        let unit = ta.nbytes() as isize;
        let params = params![
            offset_ptr(&y),
            (y.strides()[0] / unit) as c_int,
            offset_ptr(&x),
            (x.strides()[0] / unit) as c_int,
            offset_ptr(&w),
            epsilon as f32
        ];

        stream.launch(
            &kernel,
            (n as c_uint, block_dim as c_uint, 0),
            &params.to_ptrs(),
        );
    }
}

fn code(dev: &Device, ta: DigitLayout, tw: DigitLayout, d: usize) -> (String, usize) {
    const CODE: &str = include_str!("rms_norm.cuh");
    let ta = cuda_type(ta);
    let tw = cuda_type(tw);
    let block_size = dev.block_limit().max_threads;
    let (body, n_thread_block) = if d <= block_size {
        (
            format!("padding<{d}>(y, stride_y, x, stride_x, w, epsilon)"),
            d,
        )
    } else {
        let n_threads_warp = dev.warp_size();
        assert_eq!(d % n_threads_warp, 0);
        let max_num_warp_block = block_size / n_threads_warp;
        // num_warp_block in [1, max_num_warp_block]
        // num_threads_warp
        // num_items_thread in [1, 2, 4, 8] // 8 = 128bit / sizeof(half)
        // TODO 也许还能分得更好
        let to_divid = d / n_threads_warp;
        let num_warps_block = max_num_warp_block;
        let num_threads_block = n_threads_warp * num_warps_block;
        let num_items_thread = to_divid.div_ceil(num_warps_block);
        (
            format!(
                "folding<{num_threads_block}, {num_items_thread}>(y, stride_y, x, stride_x, w, epsilon, {d})"
            ),
            block_size,
        )
    };
    let code = format!(
        r#"{CODE}

extern "C" __global__ void rms_norm(
    {ta} *__restrict__ y,
    int  const stride_y,
    {ta} const *__restrict__ x,
    int  const stride_x,
    {tw} const *__restrict__ w,
    float epsilon
){{
    {body};
}}"#
    );
    (code, n_thread_block)
}
