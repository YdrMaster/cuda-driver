#[test]
fn test() {
    use cuda::{
        nvrtc::{compile, KernelFn},
        AsRaw,
    };
    use rand::Rng;
    use std::ffi::c_void;
    use test_utils::diff;

    const ROW: usize = 256;
    const COL: usize = 256;

    fn src() -> String {
        format!(
            r#"
#include <cub/block/block_reduce.cuh>

extern "C" __global__ void reduceMean(
    float const *__restrict__ x_,
    float       *__restrict__ y_
) {{
    auto x = x_ + blockIdx.x * blockDim.x;
    auto y = y_ + blockIdx.x;

    using BlockReduce = cub::BlockReduce<float, {COL}>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    auto acc = BlockReduce(tempStorage).Reduce(x[threadIdx.x], cub::Sum());
    if (threadIdx.x == 0) *y = acc / {COL};
}}
"#
        )
    }

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        compile(&src(), &["reduceMean"], ctx);
        let function = KernelFn::get("reduceMean").unwrap();

        let stream = ctx.stream();
        let mut rng = rand::thread_rng();
        let mut x_data = vec![0.0f32; ROW * COL];
        rng.fill(&mut x_data[..]);
        let x = stream.from_slice(&x_data);
        let y = stream.malloc_for::<f32>(ROW);
        let y = y.as_slice(ctx);

        {
            let x_ptr = unsafe { x.as_raw() };
            let y_ptr = unsafe { y.as_raw() };
            let params: [*const c_void; 2] = [(&x_ptr) as *const _ as _, (&y_ptr) as *const _ as _];
            function.launch(
                (ROW as _, 1, 1),
                (COL as _, 1, 1),
                params.as_ptr(),
                0,
                Some(&stream),
            );
            stream.synchronize();
        }

        let mut result = vec![0.0f32; ROW];
        let mut answer = vec![0.0f32; ROW];
        for i in 0..ROW {
            let mut sum = 0.0;
            for j in 0..COL {
                sum += x_data[i * COL + j];
            }
            answer[i] = sum / COL as f32;
        }

        y.copy_out(&mut result);
        let (abs_diff, rel_diff) = diff(&result, &answer);
        assert!(abs_diff < 1e-6);
        assert!(rel_diff < 1e-6)
    });
}
