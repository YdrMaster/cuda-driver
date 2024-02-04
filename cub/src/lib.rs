#[cfg(test)]
mod test {
    use cuda::{nvrtc::compile, AsRaw, KernelFn};
    use rand::Rng;
    use std::ffi::{c_uint, c_void};

    /// 计算 reduceMean 并与 kernel 函数的结果进行比较。
    fn check(data: &[f32], result: &[f32]) -> (f64, f64) {
        let row = result.len();
        let col = data.len() / row;
        debug_assert_eq!(data.len() % row, 0);

        let mut answer = vec![0.0f32; row];
        for (i, ans) in answer.iter_mut().enumerate() {
            *ans = data[i * col..][..col].iter().sum::<f32>() / col as f32;
        }
        test_utils::diff(&result, &answer)
    }

    #[test]
    fn general() {
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
    auto x = x_ + blockIdx.x * {COL};
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
                let params: [*const c_void; 2] =
                    [(&x_ptr) as *const _ as _, (&y_ptr) as *const _ as _];
                function.launch(
                    ROW as c_uint,
                    COL as c_uint,
                    params.as_ptr(),
                    0,
                    Some(&stream),
                );
                stream.synchronize();
            }

            let mut result = vec![0.0f32; ROW];
            y.copy_out(&mut result);
            let (abs_diff, rel_diff) = check(&x_data, &result);
            assert!(abs_diff < 1e-6, "abs_diff: {abs_diff}");
            assert!(rel_diff < 1e-6, "rel_diff: {rel_diff}");
        });
    }

    #[test]
    fn padding() {
        const ROW: usize = 256;
        const COL: usize = 711;
        const BLOCK_SIZE: usize = 1024;

        fn src() -> String {
            format!(
                r#"
#include <cub/block/block_reduce.cuh>

extern "C" __global__ void reduceMean(
    float const *__restrict__ x_,
    float       *__restrict__ y_,
    unsigned int item_size
) {{
    auto x = x_ + blockIdx.x * item_size;
    auto y = y_ + blockIdx.x;

    using BlockReduce = cub::BlockReduce<float, {BLOCK_SIZE}>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    auto acc = BlockReduce(tempStorage).Reduce(x[threadIdx.x], cub::Sum(), item_size);
    if (threadIdx.x == 0) *y = acc / float(item_size);
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
                let item_size = COL as u32;
                let params: &[*const c_void] = &[
                    (&x_ptr) as *const _ as _,
                    (&y_ptr) as *const _ as _,
                    (&item_size) as *const _ as _,
                ];
                function.launch(
                    ROW as c_uint,
                    BLOCK_SIZE as c_uint,
                    params.as_ptr(),
                    0,
                    Some(&stream),
                );
                stream.synchronize();
            }

            let mut result = vec![0.0f32; ROW];
            y.copy_out(&mut result);
            let (abs_diff, rel_diff) = check(&x_data, &result);
            assert!(abs_diff < 1e-6, "abs_diff: {abs_diff}");
            assert!(rel_diff < 1e-6, "rel_diff: {rel_diff}");
        });
    }
}
