use cuda::{AsRaw, KernelFn};
use rand::Rng;
use std::ffi::{c_uint, c_void};
use test_utils::diff;

use crate::ReduceMean;

#[test]
fn bench() {
    const ROW: usize = 2048;
    const COL: usize = 2048;
    const VALID: usize = 256;
    const BLOCK_SIZE: usize = 512;

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    let mut result0 = vec![0.0f32; ROW];
    let mut result1 = vec![0.0f32; ROW];
    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let mut rng = rand::thread_rng();
        let mut x_data = vec![0.0f32; ROW * COL];
        rng.fill(&mut x_data[..]);
        let x = stream.from_slice(&x_data);
        let y = stream.malloc_for::<f32>(ROW);
        let x = x.as_slice(ctx);
        let y = y.as_slice(ctx);

        {
            let name = "reduce_mean_bench";
            let code = format!(
                r#"
#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>

extern "C" __global__ void {name}(
    float const *__restrict__ x_,
    float       *__restrict__ y_
) {{
    auto x = x_ + blockIdx.x * {COL};
    auto y = y_ + blockIdx.x;
    float val[1];
    {{
        using BlockOp = cub::BlockLoad<float, {VALID}, 1>;
        __shared__ typename BlockOp::TempStorage temp_storage;
        BlockOp(temp_storage).Load(x, val);
    }}
    {{
        using BlockReduce = cub::BlockReduce<float, {VALID}>;
        __shared__ typename BlockReduce::TempStorage temp_storage;
        *val = BlockReduce(temp_storage).Reduce(*val, cub::Sum());
    }}

    if (threadIdx.x == 0) *y = *val / {VALID};
}}
"#
            );
            ctx.compile(code);
            let f = KernelFn::get(&name).unwrap();
            let x_ptr = unsafe { x.as_raw() };
            let y_ptr = unsafe { y.as_raw() };
            let params: [*const c_void; 2] = [(&x_ptr) as *const _ as _, (&y_ptr) as *const _ as _];

            let ela = stream.bench(
                |_, stream| {
                    f.launch(
                        ROW as c_uint,
                        VALID as c_uint,
                        params.as_ptr(),
                        0,
                        Some(stream),
                    );
                },
                10000,
                10,
            );
            println!("ela: {ela:?}");
            y.copy_out(&mut result0);
        }
        {
            let kernel = ReduceMean::new(COL, BLOCK_SIZE, ctx);
            let ela = stream.bench(
                |_, stream| {
                    kernel.launch(&x, &y, VALID, stream);
                },
                10000,
                10,
            );
            println!("ela: {ela:?}");
            y.copy_out(&mut result1);
        }
        let diff = diff(&result0, &result1);
        println!("diff: {diff:?}");
    });
}
