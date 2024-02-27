use cuda::{AsRaw, ContextGuard, CudaDataType, KernelFn, Stream};
use rand::Rng;
use std::ffi::{c_uint, c_void};
use test_utils::diff;

struct Softmax(KernelFn);

impl Softmax {
    pub fn new(
        data_type: CudaDataType,
        _max_item_size: usize,
        block_size: usize,
        ctx: &ContextGuard,
    ) -> Self {
        let ty_arg = data_type.name();
        let baseline = format!("softmax_baseline_{block_size}");

        const RMS_NORMALIZATION: &str = include_str!("../templates/softmax_bench.cuh");
        let code = format!(
            r#"{RMS_NORMALIZATION}

extern "C" __global__ void {baseline}(
    {ty_arg} *__restrict__ x,
    unsigned int const leading_dim,
    unsigned int const algo
){{
    switch (algo) {{
        case 0:
            softmax0<{block_size}>(x, leading_dim);
            break;
        case 1:
            softmax1<{block_size}>(x, leading_dim);
            break;
        case 2:
            softmax2<{block_size}>(x, leading_dim);
            break;
        case 3:
            softmax3<{block_size}>(x, leading_dim);
            break;
        case 4:
            softmax4<{block_size}>(x, leading_dim);
            break;
        case 5:
            softmax5<{block_size}>(x, leading_dim);
            break;
    }}
}}
"#
        );

        ctx.compile(code);
        Self(KernelFn::get(baseline).unwrap())
    }
}

#[test]
fn once() {
    const ROW: usize = 4096;
    const COL: usize = 1024;

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let mut rng = rand::thread_rng();
        let mut x_data = vec![0.0f32; ROW * COL];
        rng.fill(&mut x_data[..]);
        let x = stream.from_slice(&x_data);
        let x = x.as_slice(ctx);

        let softmax = Softmax::new(CudaDataType::float, COL, COL, ctx);
        let x_ptr = unsafe { x.as_raw() };
        let leading_dim = COL as c_uint;

        for _ in 0..10 {
            let algo = 0 as c_uint;
            let params: [*const c_void; 3] = [
                (&x_ptr) as *const _ as _,
                (&leading_dim) as *const _ as _,
                (&algo) as *const _ as _,
            ];
            softmax.0.launch(
                ROW as c_uint,
                COL as c_uint,
                params.as_ptr(),
                COL,
                Some(&stream),
            );
        }
        stream.synchronize();
    });
}

#[test]
fn bench() {
    const ROW: usize = 4096;
    const COL: usize = 1024;

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    let mut results = [
        vec![0.0f32; ROW * COL],
        vec![0.0f32; ROW * COL],
        vec![0.0f32; ROW * COL],
        vec![0.0f32; ROW * COL],
        vec![0.0f32; ROW * COL],
        vec![0.0f32; ROW * COL],
    ];
    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let mut rng = rand::thread_rng();
        let mut x_data = vec![0.0f32; ROW * COL];
        rng.fill(&mut x_data[..]);
        let x = stream.from_slice(&x_data);
        let x = x.as_slice(ctx);

        let softmax = Softmax::new(CudaDataType::float, COL, COL, ctx);
        let x_ptr = unsafe { x.as_raw() };
        let leading_dim = COL as c_uint;

        let f = |algo: c_uint, stream: &Stream| {
            let params: [*const c_void; 3] = [
                (&x_ptr) as *const _ as _,
                (&leading_dim) as *const _ as _,
                (&algo) as *const _ as _,
            ];
            softmax.0.launch(
                ROW as c_uint,
                COL as c_uint,
                params.as_ptr(),
                COL,
                Some(stream),
            )
        };

        for algo in 0..=5 {
            let ela = stream.bench(|_, stream| f(algo, stream), 10000, 10);
            x.copy_out(&mut results[algo as usize]);
            println!("softmax{algo}: {ela:?}");
        }
        for i in 1..=3 {
            println!("softmax{i}: {:?}", diff(&results[i], &results[0]));
        }
    });
}
