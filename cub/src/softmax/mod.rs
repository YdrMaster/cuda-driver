use cuda::{ContextGuard, CudaDataType, KernelFn, Stream};
use std::ffi::c_uint;

pub struct Softmax {
    baseline: KernelFn,
    block_size: c_uint,
}

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
    }}
}}
"#
        );

        ctx.compile(code);
        Self {
            baseline: KernelFn::get(&baseline).unwrap(),
            block_size: block_size as _,
        }
    }
}

#[test]
fn bench() {
    use cuda::AsRaw;
    use rand::Rng;
    use std::ffi::c_void;

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

        let f = |algo: c_uint, stream: &Stream| {
            let params: [*const c_void; 3] = [
                (&x_ptr) as *const _ as _,
                (&leading_dim) as *const _ as _,
                (&algo) as *const _ as _,
            ];
            softmax.baseline.launch(
                ROW as c_uint,
                COL as c_uint,
                params.as_ptr(),
                COL,
                Some(stream),
            )
        };

        for algo in 0..=2 {
            let ela = stream.bench(|_, stream| f(algo, stream), 10000, 10);
            println!("softmax{algo}: {ela:?}");
        }
    });
}
