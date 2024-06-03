#[test]
fn bench() {
    use cuda::{memcpy_d2h, Ptx};
    use half::f16;
    use std::{ffi::CString, time::Instant};

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };

    const NORMAL: &str = include_str!("cuda_c.cuh");
    const NORMAL_NAME: &str = "normal";
    let normal_name = CString::new(NORMAL_NAME).unwrap();
    let normal_ptx = {
        let code = format!(
            r#"{NORMAL}
extern "C" __global__ void {NORMAL_NAME}(
    half2              *__restrict__ x,
    unsigned int const *__restrict__ pos,
    float theta,
    unsigned int const leading_dim,
    int nt, int nh
){{
    rope(x, pos, theta, leading_dim, nt, nh);
}}
"#
        );
        let (result, log) = Ptx::compile(code, dev.compute_capability());
        if !log.trim().is_empty() {
            println!("{log}");
        }
        result.unwrap()
    };

    const CUTLASS: &str = include_str!("cutlass.cuh");
    const CUTLASS_NAME: &str = "cutlass_";
    let cutlass_name = CString::new(CUTLASS_NAME).unwrap();
    let cutlass_ptx = {
        let code: String = format!(
            r#"{CUTLASS}
extern "C" __global__ void {CUTLASS_NAME}(
    half2              *__restrict__ x,
    unsigned int const *__restrict__ pos,
    float theta,
    unsigned int const leading_dim,
    int nt, int nh
){{
    rope(x, pos, theta, leading_dim, nt, nh);
}}
"#
        );
        let (result, log) = Ptx::compile(code, dev.compute_capability());
        if !log.trim().is_empty() {
            println!("{log}");
        }
        result.unwrap()
    };

    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let nt: u32 = 1;
        let nh: u32 = 64;
        let dh: u32 = 128;
        let len = nt * nh * dh;
        let mut x = vec![f16::from_f32(1.); len as usize];
        let mut x_cutlass = vec![f16::from_f32(1.); len as usize];
        let pos = vec![1u32; nt as usize];
        // println!("{:?}", pos);
        let theta = 1e10f32;
        let leading_dim = 0;
        let time = Instant::now();
        let mut dev_x = stream.from_host(&x);
        let mut dev_x_cutlass = stream.from_host(&x_cutlass);
        let dev_pos = stream.from_host(&pos);
        println!("malloc {:?}", time.elapsed());
        let ptr_x = dev_x.as_mut_ptr();
        let ptr_x_cutlass = dev_x_cutlass.as_mut_ptr();
        let ptr_pos = dev_pos.as_ptr();
        let params = cuda::params![ptr_x, ptr_pos, theta, leading_dim, nt, nh];
        let params_cutlass = cuda::params![ptr_x_cutlass, ptr_pos, theta, leading_dim, nt, nh];

        let normal = ctx.load(&normal_ptx);
        let normal = normal.get_kernel(&normal_name);
        let cutlass = ctx.load(&cutlass_ptx);
        let cutlass = cutlass.get_kernel(&cutlass_name);

        normal.launch((nh, nt), dh / 2, params.as_ptr(), 0, Some(&stream));
        memcpy_d2h(&mut x, &dev_x);
        cutlass.launch((nh, nt), dh / 2, params_cutlass.as_ptr(), 0, Some(&stream));
        memcpy_d2h(&mut x_cutlass, &dev_x_cutlass);

        // compare results
        for i in 0..len as usize {
            assert_eq!(x[i], x_cutlass[i]);
        }
    });
}
