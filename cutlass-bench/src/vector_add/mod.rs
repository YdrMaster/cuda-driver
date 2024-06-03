#[test]
fn bench() {
    use cuda::{memcpy_d2h, Ptx};
    use half::f16;
    use std::{ffi::CString, time::Instant};

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };

    const ITEMS_PER_THREAD: usize = 8;

    const NORMAL: &str = include_str!("cuda_c.cuh");
    const NORMAL_NAME: &str = "normal";
    let normal_name = CString::new(NORMAL_NAME).unwrap();
    let normal_ptx = {
        let code = format!(
            r#"{NORMAL}

extern "C" __global__ void {NORMAL_NAME}(
    half       *__restrict__ z,
    half const *__restrict__ x,
    half const *__restrict__ y,
    float const kx,
    float const ky,
    float const b,
    int   const num
){{
    vector_add<{ITEMS_PER_THREAD}>
    (z, x, y, kx, ky, b, num);
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
    half       *__restrict__ z,
    half const *__restrict__ x,
    half const *__restrict__ y,
    float const kx,
    float const ky,
    float const b,
    int   const num
){{
    vector_add<{ITEMS_PER_THREAD}>
    (z, x, y, kx, ky, b, num);
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

        let len = 128 * 1024 * ITEMS_PER_THREAD;
        let mut z = vec![f16::default(); len];
        let x = vec![f16::from_f32(1.); len];
        let y = vec![f16::from_f32(2.); len];

        let time = Instant::now();
        let mut dev_z = stream.malloc::<f16>(len);
        let dev_x = stream.from_host(&x);
        let dev_y = stream.from_host(&y);
        println!("malloc {:?}", time.elapsed());

        let ptr_z = dev_z.as_mut_ptr();
        let ptr_x = dev_x.as_ptr();
        let ptr_y = dev_y.as_ptr();
        let params = cuda::params![ptr_z, ptr_x, ptr_y, 2.0f32, 2.0f32, 1.0f32, len];

        let normal = ctx.load(&normal_ptx);
        let normal = normal.get_kernel(&normal_name);
        let cutlass = ctx.load(&cutlass_ptx);
        let cutlass = cutlass.get_kernel(&cutlass_name);

        normal.launch(128, 1024, params.as_ptr(), 0, Some(&stream));
        memcpy_d2h(&mut z, &dev_z);
        assert!(z.iter().all(|&x| x == f16::from_f32(7.)));
        cutlass.launch(128, 1024, params.as_ptr(), 0, Some(&stream));
        memcpy_d2h(&mut z, &dev_z);
        assert!(z.iter().all(|&x| x == f16::from_f32(7.)));
    });
}
