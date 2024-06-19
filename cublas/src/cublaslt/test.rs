use std::iter::zip;

use cuda::{memcpy_d2h, AsRaw, DevMem, Device, Stream};
use rand::Rng;

use crate::{
    bindings::{cublasComputeType_t::CUBLAS_COMPUTE_32F, cudaDataType},
    cublaslt::CublasLtMatMulLayout,
    Cublas, CublasLt, CublasLtMatMulDescriptor, CublasLtMatrix, CublasLtMatrixLayout, MatrixOrder,
};

fn rand_blob<'ctx>(len: usize, stream: &Stream<'ctx>) -> DevMem<'ctx> {
    let mut rng = rand::thread_rng();
    let mut mem = vec![0.0f32; len];
    rng.fill(&mut mem[..]);
    stream.from_host(&mem)
}

#[test]
fn general() {
    cuda::init();
    let Some(dev) = Device::fetch() else {
        return;
    };

    const M: usize = 5376;
    const K: usize = 2048;
    const N: usize = 256;
    const ALPHA: f32 = 1.;
    const BETA: f32 = 0.;

    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let dev_a = rand_blob(M * K, &stream);
        let dev_b = rand_blob(K * N, &stream);
        let mut dev_c = stream.malloc::<f32>(M * N);

        let cublas = Cublas::new(ctx);
        cublas.set_stream(&stream);

        cublas!(cublasGemmEx(
            cublas.as_raw(),
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_N,
            N as _,
            M as _,
            K as _,
            ((&ALPHA) as *const f32).cast(),
            dev_b.as_ptr() as _,
            cudaDataType_t::CUDA_R_32F,
            N as _,
            dev_a.as_ptr() as _,
            cudaDataType_t::CUDA_R_32F,
            K as _,
            ((&BETA) as *const f32).cast(),
            dev_c.as_mut_ptr() as _,
            cudaDataType_t::CUDA_R_32F,
            N as _,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
        ));
        let mut answer = vec![0.0f32; M * N];
        memcpy_d2h(&mut answer, &dev_c);

        let a_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
            rows: M as _,
            cols: K as _,
            major_stride: K as _,
            order: MatrixOrder::RowMajor,
            data_type: cudaDataType::CUDA_R_32F,
            batch: 1,
            stride: 0,
        });
        let b_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
            rows: K as _,
            cols: N as _,
            major_stride: N as _,
            order: MatrixOrder::RowMajor,
            data_type: cudaDataType::CUDA_R_32F,
            batch: 1,
            stride: 0,
        });
        let c_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
            rows: M as _,
            cols: N as _,
            major_stride: N as _,
            order: MatrixOrder::RowMajor,
            data_type: cudaDataType::CUDA_R_32F,
            batch: 1,
            stride: 0,
        });
        let mat_mul = CublasLtMatMulDescriptor::new(CUBLAS_COMPUTE_32F, cudaDataType::CUDA_R_32F);
        let layout = CublasLtMatMulLayout {
            mat_mul: &mat_mul,
            a: &a_desc,
            b: &b_desc,
            c: &c_desc,
            d: &c_desc,
        };

        let cublaslt = CublasLt::new(ctx);
        let mut algo = cublaslt.tune(layout, usize::MAX, 1);
        let (algo, workspace_size) = algo.pop().unwrap();
        let mut workspace = stream.malloc::<u8>(workspace_size);
        cublaslt.mat_mul(
            layout,
            algo,
            dev_c.as_mut_ptr(),
            ALPHA,
            dev_a.as_ptr(),
            dev_b.as_ptr(),
            BETA,
            dev_c.as_ptr(),
            &mut *workspace,
            &stream,
        );

        let mut result = vec![0.0f32; M * N];
        memcpy_d2h(&mut result, &dev_c);
        assert_eq!(result, answer);
    });
}

#[test]
fn bench() {
    cuda::init();
    let Some(dev) = Device::fetch() else {
        return;
    };

    let n = 2048 + 256 + 256;
    let k = 2048;
    let alpha = 1.0f32;
    let beta = 0.0f32;

    println!("m = ?");
    println!("n = {n}");
    println!("k = {k}");

    for m in 1..=1024 {
        println!("m = {m} =========================");
        dev.context().apply(|ctx| {
            let stream = ctx.stream();
            let dev_a = rand_blob(m * k, &stream);
            let dev_b = rand_blob(k * n, &stream);
            let mut dev_c = stream.malloc::<f32>(m * n);

            let cublas = Cublas::new(ctx);
            cublas.set_stream(&stream);

            let mut f = || {
                cublas!(cublasGemmEx(
                    cublas.as_raw(),
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    n as _,
                    m as _,
                    k as _,
                    ((&alpha) as *const f32).cast(),
                    dev_b.as_ptr() as _,
                    cudaDataType_t::CUDA_R_32F,
                    n as _,
                    dev_a.as_ptr() as _,
                    cudaDataType_t::CUDA_R_32F,
                    k as _,
                    ((&beta) as *const f32).cast(),
                    dev_c.as_mut_ptr() as _,
                    cudaDataType_t::CUDA_R_32F,
                    n as _,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
                ))
            };
            let cublas_time = stream.bench(|_, _| f(), 36, 4);
            println!("cublas  : {cublas_time:>9?}");

            let mut answer = vec![0.0f32; m * n];
            memcpy_d2h(&mut answer, &dev_c);

            let a_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
                rows: m as _,
                cols: k as _,
                major_stride: k as _,
                order: MatrixOrder::RowMajor,
                data_type: cudaDataType::CUDA_R_32F,
                batch: 1,
                stride: 0,
            });
            let b_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
                rows: k as _,
                cols: n as _,
                major_stride: n as _,
                order: MatrixOrder::RowMajor,
                data_type: cudaDataType::CUDA_R_32F,
                batch: 1,
                stride: 0,
            });
            let c_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
                rows: m as _,
                cols: n as _,
                major_stride: n as _,
                order: MatrixOrder::RowMajor,
                data_type: cudaDataType::CUDA_R_32F,
                batch: 1,
                stride: 0,
            });
            let mat_mul =
                CublasLtMatMulDescriptor::new(CUBLAS_COMPUTE_32F, cudaDataType::CUDA_R_32F);
            let layout = CublasLtMatMulLayout {
                mat_mul: &mat_mul,
                a: &a_desc,
                b: &b_desc,
                c: &c_desc,
                d: &c_desc,
            };

            let mut result = vec![0.0f32; m * n];
            let cublaslt = CublasLt::new(ctx);
            let algo = cublaslt.tune(layout, usize::MAX, 10);
            for (algo, workspace_size) in algo {
                let mut workspace = stream.malloc::<u8>(workspace_size);
                let time = stream.bench(
                    |_, stream| {
                        cublaslt.mat_mul(
                            layout,
                            algo,
                            dev_c.as_mut_ptr(),
                            alpha,
                            dev_a.as_ptr(),
                            dev_b.as_ptr(),
                            beta,
                            dev_c.as_ptr(),
                            &mut *workspace,
                            &stream,
                        );
                    },
                    36,
                    4,
                );
                workspace.drop_on(&stream);
                memcpy_d2h(&mut result, &dev_c);
                if time < cublas_time.mul_f64(1.05) {
                    println!(
                        "cublaslt: {time:>9?} {:>5.1}% {:>7.2e} | {algo:?}",
                        (cublas_time.as_secs_f64() - time.as_secs_f64())
                            / cublas_time.as_secs_f64()
                            * 100.0,
                        zip(&result, &answer).fold(0., |max, (r, a)| f32::max(max, (r - a).abs()))
                    );
                }
            }
        });
    }
}
