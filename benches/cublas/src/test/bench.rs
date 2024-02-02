use super::{rand_blob, ALPHA, BETA, K, M, N, TIMES};
use crate::{CublasLtHandle, CublasLtMatrix, CublasLtMatrixLayout, MatrixOrder};
use cuda::{AsRaw, Device};
use std::ptr::null_mut;

#[test]
fn general() {
    cuda::init();
    let Some(dev) = Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let dev_a = rand_blob(M * K, &stream);
        let dev_b = rand_blob(K * N, &stream);
        let dev_c = stream.malloc_for::<f32>(M * N);

        {
            let a_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
                rows: M as _,
                cols: K as _,
                major_stride: K as _,
                order: MatrixOrder::RowMajor,
                ..Default::default()
            });
            let b_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
                rows: K as _,
                cols: N as _,
                major_stride: N as _,
                order: MatrixOrder::RowMajor,
                ..Default::default()
            });
            let c_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
                rows: M as _,
                cols: N as _,
                major_stride: N as _,
                order: MatrixOrder::RowMajor,
                ..Default::default()
            });

            let matmul = cublaslt_matmul!(CUBLAS_COMPUTE_32F, CUDA_R_32F);
            let handle = CublasLtHandle::create_on(ctx);
            let (algo, workspace_size) = handle.tune(&matmul, &a_desc, &b_desc, &c_desc, &c_desc);
            let workspace = stream.malloc(workspace_size);
            let workspace = unsafe { workspace.as_slice_unchecked(workspace_size) };

            let f = || {
                matmul!(with handle, on stream;
                        do matmul, use algo, use workspace;
                       (1.; a_desc, dev_a; b_desc, dev_b)
                    => (0.; c_desc, dev_c; c_desc, dev_c));
            };

            println!(
                "cublasLt: {} ms",
                stream.bench(|_, _| f(), TIMES, 10).as_secs_f32() * 1000.
            );
        }

        {
            let mut cublas_handle = null_mut();
            cublas!(cublasCreate_v2(&mut cublas_handle));
            cublas!(cublasSetStream_v2(cublas_handle, stream.as_raw() as _));

            let f = || {
                cublas!(cublasGemmEx(
                    cublas_handle,
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    N as _,
                    M as _,
                    K as _,
                    ((&ALPHA) as *const f32).cast(),
                    dev_b.as_raw() as _,
                    cudaDataType_t::CUDA_R_32F,
                    N as _,
                    dev_a.as_raw() as _,
                    cudaDataType_t::CUDA_R_32F,
                    K as _,
                    ((&BETA) as *const f32).cast(),
                    dev_c.as_raw() as _,
                    cudaDataType_t::CUDA_R_32F,
                    N as _,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
                ))
            };

            println!(
                "cublas: {} ms",
                stream.bench(|_, _| f(), TIMES, 10).as_secs_f32() * 1000.
            );
        }
    });
}

#[test]
fn batching() {
    const BATCH: usize = 32;

    cuda::init();
    let Some(dev) = Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let dev_a = rand_blob(BATCH * M * K, &stream);
        let dev_b = rand_blob(BATCH * K * N, &stream);
        let dev_c = stream.malloc_for::<f32>(BATCH * M * N);

        {
            let mut cublaslt_handle = null_mut();
            cublas!(cublasLtCreate(&mut cublaslt_handle));

            let a_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
                rows: M as _,
                cols: K as _,
                major_stride: K as _,
                order: MatrixOrder::RowMajor,
                batch_count: BATCH as _,
                batch_stride: (M * K) as _,
                ..Default::default()
            });
            let b_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
                rows: K as _,
                cols: N as _,
                major_stride: N as _,
                order: MatrixOrder::RowMajor,
                batch_count: BATCH as _,
                batch_stride: (K * N) as _,
                ..Default::default()
            });
            let c_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
                rows: M as _,
                cols: N as _,
                major_stride: N as _,
                order: MatrixOrder::RowMajor,
                batch_count: BATCH as _,
                batch_stride: (M * N) as _,
                ..Default::default()
            });

            let matmul = cublaslt_matmul!(CUBLAS_COMPUTE_32F, CUDA_R_32F);
            let handle = CublasLtHandle::create_on(ctx);
            let (algo, workspace_size) = handle.tune(&matmul, &a_desc, &b_desc, &c_desc, &c_desc);
            let workspace = stream.malloc(workspace_size);
            let workspace = unsafe { workspace.as_slice_unchecked(workspace_size) };

            let f = || {
                matmul!(with handle, on stream;
                        do matmul, use algo, use workspace;
                       (1.; a_desc, dev_a; b_desc, dev_b)
                    => (0.; c_desc, dev_c; c_desc, dev_c));
            };

            println!(
                "cublasLt: {} ms",
                stream.bench(|_, _| f(), TIMES, 10).as_secs_f32() * 1000.
            );
        }
        {
            let mut cublas_handle = null_mut();
            cublas!(cublasCreate_v2(&mut cublas_handle));
            cublas!(cublasSetStream_v2(cublas_handle, stream.as_raw() as _));

            let f = || {
                cublas!(cublasGemmStridedBatchedEx(
                    cublas_handle,
                    cublasOperation_t::CUBLAS_OP_N,
                    cublasOperation_t::CUBLAS_OP_N,
                    N as _,
                    M as _,
                    K as _,
                    ((&ALPHA) as *const f32).cast(),
                    dev_b.as_raw() as _,
                    cudaDataType_t::CUDA_R_32F,
                    N as _,
                    (K * N) as _,
                    dev_a.as_raw() as _,
                    cudaDataType_t::CUDA_R_32F,
                    K as _,
                    (M * K) as _,
                    ((&BETA) as *const f32).cast(),
                    dev_c.as_raw() as _,
                    cudaDataType_t::CUDA_R_32F,
                    N as _,
                    (M * N) as _,
                    BATCH as _,
                    cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
                ));
            };

            println!(
                "cublas: {} ms",
                stream.bench(|_, _| f(), TIMES, 10).as_secs_f32() * 1000.
            );

            let f = || {
                for _ in 0..BATCH {
                    cublas!(cublasGemmEx(
                        cublas_handle,
                        cublasOperation_t::CUBLAS_OP_N,
                        cublasOperation_t::CUBLAS_OP_N,
                        N as _,
                        M as _,
                        K as _,
                        ((&ALPHA) as *const f32).cast(),
                        dev_b.as_raw() as _,
                        cudaDataType_t::CUDA_R_32F,
                        N as _,
                        dev_a.as_raw() as _,
                        cudaDataType_t::CUDA_R_32F,
                        K as _,
                        ((&BETA) as *const f32).cast(),
                        dev_c.as_raw() as _,
                        cudaDataType_t::CUDA_R_32F,
                        N as _,
                        cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
                    ))
                }
            };

            println!(
                "cublas loop: {} ms",
                stream.bench(|_, _| f(), TIMES, 10).as_secs_f32() * 1000.
            );
        }
    });
}

#[test]
fn broadcast() {
    const BATCH: usize = 32;

    cuda::init();
    let Some(dev) = Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let dev_a = rand_blob(M * K, &stream);
        let dev_b = rand_blob(BATCH * K * N, &stream);
        let dev_c = stream.malloc_for::<f32>(BATCH * M * N);

        {
            let mut cublaslt_handle = null_mut();
            cublas!(cublasLtCreate(&mut cublaslt_handle));

            let a_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
                rows: M as _,
                cols: K as _,
                major_stride: K as _,
                order: MatrixOrder::RowMajor,
                batch_count: BATCH as _,
                batch_stride: 0,
                ..Default::default()
            });
            let b_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
                rows: K as _,
                cols: N as _,
                major_stride: N as _,
                order: MatrixOrder::RowMajor,
                batch_count: BATCH as _,
                batch_stride: (K * N) as _,
                ..Default::default()
            });
            let c_desc = CublasLtMatrix::from(CublasLtMatrixLayout {
                rows: M as _,
                cols: N as _,
                major_stride: N as _,
                order: MatrixOrder::RowMajor,
                batch_count: BATCH as _,
                batch_stride: (M * N) as _,
                ..Default::default()
            });

            let matmul = cublaslt_matmul!(CUBLAS_COMPUTE_32F, CUDA_R_32F);
            let handle = CublasLtHandle::create_on(ctx);
            let (algo, workspace_size) = handle.tune(&matmul, &a_desc, &b_desc, &c_desc, &c_desc);
            let workspace = stream.malloc(workspace_size);
            let workspace = unsafe { workspace.as_slice_unchecked(workspace_size) };

            let f = || {
                matmul!(with handle, on stream;
                        do matmul, use algo, use workspace;
                       (1.; a_desc, dev_a; b_desc, dev_b)
                    => (0.; c_desc, dev_c; c_desc, dev_c));
            };

            println!(
                "cublasLt: {} ms",
                stream.bench(|_, _| f(), TIMES, 10).as_secs_f32() * 1000.
            );
        }
        {
            let mut cublas_handle = null_mut();
            cublas!(cublasCreate_v2(&mut cublas_handle));
            cublas!(cublasSetStream_v2(cublas_handle, stream.as_raw() as _));

            let f = || {
                for i in 0..BATCH {
                    cublas!(cublasGemmEx(
                        cublas_handle,
                        cublasOperation_t::CUBLAS_OP_N,
                        cublasOperation_t::CUBLAS_OP_N,
                        N as _,
                        M as _,
                        K as _,
                        ((&ALPHA) as *const f32).cast(),
                        (dev_b.as_raw() as *const f32).add(i * K * N).cast(),
                        cudaDataType_t::CUDA_R_32F,
                        N as _,
                        dev_a.as_raw() as _,
                        cudaDataType_t::CUDA_R_32F,
                        K as _,
                        ((&BETA) as *const f32).cast(),
                        (dev_c.as_raw() as *mut f32).add(i * M * N).cast(),
                        cudaDataType_t::CUDA_R_32F,
                        N as _,
                        cublasComputeType_t::CUBLAS_COMPUTE_32F,
                        cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
                    ))
                }
            };

            println!(
                "cublas loop: {} ms",
                stream.bench(|_, _| f(), TIMES, 10).as_secs_f32() * 1000.
            );
        }
    });
}
