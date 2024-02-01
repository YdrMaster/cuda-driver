use super::{rand_blob, uninit_blob, ALPHA, BETA, K, M, N};
use crate::{
    bindings::cublas,
    mat_mul::{matmul, tune},
    matrix::{CublasLtMatrix, CublasLtMatrixLayout, MatrixOrder},
};
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
        let dev_c = uninit_blob(M * N, &stream);

        let mut cublas_handle = null_mut();
        cublas!(cublasCreate_v2(&mut cublas_handle));
        cublas!(cublasSetStream_v2(cublas_handle, stream.as_raw() as _));

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
        ));
        let mut ans = vec![0.0f32; M * N];
        unsafe { dev_c.copy_out(&mut ans, ctx) };

        let mut cublaslt_handle = null_mut();
        cublas!(cublasLtCreate(&mut cublaslt_handle));

        let mut matmul_desc = null_mut();
        cublas!(cublasLtMatmulDescCreate(
            &mut matmul_desc,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType_t::CUDA_R_32F,
        ));

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

        let (algo, workspace_size) = tune(cublaslt_handle, matmul_desc, &a_desc, &b_desc, &c_desc);
        let workspace = stream.malloc(workspace_size);
        matmul(
            cublaslt_handle,
            matmul_desc,
            1.,
            (&a_desc, unsafe { dev_a.as_raw() } as _),
            (&b_desc, unsafe { dev_b.as_raw() } as _),
            0.,
            (&c_desc, unsafe { dev_c.as_raw() } as _),
            algo,
            (workspace_size, unsafe { workspace.as_raw() } as _),
            &stream,
        );

        let mut result = vec![0.0f32; M * N];
        unsafe { dev_c.copy_out(&mut result, ctx) };

        let mut max = 0.0f32;
        for (a, b) in ans.iter().zip(result.iter()) {
            max = max.max((a - b).abs());
        }
        println!("max: {max}");
        assert!(max < 1e-5);
    });
}

#[test]
fn batching() {
    const BATCH: usize = 10;

    cuda::init();
    let Some(dev) = Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let dev_a = rand_blob(BATCH * M * K, &stream);
        let dev_b = rand_blob(BATCH * K * N, &stream);
        let dev_c = uninit_blob(BATCH * M * N, &stream);

        let mut cublas_handle = null_mut();
        cublas!(cublasCreate_v2(&mut cublas_handle));
        cublas!(cublasSetStream_v2(cublas_handle, stream.as_raw() as _));

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
        let mut ans = vec![0.0f32; BATCH * M * N];
        unsafe { dev_c.copy_out(&mut ans, ctx) };

        let mut cublaslt_handle = null_mut();
        cublas!(cublasLtCreate(&mut cublaslt_handle));

        let mut matmul_desc = null_mut();
        cublas!(cublasLtMatmulDescCreate(
            &mut matmul_desc,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType_t::CUDA_R_32F,
        ));

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

        let (algo, workspace_size) = tune(cublaslt_handle, matmul_desc, &a_desc, &b_desc, &c_desc);
        let workspace = stream.malloc(workspace_size);
        matmul(
            cublaslt_handle,
            matmul_desc,
            1.,
            (&a_desc, unsafe { dev_a.as_raw() } as _),
            (&b_desc, unsafe { dev_b.as_raw() } as _),
            0.,
            (&c_desc, unsafe { dev_c.as_raw() } as _),
            algo,
            (workspace_size, unsafe { workspace.as_raw() } as _),
            &stream,
        );

        let mut result = vec![0.0f32; BATCH * M * N];
        unsafe { dev_c.copy_out(&mut result, ctx) };

        let mut max = 0.0f32;
        for (a, b) in ans.iter().zip(result.iter()) {
            max = max.max((a - b).abs());
        }
        println!("max: {max}");
        assert!(max < 1e-5);
    });
}

#[test]
fn broadcast() {
    const BATCH: usize = 10;
    const M: usize = 1024;
    const K: usize = 64;
    const N: usize = 256;

    cuda::init();
    let Some(dev) = Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let dev_a = rand_blob(M * K, &stream);
        let dev_b = rand_blob(BATCH * K * N, &stream);
        let dev_c = uninit_blob(BATCH * M * N, &stream);

        let mut cublas_handle = null_mut();
        cublas!(cublasCreate_v2(&mut cublas_handle));
        cublas!(cublasSetStream_v2(cublas_handle, stream.as_raw() as _));

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
            ));
        }
        let mut ans = vec![0.0f32; BATCH * M * N];
        unsafe { dev_c.copy_out(&mut ans, ctx) };

        let mut cublaslt_handle = null_mut();
        cublas!(cublasLtCreate(&mut cublaslt_handle));

        let mut matmul_desc = null_mut();
        cublas!(cublasLtMatmulDescCreate(
            &mut matmul_desc,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cudaDataType_t::CUDA_R_32F,
        ));

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

        let (algo, workspace_size) = tune(cublaslt_handle, matmul_desc, &a_desc, &b_desc, &c_desc);
        let workspace = stream.malloc(workspace_size);
        matmul(
            cublaslt_handle,
            matmul_desc,
            1.,
            (&a_desc, unsafe { dev_a.as_raw() } as _),
            (&b_desc, unsafe { dev_b.as_raw() } as _),
            0.,
            (&c_desc, unsafe { dev_c.as_raw() } as _),
            algo,
            (workspace_size, unsafe { workspace.as_raw() } as _),
            &stream,
        );

        let mut result = vec![0.0f32; BATCH * M * N];
        unsafe { dev_c.copy_out(&mut result, ctx) };

        let mut max = 0.0f32;
        for (a, b) in ans.iter().zip(result.iter()) {
            max = max.max((a - b).abs());
        }
        println!("max: {max}");
        assert!(max < 1e-5);
    });
}
