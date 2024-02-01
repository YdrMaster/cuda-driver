use crate::{bindings as cu, CublasLtMatrix};
use cuda::{AsRaw, Stream};
use std::{
    ffi::c_void,
    mem::{size_of, size_of_val, MaybeUninit},
    ptr::null_mut,
};

#[repr(transparent)]
pub struct CublasLtMatMulDescriptor(cu::cublasLtMatmulDesc_t);

unsafe impl Send for CublasLtMatMulDescriptor {}
unsafe impl Sync for CublasLtMatMulDescriptor {}

impl AsRaw for CublasLtMatMulDescriptor {
    type Raw = cu::cublasLtMatmulDesc_t;

    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Drop for CublasLtMatMulDescriptor {
    #[inline]
    fn drop(&mut self) {
        cublas!(cublasLtMatmulDescDestroy(self.0));
    }
}

impl CublasLtMatMulDescriptor {
    #[inline]
    pub fn new(compute_type: cu::cublasComputeType_t, scale_type: cu::cudaDataType) -> Self {
        let mut desc = null_mut();
        cublas!(cublasLtMatmulDescCreate(
            &mut desc,
            compute_type,
            scale_type
        ));
        Self(desc)
    }
}

#[test]
fn test_behavior() {
    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    let desc = dev
        .context()
        .apply(|_| cublaslt_matmul!(CUBLAS_COMPUTE_32F, CUDA_R_32F));
    dev.context().apply(|_| {
        // cublasLtMatmulDesc_t 在一个上下文中创建后，可以在其他上下文中使用
        cublas!(cublasLtMatmulDescSetAttribute(
            desc.as_raw(),
            cu::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &cu::cublasOperation_t::CUBLAS_OP_N as *const _ as _,
            std::mem::size_of::<cu::cublasOperation_t>(),
        ));
    });
    // 释放 cublas 对象不需要存在当前上下文
    drop(desc);
}

pub fn tune(
    handle: cu::cublasLtHandle_t,
    caluculation: cu::cublasLtMatmulDesc_t,
    a: &CublasLtMatrix,
    b: &CublasLtMatrix,
    c: &CublasLtMatrix,
) -> (cu::cublasLtMatmulAlgo_t, usize) {
    let mut device = 0;
    driver!(cuCtxGetDevice(&mut device));

    let mut alignment = 0;
    driver!(cuDeviceGetAttribute(
        &mut alignment,
        CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
        device
    ));

    let workspace = u64::MAX;
    let alignment = alignment as u32;

    let mut preference = null_mut();
    cublas!(cublasLtMatmulPreferenceCreate(&mut preference));
    cublas!(cublasLtMatmulPreferenceSetAttribute(
        preference,
        cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace as *const _ as _,
        size_of_val(&workspace),
    ));
    cublas!(cublasLtMatmulPreferenceSetAttribute(
        preference,
        cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
        &alignment as *const _ as _,
        size_of_val(&alignment),
    ));
    cublas!(cublasLtMatmulPreferenceSetAttribute(
        preference,
        cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        &alignment as *const _ as _,
        size_of_val(&alignment),
    ));
    cublas!(cublasLtMatmulPreferenceSetAttribute(
        preference,
        cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
        &alignment as *const _ as _,
        size_of_val(&alignment),
    ));
    cublas!(cublasLtMatmulPreferenceSetAttribute(
        preference,
        cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        &alignment as *const _ as _,
        size_of_val(&alignment),
    ));

    let mut result = MaybeUninit::uninit();
    let mut ans_n = 0;
    cublas!(cublasLtMatmulAlgoGetHeuristic(
        handle,
        caluculation,
        a.as_raw(),
        b.as_raw(),
        c.as_raw(),
        c.as_raw(),
        preference,
        1,
        result.as_mut_ptr(),
        &mut ans_n,
    ));
    assert_eq!(ans_n, 1);

    let result = unsafe { result.assume_init() };
    (result.algo, result.workspaceSize)
}

pub fn matmul(
    handle: cu::cublasLtHandle_t,
    caluculation: cu::cublasLtMatmulDesc_t,
    alpha: f32,
    a: (&CublasLtMatrix, *const c_void),
    b: (&CublasLtMatrix, *const c_void),
    beta: f32,
    c: (&CublasLtMatrix, *mut c_void),
    algo: cu::cublasLtMatmulAlgo_t,
    workspace: (usize, *mut c_void),
    stream: &Stream,
) {
    cublas!(cublasLtMatmul(
        handle,
        caluculation,
        &alpha as *const _ as _,
        a.1,
        a.0.as_raw(),
        b.1,
        b.0.as_raw(),
        &beta as *const _ as _,
        c.1,
        c.0.as_raw(),
        c.1,
        c.0.as_raw(),
        &algo,
        workspace.1,
        workspace.0,
        stream.as_raw() as _,
    ));
}
