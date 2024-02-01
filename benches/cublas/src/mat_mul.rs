use crate::{
    bindings::{self as cu, cublas},
    matrix::CublasLtMatrix,
};
use cuda::{driver, AsRaw, Stream};
use std::ptr::null_mut;
use std::{
    ffi::c_void,
    mem::{size_of_val, MaybeUninit},
};

#[allow(dead_code)]
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

#[allow(dead_code)]
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
