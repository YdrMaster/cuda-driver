mod matrix;
mod multiply;

use crate::bindings::{hcblasLtHandle_t, hcblasLtMatmulAlgo_t, hpccDataType_t};
use cuda::{AsRaw, CurrentCtx, DevByte, Stream, impl_spore};
use std::{ffi::c_void, marker::PhantomData, mem::size_of_val, ptr::null_mut};

pub use matrix::{CublasLtMatrix, CublasLtMatrixLayout, MatrixOrder};
pub use multiply::CublasLtMatMulDescriptor;

impl_spore!(CublasLt and CublasLtSpore by (CurrentCtx, hcblasLtHandle_t));

impl Drop for CublasLt<'_> {
    #[inline]
    fn drop(&mut self) {
        cublas!(hcblasLtDestroy(self.0.rss));
    }
}

impl AsRaw for CublasLt<'_> {
    type Raw = hcblasLtHandle_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss
    }
}

#[derive(Clone, Copy)]
pub struct CublasLtMatMulLayout<'a> {
    mat_mul: &'a CublasLtMatMulDescriptor,
    a: &'a CublasLtMatrix,
    b: &'a CublasLtMatrix,
    c: &'a CublasLtMatrix,
    d: &'a CublasLtMatrix,
}

impl CublasLt<'_> {
    #[inline]
    pub fn new(ctx: &CurrentCtx) -> Self {
        let mut handle = null_mut();
        cublas!(hcblasLtCreate(&mut handle));
        Self(unsafe { ctx.wrap_raw(handle) }, PhantomData)
    }

    pub fn tune(
        &self,
        layout: CublasLtMatMulLayout,
        max_workspace: usize,
        request_count: usize,
    ) -> Vec<(hcblasLtMatmulAlgo_t, usize)> {
        let workspace = max_workspace as u64;

        let mut preference = null_mut();
        cublas!(hcblasLtMatmulPreferenceCreate(&mut preference));
        cublas!(hcblasLtMatmulPreferenceSetAttribute(
            preference,
            hcblasLtMatmulPreferenceAttributes_t::HCBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
            (&raw const workspace).cast(),
            size_of_val(&workspace),
        ));

        let mut result = Vec::with_capacity(request_count);
        #[allow(clippy::uninit_vec)]
        unsafe {
            result.set_len(request_count);
        }

        let mut ans_n = 0;
        cublas!(hcblasLtMatmulAlgoGetHeuristic(
            self.0.rss,
            layout.mat_mul.as_raw(),
            layout.a.as_raw(),
            layout.b.as_raw(),
            layout.c.as_raw(),
            layout.d.as_raw(),
            preference,
            request_count as _,
            result.as_mut_ptr(),
            &mut ans_n,
        ));

        result
            .into_iter()
            .take(ans_n as _)
            .map(|r| (r.algo, r.workspaceSize))
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn mat_mul(
        &self,

        layout: CublasLtMatMulLayout,
        algo: hcblasLtMatmulAlgo_t,

        d_ptr: *mut DevByte,
        alpha: f32,
        a_ptr: *const DevByte,
        b_ptr: *const DevByte,
        beta: f32,
        c_ptr: *const DevByte,

        workspace: &mut [DevByte],
        stream: &Stream,
    ) {
        let mut dt = hpccDataType_t::HPCC_R_16F;
        let mut written = 0;
        cublas!(hcblasLtMatmulDescGetAttribute(
            layout.mat_mul.as_raw(),
            hcblasLtMatmulDescAttributes_t::HCBLASLT_MATMUL_DESC_SCALE_TYPE,
            &mut dt as *mut _ as _,
            size_of_val(&dt),
            &mut written,
        ));
        assert_eq!(written, size_of_val(&dt));

        cublas!(hcblasLtMatmul(
            self.0.rss,
            layout.mat_mul.as_raw(),
            FloatScalar::convert(alpha, dt).as_ptr(),
            a_ptr.cast(),
            layout.a.as_raw(),
            b_ptr.cast(),
            layout.b.as_raw(),
            FloatScalar::convert(beta, dt).as_ptr(),
            c_ptr.cast(),
            layout.c.as_raw(),
            d_ptr.cast(),
            layout.d.as_raw(),
            &algo,
            workspace.as_mut_ptr().cast(),
            workspace.len(),
            stream.as_raw() as _,
        ));
    }
}

#[repr(transparent)]
struct FloatScalar([u8; 8]);

impl FloatScalar {
    fn new<T>(val: T) -> Self {
        let mut ans = Self([0; 8]);
        debug_assert!(size_of_val(&val) <= ans.0.len());
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&raw const val).cast(),
                ans.0.as_mut_ptr(),
                size_of_val(&val),
            )
        };
        ans
    }

    fn convert(val: f32, target: hpccDataType_t) -> Self {
        match target {
            hpccDataType_t::HPCC_R_16F => Self::new(half::f16::from_f32(val)),
            hpccDataType_t::HPCC_R_16BF => Self::new(half::bf16::from_f32(val)),
            hpccDataType_t::HPCC_R_32F => Self::new(val),
            hpccDataType_t::HPCC_R_64F => Self::new(val as f64),
            _ => unimplemented!(),
        }
    }

    #[inline]
    fn as_ptr(&self) -> *const c_void {
        self.0.as_ptr().cast()
    }
}

// #[cfg(test)]
// mod test;
