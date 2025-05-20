mod param;

use crate::bindings::{cublasHandle_t, cublasOperation_t};
use cuda::{AsRaw, CurrentCtx, DevByte, Stream, impl_spore};
use std::{marker::PhantomData, ptr::null_mut};

pub use param::{Computation, GemmScheme};

impl_spore!(Cublas and CublasSpore by (CurrentCtx, cublasHandle_t));

impl Drop for Cublas<'_> {
    #[inline]
    fn drop(&mut self) {
        cublas!(cublasDestroy_v2(self.0.rss))
    }
}

impl AsRaw for Cublas<'_> {
    type Raw = cublasHandle_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss
    }
}

impl Cublas<'_> {
    #[inline]
    pub fn new(ctx: &CurrentCtx) -> Self {
        let mut handle = null_mut();
        cublas!(cublasCreate_v2(&mut handle));
        Self(unsafe { ctx.wrap_raw(handle) }, PhantomData)
    }

    #[inline]
    pub fn bind(stream: &Stream) -> Self {
        let mut ans = Self::new(stream.ctx());
        ans.set_stream(stream);
        ans
    }

    #[inline]
    pub fn set_stream(&mut self, stream: &Stream) {
        cublas!(cublasSetStream_v2(self.0.rss, stream.as_raw().cast()))
    }

    /// 调用 cublas 矩阵乘
    ///
    /// # Safety
    ///
    /// 这个函数使用指向显存的裸指针。
    // 对 C API 的封装
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm(
        &self,
        m: usize,
        n: usize,
        k: usize,
        scalar: impl Computation,
        a: *const DevByte,
        trans_a: bool,
        lda: isize,
        b: *const DevByte,
        trans_b: bool,
        ldb: isize,
        c: *mut DevByte,
        ldc: isize,
    ) {
        cublas!(cublasGemmEx(
            self.0.rss,
            op(trans_a),
            op(trans_b),
            m as _,
            n as _,
            k as _,
            scalar.alpha(),
            a.cast(),
            scalar.a_type(),
            lda as _,
            b.cast(),
            scalar.b_type(),
            ldb as _,
            scalar.beta(),
            c.cast(),
            scalar.c_type(),
            ldc as _,
            scalar.compute_type(),
            cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
        ))
    }

    /// 调用 cublas 批量矩阵乘
    ///
    /// # Safety
    ///
    /// 这个函数使用指向显存的裸指针。
    // 对 C API 的封装
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn gemm_batched(
        &self,
        m: usize,
        n: usize,
        k: usize,
        scalar: impl Computation,
        a: *const DevByte,
        trans_a: bool,
        lda: isize,
        b: *const DevByte,
        trans_b: bool,
        ldb: isize,
        c: *mut DevByte,
        ldc: isize,

        batch: usize,
        stride_c: isize,
        stride_a: isize,
        stride_b: isize,
    ) {
        cublas!(cublasGemmStridedBatchedEx(
            self.0.rss,
            op(trans_a),
            op(trans_b),
            m as _,
            n as _,
            k as _,
            scalar.alpha(),
            a.cast(),
            scalar.a_type(),
            lda as _,
            stride_a as _,
            b.cast(),
            scalar.b_type(),
            ldb as _,
            stride_b as _,
            scalar.beta(),
            c.cast(),
            scalar.c_type(),
            ldc as _,
            stride_c as _,
            batch as _,
            scalar.compute_type(),
            cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
        ))
    }
}

fn op(trans: bool) -> cublasOperation_t {
    if trans {
        cublasOperation_t::CUBLAS_OP_T
    } else {
        cublasOperation_t::CUBLAS_OP_N
    }
}

#[cfg(test)]
mod test;
