use crate::bindings as cublas;
use cuda::{impl_spore, AsRaw, ContextGuard};
use std::{marker::PhantomData, ptr::null_mut};

impl_spore!(CublasLt and CublasLtSpore by cublas::cublasLtHandle_t);

impl Drop for CublasLt<'_> {
    #[inline]
    fn drop(&mut self) {
        cublas!(cublas::cublasLtDestroy(self.0.res));
    }
}

impl AsRaw for CublasLt<'_> {
    type Raw = cublas::cublasLtHandle_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.res
    }
}

impl<'ctx> CublasLt<'ctx> {
    #[inline]
    pub fn new(ctx: &'ctx ContextGuard) -> Self {
        let mut handle = null_mut();
        cublas!(cublas::cublasLtCreate(&mut handle));
        Self(unsafe { ctx.wrap_resource(handle) }, PhantomData)
    }
}
