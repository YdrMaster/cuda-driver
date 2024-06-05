use crate::bindings as cublas;
use cuda::{impl_spore, AsRaw, ContextGuard};
use std::{marker::PhantomData, ptr::null_mut};

impl_spore!(Cublas and CublasSpore by cublas::cublasHandle_t);

impl Drop for Cublas<'_> {
    #[inline]
    fn drop(&mut self) {
        cublas!(cublas::cublasDestroy_v2(self.0.res));
    }
}

impl AsRaw for Cublas<'_> {
    type Raw = cublas::cublasHandle_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.res
    }
}

impl<'ctx> Cublas<'ctx> {
    #[inline]
    pub fn new(ctx: &'ctx ContextGuard) -> Self {
        let mut handle = null_mut();
        cublas!(cublas::cublasCreate_v2(&mut handle));
        Self(unsafe { ctx.wrap_resource(handle) }, PhantomData)
    }

    #[inline]
    pub fn set_stream(&self, stream: &cuda::Stream) {
        cublas!(cublas::cublasSetStream_v2(
            self.0.res,
            stream.as_raw().cast()
        ));
    }
}
