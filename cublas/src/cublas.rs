use crate::bindings::cublasHandle_t;
use cuda::{impl_spore, AsRaw, CurrentCtx, Stream};
use std::{marker::PhantomData, ptr::null_mut};

impl_spore!(Cublas and CublasSpore by cublasHandle_t);

impl Drop for Cublas<'_> {
    #[inline]
    fn drop(&mut self) {
        cublas!(cublasDestroy_v2(self.0.raw));
    }
}

impl AsRaw for Cublas<'_> {
    type Raw = cublasHandle_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.raw
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
        cublas!(cublasSetStream_v2(self.0.raw, stream.as_raw().cast()));
    }
}
