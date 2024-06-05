use crate::{bindings as cuda, impl_spore, AsRaw, ContextGuard};
use std::{marker::PhantomData, ptr::null_mut};

impl_spore!(Stream and StreamSpore by cuda::CUstream);

impl ContextGuard<'_> {
    #[inline]
    pub fn stream(&self) -> Stream {
        let mut stream = null_mut();
        driver!(cuStreamCreate(&mut stream, 0));
        Stream(unsafe { self.wrap_resource(stream) }, PhantomData)
    }
}

impl Drop for Stream<'_> {
    #[inline]
    fn drop(&mut self) {
        self.synchronize();
        driver!(cuStreamDestroy_v2(self.0.res));
    }
}

impl AsRaw for Stream<'_> {
    type Raw = cuda::CUstream;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.res
    }
}

impl Stream<'_> {
    #[inline]
    pub fn synchronize(&self) {
        driver!(cuStreamSynchronize(self.0.res));
    }
}

impl<'ctx> Stream<'ctx> {
    #[inline]
    pub fn ctx(&self) -> &ContextGuard<'ctx> {
        unsafe { std::mem::transmute(&self.0.ctx) }
    }
}
