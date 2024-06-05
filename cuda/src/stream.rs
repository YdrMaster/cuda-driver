use crate::{bindings as cuda, impl_spore, AsRaw, ContextGuard, ResourceWrapper};
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

    #[inline]
    pub unsafe fn wrap_resource<T>(&self, res: T) -> ResourceWrapper<T> {
        ResourceWrapper {
            ctx: self.0.ctx,
            res,
        }
    }
}
