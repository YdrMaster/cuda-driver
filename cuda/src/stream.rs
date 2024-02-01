use super::{bindings as cuda, context::ContextGuard, AsRaw};
use std::{marker::PhantomData, ptr::null_mut};

pub struct Stream<'a>(cuda::CUstream, PhantomData<&'a ()>);

impl ContextGuard<'_> {
    pub fn stream(&self) -> Stream {
        let mut stream: cuda::CUstream = null_mut();
        driver!(cuStreamCreate(&mut stream, 0));
        Stream(stream, PhantomData)
    }
}

impl Drop for Stream<'_> {
    #[inline]
    fn drop(&mut self) {
        self.synchronize();
        driver!(cuStreamDestroy_v2(self.0));
    }
}

impl AsRaw for Stream<'_> {
    type Raw = cuda::CUstream;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Stream<'_> {
    #[inline]
    pub fn synchronize(&self) {
        driver!(cuStreamSynchronize(self.0));
    }
}
