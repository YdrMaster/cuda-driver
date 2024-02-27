use crate::{bindings as cuda, AsRaw, ContextGuard};
use std::ptr::null_mut;

pub struct Stream<'a>(cuda::CUstream, &'a ContextGuard<'a>);

impl ContextGuard<'_> {
    #[inline]
    pub fn stream(&self) -> Stream {
        let mut stream = null_mut();
        driver!(cuStreamCreate(&mut stream, 0));
        Stream(stream, self)
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

    #[inline]
    pub fn ctx(&self) -> &ContextGuard {
        self.1
    }
}
