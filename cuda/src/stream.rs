use crate::{
    bindings as cuda, context::ResourceOwnership, not_owned, owned, spore_convention, AsRaw,
    ContextGuard, ContextResource, ContextSpore,
};
use std::{mem::replace, ptr::null_mut};

pub struct Stream<'ctx>(cuda::CUstream, ResourceOwnership<'ctx>);

impl ContextGuard<'_> {
    #[inline]
    pub fn stream(&self) -> Stream {
        let mut stream = null_mut();
        driver!(cuStreamCreate(&mut stream, 0));
        Stream(stream, owned(self))
    }
}

impl Drop for Stream<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.1.is_owned() {
            self.synchronize();
            driver!(cuStreamDestroy_v2(self.0));
        }
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
        self.1.ctx()
    }
}

#[derive(PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct StreamSpore(cuda::CUstream);

spore_convention!(StreamSpore);

impl ContextSpore for StreamSpore {
    type Resource<'ctx> = Stream<'ctx>;

    #[inline]
    unsafe fn sprout<'ctx>(&'ctx self, ctx: &'ctx ContextGuard) -> Self::Resource<'ctx> {
        Stream(self.0, not_owned(ctx))
    }

    #[inline]
    unsafe fn kill(&mut self, ctx: &ContextGuard) {
        drop(Stream(replace(&mut self.0, null_mut()), owned(ctx)));
    }

    #[inline]
    fn is_alive(&self) -> bool {
        !self.0.is_null()
    }
}

impl<'ctx> ContextResource<'ctx> for Stream<'ctx> {
    type Spore = StreamSpore;

    #[inline]
    fn sporulate(self) -> Self::Spore {
        StreamSpore(self.0)
    }
}
