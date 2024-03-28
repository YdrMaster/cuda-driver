use crate::bindings as cublas;
use cuda::{
    ctx_eq, not_owned, owned, spore_convention, AsRaw, ContextGuard, ContextResource, ContextSpore,
    ResourceOwnership,
};
use std::{
    mem::{forget, replace},
    ptr::null_mut,
};

pub struct Cublas<'ctx>(cublas::cublasHandle_t, ResourceOwnership<'ctx>);

impl Drop for Cublas<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.1.is_owned() {
            cublas!(cublas::cublasDestroy_v2(self.0));
        }
    }
}

impl AsRaw for Cublas<'_> {
    type Raw = cublas::cublasHandle_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl<'ctx> Cublas<'ctx> {
    #[inline]
    pub fn new(ctx: &'ctx ContextGuard) -> Self {
        let mut handle = null_mut();
        cublas!(cublas::cublasCreate_v2(&mut handle));
        Self(handle, owned(ctx))
    }

    #[inline]
    pub fn set_stream(&self, stream: &cuda::Stream) {
        assert!(ctx_eq(self.1.ctx(), stream.ctx()));
        cublas!(cublas::cublasSetStream_v2(self.0, stream.as_raw().cast()));
    }
}

#[derive(PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct CublasSpore(cublas::cublasHandle_t);

spore_convention!(CublasSpore);

impl ContextSpore for CublasSpore {
    type Resource<'ctx> = Cublas<'ctx>;

    #[inline]
    unsafe fn sprout<'ctx>(&self, ctx: &'ctx ContextGuard) -> Self::Resource<'ctx> {
        Cublas(self.0, not_owned(ctx))
    }

    #[inline]
    unsafe fn kill(&mut self, ctx: &ContextGuard) {
        drop(Cublas(replace(&mut self.0, null_mut()), owned(ctx)));
    }

    #[inline]
    fn is_alive(&self) -> bool {
        !self.0.is_null()
    }
}

impl<'ctx> ContextResource<'ctx> for Cublas<'ctx> {
    type Spore = CublasSpore;

    #[inline]
    fn sporulate(self) -> Self::Spore {
        let handle = self.0;
        forget(self);
        CublasSpore(handle)
    }
}
