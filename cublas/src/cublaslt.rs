use crate::bindings as cublas;
use cuda::{
    not_owned, owned, spore_convention, AsRaw, ContextGuard, ContextResource, ContextSpore,
    ResourceOwnership,
};
use std::{mem::replace, ptr::null_mut};

pub struct CublasLt<'ctx>(cublas::cublasLtHandle_t, ResourceOwnership<'ctx>);

impl Drop for CublasLt<'_> {
    #[inline]
    fn drop(&mut self) {
        cublas!(cublas::cublasLtDestroy(self.0));
    }
}

impl AsRaw for CublasLt<'_> {
    type Raw = cublas::cublasLtHandle_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl<'ctx> CublasLt<'ctx> {
    #[inline]
    pub fn new(ctx: &'ctx ContextGuard) -> Self {
        let mut handle = null_mut();
        cublas!(cublas::cublasLtCreate(&mut handle));
        Self(handle, owned(ctx))
    }
}

#[derive(PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct CublasLtSpore(cublas::cublasLtHandle_t);

spore_convention!(CublasLtSpore);

impl ContextSpore for CublasLtSpore {
    type Resource<'ctx> = CublasLt<'ctx>;

    #[inline]
    unsafe fn sprout<'ctx>(&'ctx self, ctx: &'ctx ContextGuard) -> Self::Resource<'ctx> {
        CublasLt(self.0, not_owned(ctx))
    }

    #[inline]
    unsafe fn kill(&mut self, ctx: &ContextGuard) {
        drop(CublasLt(replace(&mut self.0, null_mut()), owned(ctx)));
    }

    #[inline]
    fn is_alive(&self) -> bool {
        !self.0.is_null()
    }
}

impl<'ctx> ContextResource<'ctx> for CublasLt<'ctx> {
    type Spore = CublasLtSpore;

    #[inline]
    fn sporulate(self) -> Self::Spore {
        CublasLtSpore(self.0)
    }
}
