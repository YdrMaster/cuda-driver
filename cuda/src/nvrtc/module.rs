use super::ptx::Ptx;
use crate::{
    bindings as cuda, context::ResourceOwnership, not_owned, owned, spore_convention, AsRaw,
    ContextGuard, ContextResource, ContextSpore,
};
use std::{mem::replace, ptr::null_mut};

pub struct Module<'ctx>(cuda::CUmodule, ResourceOwnership<'ctx>);

impl ContextGuard<'_> {
    #[inline]
    pub fn load<'ctx>(&'ctx self, ptx: &Ptx) -> Module<'ctx> {
        let mut module = null_mut();
        driver!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
        Module(module, owned(self))
    }
}

impl Drop for Module<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.1.is_owned() {
            driver!(cuModuleUnload(self.0));
        }
    }
}

impl AsRaw for Module<'_> {
    type Raw = cuda::CUmodule;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

#[derive(PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct ModuleSpore(cuda::CUmodule);

spore_convention!(ModuleSpore);

impl ContextSpore for ModuleSpore {
    type Resource<'ctx> = Module<'ctx>;

    #[inline]
    unsafe fn sprout<'ctx>(&self, ctx: &'ctx ContextGuard) -> Self::Resource<'ctx> {
        Module(self.0, not_owned(ctx))
    }

    #[inline]
    unsafe fn kill(&mut self, ctx: &ContextGuard) {
        drop(Module(replace(&mut self.0, null_mut()), owned(ctx)));
    }

    #[inline]
    fn is_alive(&self) -> bool {
        !self.0.is_null()
    }
}

impl<'ctx> ContextResource<'ctx> for Module<'ctx> {
    type Spore = ModuleSpore;

    #[inline]
    fn sporulate(self) -> Self::Spore {
        ModuleSpore(self.0)
    }
}
