use super::ptx::Ptx;
use crate::{bindings as cuda, impl_spore, AsRaw, ContextGuard};
use std::{marker::PhantomData, ptr::null_mut};

impl_spore!(Module and ModuleSpore by cuda::CUmodule);

impl ContextGuard<'_> {
    #[inline]
    pub fn load<'ctx>(&'ctx self, ptx: &Ptx) -> Module<'ctx> {
        let mut module = null_mut();
        driver!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
        Module(unsafe { self.wrap_resource(module) }, PhantomData)
    }
}

impl Drop for Module<'_> {
    #[inline]
    fn drop(&mut self) {
        driver!(cuModuleUnload(self.0.res));
    }
}

impl AsRaw for Module<'_> {
    type Raw = cuda::CUmodule;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.res
    }
}
