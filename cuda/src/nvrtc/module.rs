use super::ptx::Ptx;
use crate::{CurrentCtx, bindings::CUmodule};
use context_spore::{AsRaw, impl_spore};
use std::{marker::PhantomData, ptr::null_mut};

impl_spore!(Module and ModuleSpore by (CurrentCtx, CUmodule));

impl CurrentCtx {
    #[inline]
    pub fn load(&self, ptx: &Ptx) -> Module {
        let mut module = null_mut();
        driver!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
        Module(unsafe { self.wrap_raw(module) }, PhantomData)
    }
}

impl Drop for Module<'_> {
    #[inline]
    fn drop(&mut self) {
        driver!(cuModuleUnload(self.0.rss))
    }
}

impl AsRaw for Module<'_> {
    type Raw = CUmodule;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss
    }
}
