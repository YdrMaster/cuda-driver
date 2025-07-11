use crate::{CurrentCtx, Dim3, KernelFn, bindings::MCstream};
use context_spore::{AsRaw, impl_spore};
use std::{ffi::c_void, marker::PhantomData, ptr::null_mut};

impl_spore!(Stream and StreamSpore by (CurrentCtx, MCstream));

impl CurrentCtx {
    #[inline]
    pub fn stream(&self) -> Stream {
        let mut stream = null_mut();
        driver!(mcStreamCreate(&mut stream));
        Stream(unsafe { self.wrap_raw(stream) }, PhantomData)
    }
}

impl Drop for Stream<'_> {
    #[inline]
    fn drop(&mut self) {
        self.synchronize();
        driver!(mcStreamDestroy(self.0.rss))
    }
}

impl AsRaw for Stream<'_> {
    type Raw = MCstream;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss
    }
}

impl Stream<'_> {
    pub fn launch(
        &self,
        f: &KernelFn,
        attrs: (impl Into<Dim3>, impl Into<Dim3>, usize),
        params: &[*const c_void],
    ) -> &Self {
        let (grid, block, shared_mem) = attrs;
        let grid = grid.into();
        let block = block.into();
        driver!(mcModuleLaunchKernel(
            f.as_raw(),
            grid.x,
            grid.y,
            grid.z,
            block.x,
            block.y,
            block.z,
            shared_mem as _,
            self.0.rss,
            params.as_ptr() as _,
            null_mut(),
        ));
        self
    }

    #[inline]
    pub fn synchronize(&self) -> &Self {
        driver!(mcStreamSynchronize(self.0.rss));
        self
    }
}
