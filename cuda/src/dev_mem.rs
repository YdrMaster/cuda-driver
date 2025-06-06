use crate::{Blob, CurrentCtx, Stream, bindings::CUdeviceptr};
use context_spore::{AsRaw, impl_spore};
use std::{
    alloc::Layout,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

#[repr(transparent)]
pub struct DevByte(u8);

#[inline]
pub fn memcpy_d2h<T: Copy>(dst: &mut [T], src: &[DevByte]) {
    let len = size_of_val(dst);
    let dst = dst.as_mut_ptr().cast();
    assert_eq!(len, size_of_val(src));
    driver!(cuMemcpyDtoH_v2(dst, src.as_ptr() as _, len))
}

#[inline]
pub fn memcpy_h2d<T: Copy>(dst: &mut [DevByte], src: &[T]) {
    let len = size_of_val(src);
    let src = src.as_ptr().cast();
    assert_eq!(len, size_of_val(dst));
    driver!(cuMemcpyHtoD_v2(dst.as_ptr() as _, src, len))
}

#[inline]
pub fn memcpy_d2d(dst: &mut [DevByte], src: &[DevByte]) {
    let len = size_of_val(src);
    assert_eq!(len, size_of_val(dst));
    driver!(cuMemcpyDtoD_v2(dst.as_ptr() as _, src.as_ptr() as _, len))
}

impl Stream<'_> {
    #[inline]
    pub fn memcpy_h2d<T: Copy>(&self, dst: &mut [DevByte], src: &[T]) -> &Self {
        let len = size_of_val(src);
        assert_eq!(len, size_of_val(dst));
        driver!(cuMemcpyHtoDAsync_v2(
            dst.as_mut_ptr() as _,
            src.as_ptr().cast(),
            len,
            self.as_raw()
        ));
        self
    }

    #[inline]
    pub fn memcpy_d2d(&self, dst: &mut [DevByte], src: &[DevByte]) -> &Self {
        let len = size_of_val(src);
        assert_eq!(len, size_of_val(dst));
        driver!(cuMemcpyDtoDAsync_v2(
            dst.as_mut_ptr() as _,
            src.as_ptr() as _,
            len,
            self.as_raw()
        ));
        self
    }

    #[inline]
    pub fn memcpy_d2h<T: Copy>(&self, dst: &mut [T], src: &[DevByte]) -> &Self {
        let len = size_of_val(src);
        assert_eq!(len, size_of_val(dst));
        driver!(cuMemcpyDtoHAsync_v2(
            dst.as_mut_ptr().cast(),
            src.as_ptr() as _,
            len,
            self.as_raw()
        ));
        self
    }
}

impl_spore!(DevMem and DevMemSpore by (CurrentCtx, Blob<CUdeviceptr>));

impl CurrentCtx {
    pub fn malloc<T: Copy>(&self, len: usize) -> DevMem<'_> {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = 0;
        if len != 0 {
            driver!(cuMemAlloc_v2(&mut ptr, len))
        }
        DevMem(unsafe { self.wrap_raw(Blob { ptr, len }) }, PhantomData)
    }

    pub fn from_host<T: Copy>(&self, slice: &[T]) -> DevMem<'_> {
        let mut dev = self.malloc::<T>(slice.len());
        memcpy_h2d(&mut dev, slice);
        dev
    }
}

impl<'ctx> Stream<'ctx> {
    pub fn malloc<T: Copy>(&self, len: usize) -> DevMem<'ctx> {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = 0;
        driver!(cuMemAllocAsync(&mut ptr, len, self.as_raw()));
        DevMem(
            unsafe { self.ctx().wrap_raw(Blob { ptr, len }) },
            PhantomData,
        )
    }

    pub fn from_host<T: Copy>(&self, slice: &[T]) -> DevMem<'ctx> {
        let stream = unsafe { self.as_raw() };
        let len = size_of_val(slice);
        let src = slice.as_ptr().cast();
        let mut ptr = 0;
        driver!(cuMemAllocAsync(&mut ptr, len, stream));
        driver!(cuMemcpyHtoDAsync_v2(ptr, src, len, stream));
        DevMem(
            unsafe { self.ctx().wrap_raw(Blob { ptr, len }) },
            PhantomData,
        )
    }

    pub fn free(&self, mem: DevMem) -> &Self {
        driver!(cuMemFreeAsync(mem.0.rss.ptr, self.as_raw()));
        std::mem::forget(mem);
        self
    }
}

impl Drop for DevMem<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.0.rss.ptr != 0 {
            driver!(cuMemFree_v2(self.0.rss.ptr))
        }
    }
}

impl Deref for DevMem<'_> {
    type Target = [DevByte];
    #[inline]
    fn deref(&self) -> &Self::Target {
        if self.0.rss.len == 0 {
            &[]
        } else {
            unsafe { from_raw_parts(self.0.rss.ptr as _, self.0.rss.len) }
        }
    }
}

impl DerefMut for DevMem<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.0.rss.len == 0 {
            &mut []
        } else {
            unsafe { from_raw_parts_mut(self.0.rss.ptr as _, self.0.rss.len) }
        }
    }
}

impl AsRaw for DevMemSpore {
    type Raw = CUdeviceptr;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss.ptr
    }
}

impl DevMemSpore {
    #[inline]
    pub const fn len(&self) -> usize {
        self.0.rss.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.0.rss.len == 0
    }
}
