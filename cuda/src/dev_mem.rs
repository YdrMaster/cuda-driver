use crate::{Blob, CurrentCtx, Stream, bindings::hcDeviceptr_t};
use context_spore::{AsRaw, impl_spore};
use std::{
    alloc::Layout,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
};

#[repr(transparent)]
pub struct DevByte(u8);

#[inline]
pub fn memcpy_d2h<T: Copy>(dst: &mut [T], src: &[DevByte]) {
    let len = size_of_val(dst);
    let dst = dst.as_mut_ptr().cast();
    assert_eq!(len, size_of_val(src));
    driver!(hcMemcpyDtoH(dst, src.as_ptr() as _, len))
}

#[inline]
pub fn memcpy_h2d<T: Copy>(dst: &mut [DevByte], src: &[T]) {
    let len = size_of_val(src);
    let src = src.as_ptr().cast();
    assert_eq!(len, size_of_val(dst));
    driver!(hcMemcpyHtoD(dst.as_ptr() as _, src, len))
}

#[inline]
pub fn memcpy_d2d(dst: &mut [DevByte], src: &[DevByte]) {
    let len = size_of_val(src);
    assert_eq!(len, size_of_val(dst));
    driver!(hcMemcpyDtoD(dst.as_ptr() as _, src.as_ptr() as _, len))
}

impl Stream<'_> {
    #[inline]
    pub fn memcpy_h2d<T: Copy>(&self, dst: &mut [DevByte], src: &[T]) -> &Self {
        let len = size_of_val(src);
        assert_eq!(len, size_of_val(dst));
        driver!(hcMemcpyHtoDAsync(
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
        driver!(hcMemcpyDtoDAsync(
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
        driver!(hcMemcpyDtoHAsync(
            dst.as_mut_ptr().cast(),
            src.as_ptr() as _,
            len,
            self.as_raw()
        ));
        self
    }
}

impl_spore!(DevMem and DevMemSpore by (CurrentCtx, Blob<hcDeviceptr_t>));

impl CurrentCtx {
    pub fn malloc<T: Copy>(&self, len: usize) -> DevMem<'_> {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = null_mut();
        if len != 0 {
            driver!(hcMalloc(&mut ptr, len))
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
        let mut ptr = null_mut();
        driver!(hcMemAllocAsync(&mut ptr, len, self.as_raw()));
        DevMem(
            unsafe { self.ctx().wrap_raw(Blob { ptr, len }) },
            PhantomData,
        )
    }

    pub fn from_host<T: Copy>(&self, slice: &[T]) -> DevMem<'ctx> {
        let stream = unsafe { self.as_raw() };
        let len = size_of_val(slice);
        let src = slice.as_ptr().cast();
        let mut ptr = null_mut();
        driver!(hcMemAllocAsync(&mut ptr, len, stream));
        driver!(hcMemcpyHtoDAsync(ptr, src, len, stream));
        DevMem(
            unsafe { self.ctx().wrap_raw(Blob { ptr, len }) },
            PhantomData,
        )
    }

    pub fn free(&self, mem: DevMem) -> &Self {
        driver!(hcMemFreeAsync(mem.0.rss.ptr, self.as_raw()));
        std::mem::forget(mem);
        self
    }
}

impl Drop for DevMem<'_> {
    #[inline]
    fn drop(&mut self) {
        if !self.0.rss.ptr.is_null() {
            driver!(hcFree(self.0.rss.ptr))
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
    type Raw = hcDeviceptr_t;
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
