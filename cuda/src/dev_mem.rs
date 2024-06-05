use crate::{bindings as cuda, impl_spore, AsRaw, Blob, ContextGuard, Stream};
use std::{
    alloc::Layout,
    marker::PhantomData,
    mem::{forget, size_of_val},
    ops::{Deref, DerefMut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

#[repr(transparent)]
pub struct DevByte(#[allow(unused)] u8);

#[inline]
pub fn memcpy_d2h<T: Copy>(dst: &mut [T], src: &[DevByte]) {
    let len = size_of_val(dst);
    let dst = dst.as_mut_ptr().cast();
    assert_eq!(len, size_of_val(src));
    driver!(cuMemcpyDtoH_v2(dst, src.as_ptr() as _, len));
}

#[inline]
pub fn memcpy_h2d<T: Copy>(dst: &mut [DevByte], src: &[T]) {
    let len = size_of_val(src);
    let src = src.as_ptr().cast();
    assert_eq!(len, size_of_val(dst));
    driver!(cuMemcpyHtoD_v2(dst.as_ptr() as _, src, len));
}

#[inline]
pub fn memcpy_d2d(dst: &mut [DevByte], src: &[DevByte]) {
    let len = size_of_val(src);
    assert_eq!(len, size_of_val(dst));
    driver!(cuMemcpyDtoD_v2(dst.as_ptr() as _, src.as_ptr() as _, len));
}

impl Stream<'_> {
    #[inline]
    pub fn memcpy_h2d<T: Copy>(&self, dst: &mut [DevByte], src: &[T]) {
        let len = size_of_val(src);
        let src = src.as_ptr().cast();
        assert_eq!(len, size_of_val(dst));
        driver!(cuMemcpyHtoDAsync_v2(
            dst.as_ptr() as _,
            src,
            len,
            self.as_raw()
        ));
    }

    #[inline]
    pub fn memcpy_d2d(&self, dst: &mut [DevByte], src: &[DevByte]) {
        let len = size_of_val(src);
        assert_eq!(len, size_of_val(dst));
        driver!(cuMemcpyDtoDAsync_v2(
            dst.as_ptr() as _,
            src.as_ptr() as _,
            len,
            self.as_raw()
        ));
    }
}

impl_spore!(DevMem and DevMemSpore by Blob<cuda::CUdeviceptr>);

impl ContextGuard<'_> {
    pub fn malloc<T: Copy>(&self, len: usize) -> DevMem<'_> {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = 0;
        driver!(cuMemAlloc_v2(&mut ptr, len));
        DevMem(
            unsafe { self.wrap_resource(Blob { ptr, len }) },
            PhantomData,
        )
    }

    pub fn from_host<T: Copy>(&self, slice: &[T]) -> DevMem<'_> {
        let len = size_of_val(slice);
        let src = slice.as_ptr().cast();
        let mut ptr = 0;
        driver!(cuMemAlloc_v2(&mut ptr, len));
        driver!(cuMemcpyHtoD_v2(ptr, src, len));
        DevMem(
            unsafe { self.wrap_resource(Blob { ptr, len }) },
            PhantomData,
        )
    }
}

impl<'ctx> Stream<'ctx> {
    pub fn malloc<T: Copy>(&self, len: usize) -> DevMem<'ctx> {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = 0;
        driver!(cuMemAllocAsync(&mut ptr, len, self.as_raw()));
        DevMem(
            unsafe { self.wrap_resource(Blob { ptr, len }) },
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
            unsafe { self.wrap_resource(Blob { ptr, len }) },
            PhantomData,
        )
    }
}

impl DevMem<'_> {
    #[inline]
    pub fn drop_on(self, stream: &Stream) {
        driver!(cuMemFreeAsync(self.0.res.ptr, stream.as_raw()));
        forget(self);
    }
}

impl Drop for DevMem<'_> {
    #[inline]
    fn drop(&mut self) {
        driver!(cuMemFree_v2(self.0.res.ptr));
    }
}

impl Deref for DevMem<'_> {
    type Target = [DevByte];
    #[inline]
    fn deref(&self) -> &Self::Target {
        if self.0.res.len == 0 {
            &[]
        } else {
            unsafe { from_raw_parts(self.0.res.ptr as _, self.0.res.len) }
        }
    }
}

impl DerefMut for DevMem<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.0.res.len == 0 {
            &mut []
        } else {
            unsafe { from_raw_parts_mut(self.0.res.ptr as _, self.0.res.len) }
        }
    }
}
