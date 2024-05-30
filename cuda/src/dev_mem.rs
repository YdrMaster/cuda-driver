use crate::{
    bindings as cuda, context::ResourceOwnership, not_owned, owned, spore_convention, AsRaw,
    ContextGuard, ContextResource, ContextSpore, Stream,
};
use std::{
    alloc::Layout,
    mem::{forget, size_of_val, take},
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

pub struct DevMem<'ctx> {
    ptr: cuda::CUdeviceptr,
    len: usize,
    ownership: ResourceOwnership<'ctx>,
}

impl ContextGuard<'_> {
    pub fn malloc<T: Copy>(&self, len: usize) -> DevMem<'_> {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = 0;
        driver!(cuMemAlloc_v2(&mut ptr, len));
        DevMem {
            ptr,
            len,
            ownership: owned(self),
        }
    }

    pub fn from_host<T: Copy>(&self, slice: &[T]) -> DevMem<'_> {
        let len = size_of_val(slice);
        let src = slice.as_ptr().cast();
        let mut ptr = 0;
        driver!(cuMemAlloc_v2(&mut ptr, len));
        driver!(cuMemcpyHtoD_v2(ptr, src, len));
        DevMem {
            ptr,
            len,
            ownership: owned(self),
        }
    }
}

impl<'ctx> Stream<'ctx> {
    pub fn malloc<T: Copy>(&self, len: usize) -> DevMem<'ctx> {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = 0;
        driver!(cuMemAllocAsync(&mut ptr, len, self.as_raw()));
        DevMem {
            ptr,
            len,
            ownership: owned(self.ctx()),
        }
    }

    pub fn from_host<T: Copy>(&self, slice: &[T]) -> DevMem<'ctx> {
        let stream = unsafe { self.as_raw() };
        let len = size_of_val(slice);
        let src = slice.as_ptr().cast();
        let mut ptr = 0;
        driver!(cuMemAllocAsync(&mut ptr, len, stream));
        driver!(cuMemcpyHtoDAsync_v2(ptr, src, len, stream));
        DevMem {
            ptr,
            len,
            ownership: owned(self.ctx()),
        }
    }
}

impl DevMem<'_> {
    #[inline]
    pub fn drop_on(self, stream: &Stream) {
        if self.ownership.is_owned() {
            driver!(cuMemFreeAsync(self.ptr, stream.as_raw()));
            forget(self);
        }
    }
}

impl Drop for DevMem<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.ownership.is_owned() {
            driver!(cuMemFree_v2(self.ptr));
        }
    }
}

impl Deref for DevMem<'_> {
    type Target = [DevByte];
    #[inline]
    fn deref(&self) -> &Self::Target {
        if self.len == 0 {
            &[]
        } else {
            unsafe { from_raw_parts(self.ptr as _, self.len) }
        }
    }
}

impl DerefMut for DevMem<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.len == 0 {
            &mut []
        } else {
            unsafe { from_raw_parts_mut(self.ptr as _, self.len) }
        }
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct DevMemSpore {
    ptr: cuda::CUdeviceptr,
    len: usize,
}

spore_convention!(DevMemSpore);

impl DevMemSpore {
    /// # Safety
    ///
    /// This function must be called in the same context as the one that created the resource.
    #[inline]
    pub unsafe fn kill_on(&mut self, stream: &Stream) {
        driver!(cuMemFreeAsync(take(&mut self.ptr), stream.as_raw()));
    }
}

impl ContextSpore for DevMemSpore {
    type Resource<'ctx> = DevMem<'ctx>;

    #[inline]
    unsafe fn sprout<'ctx>(&self, ctx: &'ctx ContextGuard) -> Self::Resource<'ctx> {
        DevMem {
            ptr: self.ptr,
            len: self.len,
            ownership: not_owned(ctx),
        }
    }

    #[inline]
    unsafe fn kill(&mut self, ctx: &ContextGuard) {
        drop(DevMem {
            ptr: take(&mut self.ptr),
            len: self.len,
            ownership: owned(ctx),
        });
    }

    #[inline]
    fn is_alive(&self) -> bool {
        self.ptr != cuda::CUdeviceptr::default()
    }
}

impl<'ctx> ContextResource<'ctx> for DevMem<'ctx> {
    type Spore = DevMemSpore;

    #[inline]
    fn sporulate(self) -> Self::Spore {
        let ans = DevMemSpore {
            ptr: self.ptr,
            len: self.len,
        };
        forget(self);
        ans
    }
}
