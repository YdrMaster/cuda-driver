use crate::{
    bindings as cuda, context::ResourceOwnership, not_owned, owned, spore_convention, AsRaw,
    ContextGuard, ContextResource, ContextSpore, Stream,
};
use std::{
    alloc::Layout,
    cell::UnsafeCell,
    mem::{forget, size_of_val, take},
    ops::{Deref, DerefMut},
};

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct DevSlice {
    ptr: cuda::CUdeviceptr,
    len: usize,
}

impl AsRaw for DevSlice {
    type Raw = cuda::CUdeviceptr;

    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.ptr
    }
}

impl DevSlice {
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn copy_out<T: Copy>(&self, slice: &mut [T]) {
        let len = size_of_val(slice);
        let dst = slice.as_mut_ptr().cast();
        assert_eq!(len, self.len);
        driver!(cuMemcpyDtoH_v2(dst, self.ptr, len));
    }

    #[inline]
    pub fn copy_in_async<T: Copy>(&mut self, slice: &[T], stream: &Stream) {
        let len = size_of_val(slice);
        let src = slice.as_ptr().cast();
        assert_eq!(len, self.len);
        driver!(cuMemcpyHtoDAsync_v2(self.ptr, src, len, stream.as_raw()));
    }

    #[inline]
    pub fn copy_in<T: Copy>(&mut self, slice: &[T]) {
        let len = size_of_val(slice);
        let src = slice.as_ptr().cast();
        assert_eq!(len, self.len);
        driver!(cuMemcpyHtoD_v2(self.ptr, src, len));
    }
}

pub struct DevMem<'ctx> {
    slice: UnsafeCell<DevSlice>,
    ownership: ResourceOwnership<'ctx>,
}

impl<'ctx> Stream<'ctx> {
    pub fn malloc<T: Copy>(&self, len: usize) -> DevMem<'ctx> {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = 0;
        driver!(cuMemAllocAsync(&mut ptr, len, self.as_raw()));
        DevMem {
            slice: UnsafeCell::new(DevSlice { ptr, len }),
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
            slice: UnsafeCell::new(DevSlice { ptr, len }),
            ownership: owned(self.ctx()),
        }
    }
}

impl Drop for DevMem<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.ownership.is_owned() {
            driver!(cuMemFree_v2(self.slice.get_mut().ptr));
        }
    }
}

impl Deref for DevMem<'_> {
    type Target = DevSlice;
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { &*self.slice.get() }
    }
}

impl DerefMut for DevMem<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.slice.get_mut()
    }
}

impl DevMem<'_> {
    #[inline]
    pub unsafe fn get_mut(&self) -> &mut DevSlice {
        unsafe { &mut *self.slice.get() }
    }
}

#[derive(PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct DevMemSpore(DevSlice);

spore_convention!(DevMemSpore);

impl ContextSpore for DevMemSpore {
    type Resource<'ctx> = DevMem<'ctx>;

    #[inline]
    unsafe fn sprout<'ctx>(&self, ctx: &'ctx ContextGuard) -> Self::Resource<'ctx> {
        DevMem {
            slice: UnsafeCell::new(DevSlice {
                ptr: self.0.ptr,
                len: self.0.len,
            }),
            ownership: not_owned(ctx),
        }
    }

    #[inline]
    unsafe fn kill(&mut self, ctx: &ContextGuard) {
        drop(DevMem {
            slice: UnsafeCell::new(DevSlice {
                ptr: take(&mut self.0.ptr),
                len: self.0.len,
            }),
            ownership: owned(ctx),
        });
    }

    #[inline]
    fn is_alive(&self) -> bool {
        self.0.ptr != cuda::CUdeviceptr::default()
    }
}

impl<'ctx> ContextResource<'ctx> for DevMem<'ctx> {
    type Spore = DevMemSpore;

    #[inline]
    fn sporulate(self) -> Self::Spore {
        let ans = DevMemSpore(DevSlice {
            ptr: self.ptr,
            len: self.len,
        });
        forget(self);
        ans
    }
}
