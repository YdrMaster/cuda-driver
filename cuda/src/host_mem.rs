use crate::{
    context::ResourceOwnership, not_owned, owned, spore_convention, AsRaw, ContextGuard,
    ContextResource, ContextSpore,
};
use std::{
    alloc::Layout,
    mem::{forget, replace},
    ops::{Deref, DerefMut},
    os::raw::c_void,
    ptr::null_mut,
};

pub struct HostMem<'ctx> {
    ptr: *mut c_void,
    len: usize,
    ownership: ResourceOwnership<'ctx>,
}

impl<'ctx> ContextGuard<'ctx> {
    pub fn malloc_host<T: Copy>(&'ctx self, len: usize) -> HostMem<'ctx> {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = null_mut();
        driver!(cuMemHostAlloc(&mut ptr, len, 0));
        HostMem {
            ptr,
            len,
            ownership: owned(self),
        }
    }
}

impl Drop for HostMem<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.ownership.is_owned() {
            driver!(cuMemFreeHost(self.ptr));
        }
    }
}

impl AsRaw for HostMem<'_> {
    type Raw = *mut c_void;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.ptr
    }
}

impl Deref for HostMem<'_> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.cast(), self.len) }
    }
}

impl DerefMut for HostMem<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.cast(), self.len) }
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct HostMemSpore(*mut c_void, usize);

spore_convention!(HostMemSpore);

impl ContextSpore for HostMemSpore {
    type Resource<'ctx> = HostMem<'ctx>;

    #[inline]
    unsafe fn sprout<'ctx>(&self, ctx: &'ctx ContextGuard) -> Self::Resource<'ctx> {
        HostMem {
            ptr: self.0,
            len: self.1,
            ownership: not_owned(ctx),
        }
    }

    #[inline]
    unsafe fn kill(&mut self, ctx: &ContextGuard) {
        drop(HostMem {
            ptr: replace(&mut self.0, null_mut()),
            len: self.1,
            ownership: owned(ctx),
        });
    }

    #[inline]
    fn is_alive(&self) -> bool {
        !self.0.is_null()
    }
}

impl<'ctx> ContextResource<'ctx> for HostMem<'ctx> {
    type Spore = HostMemSpore;

    #[inline]
    fn sporulate(self) -> Self::Spore {
        let ans = HostMemSpore(self.ptr, self.len);
        forget(self);
        ans
    }
}

impl Deref for HostMemSpore {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.0.cast(), self.1) }
    }
}

impl DerefMut for HostMemSpore {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.0.cast(), self.1) }
    }
}

#[test]
fn test_behavior() {
    crate::init();
    let Some(dev) = crate::Device::fetch() else {
        return;
    };
    let mut ptr = null_mut();
    dev.context().apply(|_| {
        driver!(cuMemHostAlloc(&mut ptr, 128, 0));
        driver!(cuMemFreeHost(ptr));
    });
    ptr = null_mut();
    driver!(cuMemFreeHost(ptr));
}
