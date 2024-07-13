use crate::{impl_spore, AsRaw, Blob, CurrentCtx};
use std::{
    alloc::Layout,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    os::raw::c_void,
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
};

impl_spore!(HostMem and HostMemSpore by Blob<*mut c_void>);

impl CurrentCtx {
    pub fn malloc_host<T: Copy>(&self, len: usize) -> HostMem {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = null_mut();
        driver!(cuMemHostAlloc(&mut ptr, len, 0));
        HostMem(unsafe { self.wrap_raw(Blob { ptr, len }) }, PhantomData)
    }
}

impl Drop for HostMem<'_> {
    #[inline]
    fn drop(&mut self) {
        driver!(cuMemFreeHost(self.0.raw.ptr));
    }
}

impl AsRaw for HostMem<'_> {
    type Raw = *mut c_void;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.raw.ptr
    }
}

impl Deref for HostMem<'_> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { from_raw_parts(self.0.raw.ptr.cast(), self.0.raw.len) }
    }
}

impl DerefMut for HostMem<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { from_raw_parts_mut(self.0.raw.ptr.cast(), self.0.raw.len) }
    }
}

impl Deref for HostMemSpore {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { from_raw_parts(self.0.raw.ptr.cast(), self.0.raw.len) }
    }
}

impl DerefMut for HostMemSpore {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { from_raw_parts_mut(self.0.raw.ptr.cast(), self.0.raw.len) }
    }
}

#[test]
fn test_behavior() {
    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    let mut ptr = null_mut();
    crate::Device::new(0).context().apply(|_| {
        driver!(cuMemHostAlloc(&mut ptr, 128, 0));
        driver!(cuMemFreeHost(ptr));
    });
    ptr = null_mut();
    driver!(cuMemFreeHost(ptr));
}
