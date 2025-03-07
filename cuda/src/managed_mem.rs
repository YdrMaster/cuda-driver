use crate::{Blob, CurrentCtx, bindings::CUdeviceptr};
use context_spore::{AsRaw, impl_spore};
use std::{
    alloc::Layout,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

#[repr(transparent)]
pub struct ManByte(#[allow(unused)] u8);

#[inline]
pub fn memcpy_d2h_man<T: Copy>(dst: &mut [T], src: &[ManByte]) {
    let len = size_of_val(dst);
    let dst = dst.as_mut_ptr().cast();
    assert_eq!(len, size_of_val(src));
    driver!(cuMemcpyDtoH_v2(dst, src.as_ptr() as _, len))
}

#[inline]
pub fn memcpy_h2d_man<T: Copy>(dst: &mut [ManByte], src: &[T]) {
    let count = src.len();
    let src = src.as_ptr(); 
    let dst = dst.as_ptr() as _;
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, count);
    }
}

#[inline]
pub fn _memcpy_d2d_man(dst: &mut [ManByte], src: &[ManByte]) {
    let len = size_of_val(src);
    assert_eq!(len, size_of_val(dst));
    driver!(cuMemcpyDtoD_v2(dst.as_ptr() as _, src.as_ptr() as _, len))
}

impl_spore!(ManMem and ManMemSpore by (CurrentCtx, Blob<CUdeviceptr>));

impl CurrentCtx {
    pub fn malloc_managed<T: Copy>(&self, len: usize) -> ManMem<'_> {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = 0;
        let flags = 0x1; // CUmemAttachflags: GLOBAL = 0x1, HOST = 0x2, SINGLE = 0x4;
        if len != 0 {
            driver!(cuMemAllocManaged(&mut ptr, len, flags));
        }
        println!("managed ptr: {}", ptr);
        
        ManMem(unsafe { self.wrap_raw(Blob { ptr, len }) }, PhantomData)
    }

    pub fn from_host_man<T: Copy>(&self, slice: &[T]) -> ManMem<'_> {
        let mut dev= self.malloc_managed::<T>(slice.len());
        memcpy_h2d_man(&mut dev, slice);
        dev
    }
}

impl Drop for ManMem<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.0.rss.ptr != 0 {
            driver!(cuMemFree_v2(self.0.rss.ptr))
        }
    }
}

impl Deref for ManMem<'_> {
    type Target = [ManByte];
    #[inline]
    fn deref(&self) -> &Self::Target {
        if self.0.rss.len == 0 {
            &[]
        } else {
            unsafe { from_raw_parts(self.0.rss.ptr as _, self.0.rss.len) }
        }
    }
}

impl DerefMut for ManMem<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.0.rss.len == 0 {
            &mut []
        } else {
            unsafe { from_raw_parts_mut(self.0.rss.ptr as _, self.0.rss.len) }
        }
    }
}

impl AsRaw for ManMemSpore {
    type Raw = CUdeviceptr;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss.ptr
    }
}

impl ManMemSpore {
    #[inline]
    pub const fn len(&self) -> usize {
        self.0.rss.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.0.rss.len == 0
    }
}

#[test]
fn test_managed() {
    use rand::Rng;

    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    let dev = crate::Device::new(0);
    dev.context().apply(|ctx| {
        let mut pagable = vec![0.0f32; 256 << 10];
        rand::rng().fill(&mut *pagable);
        let pagable = unsafe {
            from_raw_parts(
                pagable.as_ptr().cast::<u8>() as *const u8,
                size_of_val(&*pagable),
            )
        };

        let pagable2 = vec![0.0f32; 256 << 10];
        let pagable2 = unsafe {
            from_raw_parts_mut(
                pagable2.as_ptr().cast::<u8>() as *mut u8,
                size_of_val(&*pagable2),
            )
        };

        let size = pagable.len();
        let mut man= ctx.malloc_managed::<u8>(size);
        memcpy_h2d_man(&mut man, pagable);
        memcpy_d2h_man(pagable2, &man);

        assert_eq!(pagable, pagable2);
    });
}