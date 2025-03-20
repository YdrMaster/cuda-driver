use crate::{Blob, DevByte, bindings::CUdeviceptr, memcpy_d2h};
use std::{
    alloc::Layout,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub struct ManBlob(Blob<CUdeviceptr>);

#[inline]
pub fn memcpy_h2d_man<T: Copy>(dst: &mut [u8], src: &[T]) {
    let count = src.len();
    let src = src.as_ptr();
    let dst = dst.as_ptr() as _;
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, count);
    }
}

impl ManBlob {
    pub fn malloc_managed<T: Copy>(len: usize) -> ManBlob {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = 0;
        // CUmemAttachflags, should be 0x1 or 0x2; GLOBAL = 0x1, HOST = 0x2; Global: Memory can be accessed by any stream on any device; Host: Memory cannot be accessed by any stream on any device
        let flags = 0x1;
        if len != 0 {
            driver!(cuMemAllocManaged(&mut ptr, len, flags));
        }
        println!("managed ptr: {}", ptr);

        ManBlob(Blob { ptr, len })
    }

    pub fn _from_host_man<T: Copy>(slice: &[T]) -> ManBlob {
        let man = ManBlob::malloc_managed::<T>(slice.len());
        memcpy_h2d_man(man.as_host_mut(), slice);
        man
    }
}

impl Drop for ManBlob {
    #[inline]
    fn drop(&mut self) {
        if self.0.ptr != 0 {
            driver!(cuMemFree_v2(self.0.ptr))
        }
    }
}

impl ManBlob {
    #[inline]
    pub fn _as_host(&self) -> &[u8] {
        if self.0.len == 0 {
            &[]
        } else {
            unsafe { from_raw_parts(self.0.ptr as _, self.0.len) }
        }
    }
    #[inline]
    pub fn as_host_mut(&self) -> &mut [u8] {
        if self.0.len == 0 {
            &mut []
        } else {
            unsafe { from_raw_parts_mut(self.0.ptr as _, self.0.len) }
        }
    }

    #[inline]
    pub fn as_dev(&self) -> &[DevByte] {
        if self.0.len == 0 {
            &[]
        } else {
            unsafe { from_raw_parts(self.0.ptr as _, self.0.len) }
        }
    }
    #[inline]
    pub fn as_dev_mut(&self) -> &mut [DevByte] {
        if self.0.len == 0 {
            &mut []
        } else {
            unsafe { from_raw_parts_mut(self.0.ptr as _, self.0.len) }
        }
    }
}

#[test]
fn test_managed() {
    use rand::Rng;

    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    let dev = crate::Device::new(0);
    dev.context().apply(|_ctx| {
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
        let man = ManBlob::malloc_managed::<u8>(size);
        memcpy_h2d_man(man.as_host_mut(), pagable);
        memcpy_d2h(pagable2, man.as_dev());

        assert_eq!(pagable, pagable2);
    });
}

#[test]
fn test_behavior_multi_stream_async_access() {
    use rand::Rng;

    if let Err(crate::NoDevice) = crate::init() {
        return;
    }

    let dev = crate::Device::new(0);
    dev.context().apply(|ctx| {
        let mut pagable = vec![0.0f32; 256 << 20];
        rand::rng().fill(&mut *pagable);
        let pagable = unsafe {
            from_raw_parts(
                pagable.as_ptr().cast::<u8>() as *const u8,
                size_of_val(&*pagable),
            )
        };

        let mut pagable2 = vec![0.0f32; 256 << 20];
        let pagable2 = unsafe {
            from_raw_parts_mut(
                pagable2.as_mut_ptr().cast::<u8>() as *mut u8,
                size_of_val(&*pagable2),
            )
        };

        let mut pagable3 = vec![0.0f32; 256 << 20];
        let pagable3 = unsafe {
            from_raw_parts_mut(
                pagable3.as_mut_ptr().cast::<u8>() as *mut u8,
                size_of_val(&*pagable3),
            )
        };

        let size = pagable.len();
        let man = ManBlob::malloc_managed::<u8>(size);

        let stream0 = ctx.stream();
        let stream1 = ctx.stream();
        let stream2 = ctx.stream();

        stream0.memcpy_h2d(man.as_dev_mut(), pagable);
        let event = stream0.record();

        stream2.memcpy_d2h(pagable3, man.as_dev());

        stream1.wait_for(&event);
        stream1.memcpy_d2h(pagable2, man.as_dev());

        assert_eq!(pagable, pagable2);
        assert_eq!(pagable, pagable3); // 在host上检验时会等待device同步, 检测不出来stream2未使用event进行同步操作
    });
}

#[test]
fn test_behavior_multi_context_access() {
    todo!()
}
