use crate::{Blob, DevByte, bindings::CUdeviceptr, memcpy_d2h};
use std::{
    alloc::Layout,
    slice::{from_raw_parts, from_raw_parts_mut},
};

pub struct ManBlob(Blob<CUdeviceptr>);

#[inline]
pub fn memcpy_h2m<T: Copy>(dst: &mut [u8], src: &[T]) {
    let count = src.len();
    let src = src.as_ptr();
    let dst = dst.as_ptr() as _;
    unsafe {
        std::ptr::copy_nonoverlapping(src, dst, count);
    }
}

#[inline]
pub fn _memcpy_m2h<T: Copy>(dst: &mut [T], src: &[u8]) {
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
        memcpy_h2m(man.as_host_mut(), slice);
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
        memcpy_h2m(man.as_host_mut(), pagable);
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
    use lazy_static::lazy_static;
    use rand::Rng;
    use std::sync::{Arc, Mutex};

    // 定义全局变量
    lazy_static! {
        static ref GLOBAL_MAN: Arc<Mutex<Option<ManBlob>>> = Arc::new(Mutex::new(None));
    }

    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    let dev = crate::Device::new(0);

    let ctx1 = dev.context();
    let ctx2 = dev.context();
    // 第一个上下文：创建并使用 `man`
    ctx1.apply(|_ctx| {
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
        let man = ManBlob::malloc_managed::<u8>(size); // 创建man对象
        memcpy_h2m(man.as_host_mut(), pagable);
        memcpy_d2h(pagable2, man.as_dev());

        assert_eq!(pagable, pagable2);

        // 将man对象存储到全局变量中
        let mut global_man = GLOBAL_MAN.lock().unwrap();
        *global_man = Some(man);
    });

    // 第二个上下文：使用第一个上下文中的 `man` 对象
    ctx2.apply(|_ctx| {
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

        let _size = pagable.len();
        // 从全局变量中获取man对象
        let global_man = GLOBAL_MAN.lock().unwrap();
        if let Some(ref man) = *global_man {
            memcpy_h2m(man.as_host_mut(), pagable);
            memcpy_d2h(pagable2, man.as_dev());

            assert_eq!(pagable, pagable2);
        }
    });
}

#[test]
fn test_behavior_multi_context_access2() {
    use std::ptr::copy_nonoverlapping;
    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    let dev = crate::Device::new(0);
    let mut man_mem = 0;
    let host = (0..256 << 10).map(|x| x as f32).collect::<Vec<_>>();
    let size = size_of_val(host.as_slice());
    let ctx1 = dev.context();
    let ctx2 = dev.context();
    ctx1.apply(|_ctx| {
        driver!(cuMemAllocManaged(&mut man_mem, size, 0x1));
        let dev_slice = unsafe { from_raw_parts_mut(man_mem as *mut DevByte, size) };
        crate::memcpy_h2d(dev_slice, host.as_slice());
    });
    let mut host2 = vec![0.0f32; host.len()];
    unsafe { copy_nonoverlapping(man_mem as *mut u8, host2.as_mut_ptr().cast(), size) }
    assert_eq!(host, host2);
    ctx2.apply(|_ctx| {
        let dev_slice = unsafe { from_raw_parts_mut(man_mem as *mut DevByte, size) };
        memcpy_d2h(host2.as_mut_slice(), &dev_slice);
    });
    assert_eq!(host, host2);
}

#[test]
fn test_multi_device_access() {
    // 允许其他设备访问此托管内存
    // driver!(cuMemAdvise(man.0.ptr, man.0.len, CUmem_advise_enum::CU_MEM_ADVISE_SET_ACCESSED_BY, dev.as_raw()));

    // 预取内存到当前上下文关联的设备
    // driver!(cuMemPrefetchAsync(man.0.ptr, man.0.len, 0, std::ptr::null_mut()));

    todo!();
}
