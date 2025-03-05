use crate::{Blob, CurrentCtx, Stream, bindings::CUdeviceptr};
use context_spore::{AsRaw, impl_spore};
use std::{
    alloc::Layout,
    marker::PhantomData,
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
    pub fn memcpy_h2d<T: Copy>(&self, dst: &mut [DevByte], src: &[T]) {
        let len = size_of_val(src);
        let src = src.as_ptr().cast();
        assert_eq!(len, size_of_val(dst));
        driver!(cuMemcpyHtoDAsync_v2(
            dst.as_ptr() as _,
            src,
            len,
            self.as_raw()
        ))
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
        ))
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

    pub fn malloc_managed<T: Copy>(&self, len: usize) -> (DevMem<'_>, u64) {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = 0;
        let flags = 0x1; // CUmemAttachflags: GLOBAL = 0x1, HOST = 0x2, SINGLE = 0x4;
        if len != 0 {
            driver!(cuMemAllocManaged(&mut ptr, len, flags));
        }
        println!("{}", ptr);
        (
            DevMem(unsafe { self.wrap_raw(Blob { ptr, len }) }, PhantomData),
            ptr,
        )
    }

    pub fn malloc_vir<T: Copy>(&self, len: usize) -> (DevMem<'_>, u64, usize, u64) {
        let len = Layout::array::<T>(len).unwrap().size();
        //获取粒度
        let mut granularity = 0;
        let prop = crate::bindings::CUmemAllocationProp_st {
            type_: crate::bindings::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes:
                crate::bindings::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            location: crate::bindings::CUmemLocation_st {
                type_: crate::bindings::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                id: 0,
            },
            win32HandleMetaData: std::ptr::null_mut(),
            // notice: 没找到对应资料, 设为全0
            allocFlags: crate::bindings::CUmemAllocationProp_st__bindgen_ty_1 {
                compressionType: 0,
                gpuDirectRDMACapable: 0,
                usage: 0,
                reserved: [0; 4],
            },
        };
        let option = crate::bindings::CUmemAllocationGranularity_flags_enum::CU_MEM_ALLOC_GRANULARITY_MINIMUM;
        driver!(cuMemGetAllocationGranularity(
            &mut granularity,
            &prop,
            option
        ));
        println!("granularity: {}", granularity);

        // 分配物理显存
        let mut allochandle = 0;
        let padded_size = ((len + granularity - 1) / granularity) * granularity;
        let flags = 0;
        driver!(cuMemCreate(&mut allochandle, padded_size, &prop, flags));
        println!("allochandle: {}", allochandle);

        // 映射到虚拟地址
        let mut ptr = 0;
        driver!(cuMemAddressReserve(&mut ptr, padded_size, 0, 0, 0));
        driver!(cuMemMap(ptr, padded_size, 0, allochandle, 0));
        let access_desc = crate::bindings::CUmemAccessDesc_st {
            location: crate::bindings::CUmemLocation_st {
                type_: crate::bindings::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                id: 0,
            },
            flags: crate::bindings::CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        driver!(cuMemSetAccess(ptr, padded_size, &access_desc, 1));
        println!("ptr: {}", ptr);

        (
            DevMem(unsafe { self.wrap_raw(Blob { ptr, len }) }, PhantomData),
            ptr,
            padded_size,
            allochandle,
        )
    }

    pub fn from_host<T: Copy>(&self, slice: &[T]) -> DevMem<'_> {
        let mut dev = self.malloc::<T>(slice.len());
        memcpy_h2d(&mut dev, slice);
        dev
    }
}

#[cfg(nvidia)]
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
}

#[cfg(nvidia)]
impl DevMem<'_> {
    #[inline]
    pub fn drop_on(self, stream: &Stream) {
        driver!(cuMemFreeAsync(self.0.rss.ptr, stream.as_raw()));
        std::mem::forget(self)
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
