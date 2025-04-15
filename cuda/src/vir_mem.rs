use crate::{Blob, CurrentCtx, bindings::CUdeviceptr};
use context_spore::{AsRaw, impl_spore};
use std::{
    alloc::Layout,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

#[repr(transparent)]
pub struct VirByte(#[allow(unused)] u8);

#[allow(dead_code)]
#[inline]
pub fn memcpy_d2h_vir<T: Copy>(dst: &mut [T], src: &[VirByte]) {
    let len = size_of_val(dst);
    let dst = dst.as_mut_ptr().cast();
    assert_eq!(len, size_of_val(src));
    driver!(cuMemcpyDtoH_v2(dst, src.as_ptr() as _, len))
}

#[inline]
pub fn memcpy_h2d_vir<T: Copy>(dst: &mut [VirByte], src: &[T]) {
    let len = size_of_val(src);
    let src = src.as_ptr().cast();
    assert_eq!(len, size_of_val(dst));
    driver!(cuMemcpyHtoD_v2(dst.as_ptr() as _, src, len))
}

#[inline]
pub fn _memcpy_d2d_vir(dst: &mut [VirByte], src: &[VirByte]) {
    let len = size_of_val(src);
    assert_eq!(len, size_of_val(dst));
    driver!(cuMemcpyDtoD_v2(dst.as_ptr() as _, src.as_ptr() as _, len))
}

impl_spore!(VirMem and VirMemSpore by (CurrentCtx, Blob<CUdeviceptr>));

pub struct MapArgs {
    pub ptr: u64,
    pub len: usize,
    pub allochandle: u64,
}

impl CurrentCtx {
    pub fn mmap<T: Copy>(&self, addr: u64, len: usize) -> (VirMem<'_>, MapArgs) {
        let len = Layout::array::<T>(len).unwrap().size();

        // 查询压缩类型
        let mut compression_supported = 0;
        driver!(cuDeviceGetAttribute(&mut compression_supported, crate::bindings::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED, 0));
        // 查询设备是否支持 GPU 直接 RDMA
        let mut gpu_direct_rdma_capable = 0;
        driver!(cuDeviceGetAttribute(
            &mut gpu_direct_rdma_capable,
            crate::bindings::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED,
            0
        ));

        //获取粒度
        let mut granularity = 0;
        // CUmemAllocationProp_v1: https://docs.nvidia.com/cuda/cuda-driver-api/structCUmemAllocationProp__v1.html#structCUmemAllocationProp__v1
        let prop = crate::bindings::CUmemAllocationProp_st {
            type_: crate::bindings::CUmemAllocationType_enum::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes:
                crate::bindings::CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            location: crate::bindings::CUmemLocation_st {
                type_: crate::bindings::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                id: 0,
            },
            win32HandleMetaData: std::ptr::null_mut(),
            allocFlags: crate::bindings::CUmemAllocationProp_st__bindgen_ty_1 {
                compressionType: compression_supported as u8,
                gpuDirectRDMACapable: gpu_direct_rdma_capable as u8,
                usage: 0, // 位掩码，表示该内存分配的预期用途; 若 usage = CU_MEM_CREATE_USAGE_TILE_POOL = 1 ，则内存分配仅用作稀疏 CUDA 数组和稀疏 CUDA mipmapped 数组的后备图块池。
                reserved: [0; 4], // 保留供未来使用, 必须为零
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
        assert!(
            len % granularity == 0,
            "len should be a multiple of granularity"
        );
        let padded_size = len;
        let flags = 0; // flags for future use, must be zero now.
        driver!(cuMemCreate(&mut allochandle, padded_size, &prop, flags));

        // 映射到虚拟地址
        let mut ptr = 0;
        let alignment = 0; // default
        let flags = 0; // Currently unused, must be zero
        driver!(cuMemAddressReserve(
            &mut ptr,
            padded_size,
            alignment,
            addr, // 从addr地址段增长虚拟地址
            flags
        ));
        let offset = 0; // currently must be zero.
        let flags = 0; // flags for future use, must be zero now
        driver!(cuMemMap(ptr, padded_size, offset, allochandle, flags));
        let access_desc = crate::bindings::CUmemAccessDesc_st {
            location: crate::bindings::CUmemLocation_st {
                type_: crate::bindings::CUmemLocationType_enum::CU_MEM_LOCATION_TYPE_DEVICE,
                id: 0,
            },
            flags: crate::bindings::CUmemAccess_flags_enum::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        let count = 1; // Number of CUmemAccessDesc in desc
        driver!(cuMemSetAccess(ptr, padded_size, &access_desc, count));
        println!("ptr: {}, len: {}", ptr, len);

        if addr == 0 {
            // 返回[ptr, ptr+len]
            (
                VirMem(unsafe { self.wrap_raw(Blob { ptr, len }) }, PhantomData),
                MapArgs {
                    ptr,
                    len,
                    allochandle,
                },
            )
        } else {
            // 返回[addr, addr + addr.len +len]
            (
                VirMem(
                    unsafe {
                        self.wrap_raw(Blob {
                            ptr: addr,
                            len: (ptr - addr) as usize + padded_size,
                        })
                    },
                    PhantomData,
                ),
                MapArgs {
                    ptr,
                    len,
                    allochandle,
                },
            )
        }
    }

    pub fn munmap(&self, vir: VirMem<'_>, args: MapArgs) {
        std::mem::forget(vir);
        driver!(cuMemUnmap(args.ptr, args.len));
        driver!(cuMemAddressFree(args.ptr, args.len));
        driver!(cuMemRelease(args.allochandle));
    }

    pub fn from_host_vir<T: Copy>(&self, slice: &[T]) -> (VirMem<'_>, MapArgs) {
        let (mut vir, args) = self.mmap::<T>(0, slice.len());
        memcpy_h2d_vir(&mut vir, slice);
        (vir, args)
    }
}

impl Drop for VirMem<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.0.rss.ptr != 0 {
            driver!(cuMemFree_v2(self.0.rss.ptr))
        }
    }
}

impl Deref for VirMem<'_> {
    type Target = [VirByte];
    #[inline]
    fn deref(&self) -> &Self::Target {
        if self.0.rss.len == 0 {
            &[]
        } else {
            unsafe { from_raw_parts(self.0.rss.ptr as _, self.0.rss.len) }
        }
    }
}

impl DerefMut for VirMem<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        if self.0.rss.len == 0 {
            &mut []
        } else {
            unsafe { from_raw_parts_mut(self.0.rss.ptr as _, self.0.rss.len) }
        }
    }
}

impl AsRaw for VirMemSpore {
    type Raw = CUdeviceptr;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss.ptr
    }
}

impl VirMemSpore {
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
fn test_vir_mem() {
    use rand::Rng;

    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    let dev = crate::Device::new(0);
    dev.context().apply(|ctx| {
        let mut pagable = vec![0.0f32; 2 << 20];
        rand::rng().fill(&mut *pagable);
        let pagable = unsafe {
            from_raw_parts(
                pagable.as_ptr().cast::<u8>() as *const u8,
                size_of_val(&*pagable),
            )
        };

        let pagable2 = vec![0.0f32; 2 << 20];
        let pagable2 = unsafe {
            from_raw_parts_mut(
                pagable2.as_ptr().cast::<u8>() as *mut u8,
                size_of_val(&*pagable2),
            )
        };

        let mut pagable3 = vec![0.0f32; 6 << 20];
        rand::rng().fill(&mut *pagable3);
        let pagable3 = unsafe {
            from_raw_parts(
                pagable3.as_ptr().cast::<u8>() as *const u8,
                size_of_val(&*pagable3),
            )
        };

        let pagable4 = vec![0.0f32; 6 << 20];
        let pagable4 = unsafe {
            from_raw_parts_mut(
                pagable4.as_ptr().cast::<u8>() as *mut u8,
                size_of_val(&*pagable4),
            )
        };

        let len = pagable.len();
        // mmap
        let (mut vir1, args1) = ctx.mmap::<u8>(0, len);
        memcpy_h2d_vir(&mut vir1, &pagable);
        memcpy_d2h_vir(pagable2, &vir1);
        assert_eq!(pagable, pagable2);

        // mmap_append
        let (mut vir2, args2) = ctx.mmap::<u8>(args1.ptr as u64, len * 2);
        memcpy_h2d_vir(&mut vir2, &pagable3);
        memcpy_d2h_vir(pagable4, &vir2);
        assert_eq!(pagable3, pagable4);

        ctx.munmap(vir1, args1);
        ctx.munmap(vir2, args2);
    });
}
