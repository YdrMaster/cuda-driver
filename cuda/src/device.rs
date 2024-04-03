use crate::{bindings as cuda, AsRaw, Dim3};
use std::ptr::null_mut;

#[repr(transparent)]
pub struct Device(cuda::CUdevice);

impl AsRaw for Device {
    type Raw = cuda::CUdevice;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Device {
    #[inline]
    pub fn new(index: i32) -> Self {
        let mut device = 0;
        driver!(cuDeviceGet(&mut device, index));
        Self(device)
    }

    #[inline]
    pub fn fetch() -> Option<Self> {
        if Self::count() > 0 {
            Some(Self::new(0))
        } else {
            None
        }
    }

    #[inline]
    pub fn count() -> usize {
        let mut count = 0;
        driver!(cuDeviceGetCount(&mut count));
        count as _
    }

    #[inline]
    pub fn compute_capability(&self) -> (i32, i32) {
        let mut major = 0;
        let mut minor = 0;
        driver!(cuDeviceComputeCapability(&mut major, &mut minor, self.0));
        (major, minor)
    }

    #[inline]
    pub fn total_memory(&self) -> usize {
        let mut bytes = 0;
        driver!(cuDeviceTotalMem_v2(&mut bytes, self.0));
        bytes as _
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.get_attribute(cuda::CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT)
            as _
    }

    pub fn max_block_dims(&self) -> (usize, Dim3) {
        use cuda::CUdevice_attribute::*;
        (
            self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) as _,
            Dim3 {
                x: self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X) as _,
                y: self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y) as _,
                z: self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z) as _,
            },
        )
    }

    pub fn set_mempool_threshold(&self, threshold: u64) {
        let mut mempool = null_mut();
        driver!(cuDeviceGetDefaultMemPool(&mut mempool, self.0));
        driver!(cuMemPoolSetAttribute(
            mempool,
            CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
            (&threshold) as *const _ as _,
        ));
    }

    #[inline]
    fn get_attribute(&self, attr: cuda::CUdevice_attribute) -> i32 {
        let mut value = 0;
        driver!(cuDeviceGetAttribute(&mut value, attr, self.0));
        value
    }
}

#[test]
fn test() {
    crate::init();
    for i in 0..Device::count() {
        let dev = Device::new(i as _);
        let (major, minor) = dev.compute_capability();
        let (max_threads, max_block_dims) = dev.max_block_dims();
        println!(
            "gpu{i}: ver{major}.{minor} mem={}, max_threads_per_block={}, max_block_dims={:?}",
            dev.total_memory(),
            max_threads,
            max_block_dims,
        );
    }
}
