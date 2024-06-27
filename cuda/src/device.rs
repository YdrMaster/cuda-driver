use crate::{
    bindings::{
        CUdevice,
        CUdevice_attribute::{self, *},
    },
    AsRaw, Dim3, MemSize,
};
use std::{cmp::Ordering, ffi::c_int, fmt, ptr::null_mut};

#[repr(transparent)]
pub struct Device(CUdevice);

impl AsRaw for Device {
    type Raw = CUdevice;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Device {
    #[inline]
    pub fn new(index: c_int) -> Self {
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
    pub fn compute_capability(&self) -> ComputeCapability {
        ComputeCapability {
            major: self.get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR),
            minor: self.get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR),
        }
    }

    #[inline]
    pub fn total_memory(&self) -> MemSize {
        let mut bytes = 0;
        driver!(cuDeviceTotalMem_v2(&mut bytes, self.0));
        bytes.into()
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.get_attribute(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT) as _
    }

    #[inline]
    pub fn warp_size(&self) -> usize {
        self.get_attribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE) as _
    }

    #[inline]
    pub fn sm_count(&self) -> usize {
        self.get_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT) as _
    }

    pub fn max_grid_dims(&self) -> Dim3 {
        Dim3 {
            x: self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X) as _,
            y: self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y) as _,
            z: self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z) as _,
        }
    }

    pub fn block_limit(&self) -> BlockLimit {
        BlockLimit {
            max_threads: self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK) as _,
            max_dims: Dim3 {
                x: self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X) as _,
                y: self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y) as _,
                z: self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z) as _,
            },
            max_smem: self
                .get_attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
                .into(),
            max_smem_optin: self
                .get_attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN)
                .into(),
            reserved_smem: self
                .get_attribute(CU_DEVICE_ATTRIBUTE_RESERVED_SHARED_MEMORY_PER_BLOCK)
                .into(),
            max_registers: self
                .get_attribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK)
                .into(),
        }
    }

    pub fn sm_limit(&self) -> SMLimit {
        SMLimit {
            max_blocks: self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCKS_PER_MULTIPROCESSOR) as _,
            max_threads: self.get_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
                as _,
            max_smem: self
                .get_attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR)
                .into(),
            max_registers: self
                .get_attribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_MULTIPROCESSOR)
                .into(),
        }
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
    fn get_attribute(&self, attr: CUdevice_attribute) -> i32 {
        let mut value = 0;
        driver!(cuDeviceGetAttribute(&mut value, attr, self.0));
        value
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct ComputeCapability {
    pub major: i32,
    pub minor: i32,
}

impl ComputeCapability {
    #[inline]
    pub fn to_arch_string(&self) -> String {
        format!("{}{}", self.major, self.minor)
    }
}

impl PartialOrd for ComputeCapability {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ComputeCapability {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        match self.major.cmp(&self.major) {
            Ordering::Equal => self.minor.cmp(&other.minor),
            other => other,
        }
    }
}

impl fmt::Display for ComputeCapability {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}.{}", self.major, self.minor)
    }
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct BlockLimit {
    pub max_threads: usize,
    pub max_dims: Dim3,
    pub max_smem: MemSize,
    pub max_smem_optin: MemSize,
    pub reserved_smem: MemSize,
    pub max_registers: MemSize,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SMLimit {
    pub max_blocks: usize,
    pub max_threads: usize,
    pub max_smem: MemSize,
    pub max_registers: MemSize,
}

#[test]
fn test() {
    crate::init();
    for i in 0..Device::count() {
        let dev = Device::new(i as _);
        let cc = dev.compute_capability();
        let block_limit = dev.block_limit();
        let sm_limit = dev.sm_limit();
        let grid = dev.max_grid_dims();
        println!(
            "\
gpu{i}
  cc = {cc}
  gmem = {}
  alignment = {}
  warp size = {}
  sm count = {}
  block limit
    threads = {} (x: {}, y: {}, z: {})
    smem = {} (reserved: {}, optin: {})
    registers = {}
  sm limit
    blocks = {}
    threads = {}
    smem = {}
    registers = {}
  grid = (x: {}, y: {}, z: {})
",
            dev.total_memory(),
            dev.alignment(),
            dev.warp_size(),
            dev.sm_count(),
            block_limit.max_threads,
            block_limit.max_dims.x,
            block_limit.max_dims.y,
            block_limit.max_dims.z,
            block_limit.max_smem,
            block_limit.reserved_smem,
            block_limit.max_smem_optin,
            block_limit.max_registers,
            sm_limit.max_blocks,
            sm_limit.max_threads,
            sm_limit.max_smem,
            sm_limit.max_registers,
            grid.x,
            grid.y,
            grid.z,
        );
    }
}
