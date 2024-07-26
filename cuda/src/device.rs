use crate::{
    bindings::{
        CUdevice,
        CUdevice_attribute::{self, *},
    },
    Dim3, MemSize, Version,
};
use context_spore::AsRaw;
use std::{ffi::c_int, fmt, ptr::null_mut};

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
    pub fn count() -> usize {
        let mut count = 0;
        driver!(cuDeviceGetCount(&mut count));
        count as _
    }

    pub fn name(&self) -> String {
        let mut name = [0u8; 256];
        driver!(cuDeviceGetName(
            name.as_mut_ptr().cast(),
            name.len() as _,
            self.0
        ));
        String::from_utf8(name.iter().take_while(|&&c| c != 0).copied().collect()).unwrap()
    }

    #[inline]
    pub fn compute_capability(&self) -> Version {
        Version {
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

    #[inline]
    pub fn info(&self) -> InfoFmt {
        InfoFmt(self)
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
    fn get_attribute(&self, attr: CUdevice_attribute) -> c_int {
        let mut value = 0;
        driver!(cuDeviceGetAttribute(&mut value, attr, self.0));
        value
    }
}

pub struct InfoFmt<'a>(&'a Device);

impl fmt::Display for InfoFmt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let block_limit = self.0.block_limit();
        let sm_limit = self.0.sm_limit();
        let grid = self.0.max_grid_dims();
        writeln!(
            f,
            "\
GPU{} ({})
  cc = {}
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
  grid = (x: {}, y: {}, z: {})",
            self.0 .0,
            self.0.name(),
            self.0.compute_capability(),
            self.0.total_memory(),
            self.0.alignment(),
            self.0.warp_size(),
            self.0.sm_count(),
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
        )
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
    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    for i in 0..Device::count() {
        println!("{}", Device::new(i as _).info());
    }
}
