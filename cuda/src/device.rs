use crate::{
    Dim3, MemSize, Version,
    bindings::{
        HCdevice,
        hcDeviceAttribute_t::{self, *},
    },
};
use context_spore::AsRaw;
use std::{ffi::c_int, fmt};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Device(HCdevice);

impl AsRaw for Device {
    type Raw = HCdevice;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Device {
    #[inline]
    pub fn new(index: c_int) -> Self {
        let mut device = 0;
        driver!(hcDeviceGet(&mut device, index));
        Self(device)
    }

    #[inline]
    pub fn count() -> usize {
        let mut count = 0;
        driver!(hcGetDeviceCount(&mut count));
        count as _
    }

    pub fn name(&self) -> String {
        let mut name = [0u8; 256];
        driver!(hcDeviceGetName(
            name.as_mut_ptr().cast(),
            name.len() as _,
            self.0
        ));
        String::from_utf8(name.iter().take_while(|&&c| c != 0).copied().collect()).unwrap()
    }

    #[inline]
    pub const fn index(&self) -> c_int {
        self.0
    }

    #[inline]
    pub fn compute_capability(&self) -> Version {
        Version {
            major: self.get_attribute(hcDeviceAttributeComputeCapabilityMajor),
            minor: self.get_attribute(hcDeviceAttributeComputeCapabilityMinor),
        }
    }

    #[inline]
    pub fn total_memory(&self) -> MemSize {
        let mut bytes = 0;
        driver!(hcDeviceTotalMem(&mut bytes, self.0));
        bytes.into()
    }

    #[inline]
    pub fn vm_supported(&self) -> bool {
        #[cfg(nvidia)]
        let attr = CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED;
        #[cfg(iluvatar)]
        let attr = CUdevice_attribute::CU_DEVICE_ATTRIBUTE_VIRTUAL_ADDRESS_MANAGEMENT_SUPPORTED;
        #[cfg(metax)]
        let attr = hcDeviceAttribute_t::hcDeviceAttributeVirtualMemoryManagementSupported;
        self.get_attribute(attr) != 0
    }

    #[inline]
    pub fn alignment(&self) -> usize {
        self.get_attribute(hcDeviceAttributeTextureAlignment) as _
    }

    #[inline]
    pub fn warp_size(&self) -> usize {
        self.get_attribute(hcDeviceAttributeWarpSize) as _
    }

    #[inline]
    pub fn sm_count(&self) -> usize {
        self.get_attribute(hcDeviceAttributeMultiProcessorCount) as _
    }

    pub fn max_grid_dims(&self) -> Dim3 {
        Dim3 {
            x: self.get_attribute(hcDeviceAttributeMaxGridDimX) as _,
            y: self.get_attribute(hcDeviceAttributeMaxGridDimY) as _,
            z: self.get_attribute(hcDeviceAttributeMaxGridDimZ) as _,
        }
    }

    pub fn block_limit(&self) -> BlockLimit {
        BlockLimit {
            max_threads: self.get_attribute(hcDeviceAttributeMaxThreadsPerBlock) as _,
            max_dims: Dim3 {
                x: self.get_attribute(hcDeviceAttributeMaxBlockDimX) as _,
                y: self.get_attribute(hcDeviceAttributeMaxBlockDimY) as _,
                z: self.get_attribute(hcDeviceAttributeMaxBlockDimZ) as _,
            },
            max_smem: self
                .get_attribute(hcDeviceAttributeMaxSharedMemoryPerBlock)
                .into(),
            max_smem_optin: self
                .get_attribute(hcDeviceAttributeMaxSharedMemoryPerBlockOptin)
                .into(),
            // #[cfg(nvidia)]
            reserved_smem: self
                .get_attribute(hcDeviceAttributeReservedSharedMemoryPerBlock)
                .into(),
            max_registers: self
                .get_attribute(hcDeviceAttributeMaxRegistersPerBlock)
                .into(),
        }
    }

    pub fn sm_limit(&self) -> SMLimit {
        SMLimit {
            // #[cfg(nvidia)]
            max_blocks: self.get_attribute(hcDevAttrMaxBlocksPerMultiprocessor) as _,
            max_threads: self.get_attribute(hcDeviceAttributeMaxThreadsPerMultiProcessor) as _,
            max_smem: self
                .get_attribute(hcDeviceAttributeMaxSharedMemoryPerMultiprocessor)
                .into(),
            max_registers: self
                .get_attribute(hcDeviceAttributeMaxRegistersPerMultiprocessor)
                .into(),
        }
    }

    #[inline]
    pub fn info(&self) -> InfoFmt {
        InfoFmt(self)
    }

    pub fn set_mempool_threshold(&self, threshold: u64) {
        let mut mempool = std::ptr::null_mut();
        driver!(hcDeviceGetDefaultMemPool(&mut mempool, self.0));
        driver!(hcMemPoolSetAttribute(
            mempool,
            hcMemPoolAttr::hcMemPoolAttrReleaseThreshold,
            (&raw const threshold) as _,
        ));
    }

    #[inline]
    fn get_attribute(&self, attr: hcDeviceAttribute_t) -> c_int {
        let mut value = 0;
        driver!(hcDeviceGetAttribute(&mut value, attr, self.0));
        value
    }
}

pub struct InfoFmt<'a>(&'a Device);

impl fmt::Display for InfoFmt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let block_limit = self.0.block_limit();
        let sm_limit = self.0.sm_limit();
        let grid = self.0.max_grid_dims();

        #[cfg(not(iluvatar))]
        let reserved = block_limit.reserved_smem;
        #[cfg(iluvatar)]
        let reserved = "unknown";

        #[cfg(not(iluvatar))]
        let sm_blocks = sm_limit.max_blocks;
        #[cfg(iluvatar)]
        let sm_blocks = "unknown";

        writeln!(
            f,
            "\
GPU{} ({})
  cc = {}
  vm supported = {}
  gmem = {}
  alignment = {}
  warp size = {}
  sm count = {}
  block limit
    threads = {} (x: {}, y: {}, z: {})
    smem = {} (reserved: {reserved}, optin: {})
    registers = {}
  sm limit
    blocks = {sm_blocks}
    threads = {}
    smem = {}
    registers = {}
  grid = (x: {}, y: {}, z: {})",
            self.0.0,
            self.0.name(),
            self.0.compute_capability(),
            self.0.vm_supported(),
            self.0.total_memory(),
            self.0.alignment(),
            self.0.warp_size(),
            self.0.sm_count(),
            block_limit.max_threads,
            block_limit.max_dims.x,
            block_limit.max_dims.y,
            block_limit.max_dims.z,
            block_limit.max_smem,
            block_limit.max_smem_optin,
            block_limit.max_registers,
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
    #[cfg(not(iluvatar))]
    pub reserved_smem: MemSize,
    pub max_registers: MemSize,
}

#[derive(Clone, PartialEq, Eq, Hash, Debug)]
pub struct SMLimit {
    #[cfg(not(iluvatar))]
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
