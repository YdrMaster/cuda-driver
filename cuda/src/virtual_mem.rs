use crate::{
    DevByte, Device,
    bindings::{
        CUdeviceptr, CUmemAccess_flags, CUmemAccessDesc, CUmemAllocationGranularity_flags,
        CUmemAllocationHandleType, CUmemAllocationProp, CUmemAllocationType,
        CUmemGenericAllocationHandle, CUmemLocation, CUmemLocationType,
    },
};
use context_spore::AsRaw;
use std::{
    mem::ManuallyDrop,
    ops::{Deref, DerefMut},
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::Arc,
};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct MemProp(CUmemAllocationProp);

impl Device {
    pub fn mem_prop(&self) -> MemProp {
        MemProp(CUmemAllocationProp {
            type_: CUmemAllocationType::CU_MEM_ALLOCATION_TYPE_PINNED,
            requestedHandleTypes: CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_NONE,
            location: CUmemLocation {
                type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                id: unsafe { self.as_raw() },
            },
            win32HandleMetaData: null_mut(),
            allocFlags: unsafe { std::mem::zeroed() },
        })
    }
}

impl MemProp {
    #[inline]
    pub fn granularity_minimum(&self) -> usize {
        self.granularity(CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_MINIMUM)
    }

    #[inline]
    pub fn granularity_recommended(&self) -> usize {
        self.granularity(CUmemAllocationGranularity_flags::CU_MEM_ALLOC_GRANULARITY_RECOMMENDED)
    }

    fn granularity(&self, type_: CUmemAllocationGranularity_flags) -> usize {
        let mut size = 0;
        driver!(cuMemGetAllocationGranularity(&mut size, &self.0, type_));
        size
    }
}

#[repr(transparent)]
pub struct VirByte(u8);

pub struct VirMem {
    ptr: CUdeviceptr,
    len: usize,
}

impl VirMem {
    pub fn new(len: usize, min_addr: usize) -> Self {
        let mut ptr = 0;
        driver!(cuMemAddressReserve(&mut ptr, len, 0, min_addr as _, 0));
        Self { ptr, len }
    }
}

impl Drop for VirMem {
    fn drop(&mut self) {
        let &mut Self { ptr, len } = self;
        driver!(cuMemAddressFree(ptr, len))
    }
}

impl Deref for VirMem {
    type Target = [VirByte];
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { from_raw_parts(self.ptr as _, self.len) }
    }
}

impl DerefMut for VirMem {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { from_raw_parts_mut(self.ptr as _, self.len) }
    }
}

pub struct PhyMem {
    location: CUmemLocation,
    handle: CUmemGenericAllocationHandle,
    len: usize,
}

impl MemProp {
    pub fn create(&self, len: usize) -> Arc<PhyMem> {
        let mut handle = 0;
        driver!(cuMemCreate(&mut handle, len, &self.0, 0));
        Arc::new(PhyMem {
            location: self.0.location,
            handle,
            len,
        })
    }
}

impl Drop for PhyMem {
    fn drop(&mut self) {
        let &mut Self { handle, .. } = self;
        driver!(cuMemRelease(handle))
    }
}

impl AsRaw for PhyMem {
    type Raw = CUmemGenericAllocationHandle;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.handle
    }
}

impl PhyMem {
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[repr(transparent)]
pub struct MappedMem(ManuallyDrop<Internal>);

/// 需要一个内部结构来控制何时自动释放。
///
/// [`MappedMem`] 自动释放时，[`Internal`] 的两个成员递归释放。主动解映射时，[`Internal`] 的成员被取出，不释放。
struct Internal {
    vir: VirMem,
    phy: Arc<PhyMem>,
}

impl VirMem {
    pub fn map(self, phy: Arc<PhyMem>) -> MappedMem {
        debug_assert!(
            self.len >= phy.len,
            "cannot map physical memory to a smaller address region"
        );
        driver!(cuMemMap(self.ptr, phy.len, 0, phy.handle, 0));

        let desc = CUmemAccessDesc {
            location: phy.location,
            flags: CUmemAccess_flags::CU_MEM_ACCESS_FLAGS_PROT_READWRITE,
        };
        driver!(cuMemSetAccess(self.ptr, phy.len, &desc, 1));

        MappedMem(ManuallyDrop::new(Internal { vir: self, phy }))
    }

    pub fn map_on(self, dev: &Device) -> MappedMem {
        let len = self.len;
        self.map(dev.mem_prop().create(len))
    }
}

impl Drop for MappedMem {
    fn drop(&mut self) {
        driver!(cuMemUnmap(self.0.vir.ptr, self.0.phy.len));
        unsafe { ManuallyDrop::drop(&mut self.0) }
    }
}

impl MappedMem {
    pub fn unmap(mut self) -> (VirMem, Arc<PhyMem>) {
        driver!(cuMemUnmap(self.0.vir.ptr, self.0.phy.len));
        let Internal { vir, phy } = unsafe { ManuallyDrop::take(&mut self.0) };
        std::mem::forget(self);
        (vir, phy)
    }
}

impl Deref for MappedMem {
    type Target = [DevByte];
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { from_raw_parts(self.0.vir.ptr as _, self.0.phy.len) }
    }
}

impl DerefMut for MappedMem {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { from_raw_parts_mut(self.0.vir.ptr as _, self.0.phy.len) }
    }
}

#[test]
fn test_behavior() {
    use crate::{Device, memcpy_d2h, memcpy_h2d, virtual_mem::VirMem};
    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    let dev = Device::new(0);

    let prop = dev.mem_prop();
    let minimum = prop.granularity_minimum();
    let recommended = prop.granularity_recommended();
    println!("minimun = {minimum}, recommended = {recommended}");

    // 分配一个较大的虚地址区域
    let virmem = VirMem::new(10 * minimum, 0);
    // 分配一个较小的物理页
    let phymem = prop.create(minimum);
    // 建立映射
    let mut mapped = virmem.map(phymem.clone());

    // 通过虚地址操作存储空间
    let host = (0..minimum / size_of::<usize>()).collect::<Box<_>>();
    // 对存储空间的操作仍然需要在上下文中进行
    dev.context().apply(|_| memcpy_h2d(&mut mapped, &host));

    // 分配另一个虚地址区域
    let virmem = VirMem::new(2 * minimum, 0);
    // 将同一个物理页映射到虚地址区域
    let mapped = virmem.map(phymem);
    // 在另一个上下文中读取存储空间
    let mut host_ = vec![0usize; host.len()];
    dev.context().apply(|_| memcpy_d2h(&mut host_, &mapped));

    assert_eq!(&*host, &*host_)
}
