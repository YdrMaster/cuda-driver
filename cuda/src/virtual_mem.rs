use crate::{
    DevByte, Device,
    bindings::{
        HCdeviceptr, hcMemAccessDesc, hcMemAccessFlags, hcMemAllocationGranularity_flags,
        hcMemAllocationHandleType, hcMemAllocationProp, hcMemAllocationType,
        hcMemGenericAllocationHandle, hcMemLocation, hcMemLocationType,
    },
};
use context_spore::AsRaw;
use std::{
    collections::BTreeMap,
    ops::{Deref, DerefMut},
    os::raw::c_void,
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
    sync::Arc,
};

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct MemProp(hcMemAllocationProp);

impl Device {
    pub fn mem_prop(&self) -> MemProp {
        MemProp(hcMemAllocationProp {
            type_: hcMemAllocationType::hcMemAllocationTypePinned,
            #[cfg(not(iluvatar))]
            requestedHandleTypes: hcMemAllocationHandleType::hcMemHandleTypeNone,
            #[cfg(iluvatar)]
            requestedHandleTypes:
                CUmemAllocationHandleType::CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR,
            location: hcMemLocation {
                type_: hcMemLocationType::hcMemLocationTypeDevice,
                id: unsafe { self.as_raw() },
            },
            win32HandleMetaData: null_mut(),
            #[cfg(not(iluvatar))]
            allocFlags: unsafe { std::mem::zeroed() },
            #[cfg(iluvatar)]
            reserved: 0,
        })
    }
}

impl MemProp {
    #[inline]
    pub fn granularity_minimum(&self) -> usize {
        self.granularity(hcMemAllocationGranularity_flags::HC_MEM_ALLOC_GRANULARITY_MINIMUM)
    }

    #[inline]
    pub fn granularity_recommended(&self) -> usize {
        self.granularity(hcMemAllocationGranularity_flags::HC_MEM_ALLOC_GRANULARITY_RECOMMENDED)
    }

    fn granularity(&self, type_: hcMemAllocationGranularity_flags) -> usize {
        let mut size = 0;
        driver!(hcMemGetAllocationGranularity(&mut size, &self.0, type_));
        size
    }
}

#[repr(transparent)]
pub struct VirByte(u8);

pub struct VirMem {
    ptr: HCdeviceptr,
    len: usize,
    /// offset -> phy
    map: BTreeMap<usize, PhyRegion>,
}

impl VirMem {
    pub fn new(len: usize, min_addr: usize) -> Self {
        let mut ptr = null_mut();
        driver!(hcMemAddressReserve(&mut ptr, len, 0, min_addr as _, 0));
        Self {
            ptr: ptr as _,
            len,
            map: [(0, len.into())].into(),
        }
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

impl Drop for VirMem {
    fn drop(&mut self) {
        let Self { ptr, len, map } = self;
        let map = std::mem::take(map);
        for (offset, region) in map {
            if let PhyRegion::Mapped(phy) = region {
                driver!(hcMemUnmap((*ptr + offset as HCdeviceptr) as _, phy.len))
            }
        }
        driver!(hcMemAddressFree(*ptr as *mut c_void, *len))
    }
}

pub struct PhyMem {
    location: hcMemLocation,
    handle: hcMemGenericAllocationHandle,
    len: usize,
}

impl MemProp {
    pub fn create(&self, len: usize) -> Arc<PhyMem> {
        let mut handle = 0;
        driver!(hcMemCreate(&mut handle, len, &self.0, 0));
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
        driver!(hcMemRelease(handle))
    }
}

impl AsRaw for PhyMem {
    type Raw = hcMemGenericAllocationHandle;
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

enum PhyRegion {
    Mapped(Arc<PhyMem>),
    Vacant(usize),
}

impl From<Arc<PhyMem>> for PhyRegion {
    fn from(value: Arc<PhyMem>) -> Self {
        Self::Mapped(value)
    }
}

impl From<usize> for PhyRegion {
    fn from(value: usize) -> Self {
        Self::Vacant(value)
    }
}

impl VirMem {
    pub fn map(&mut self, offset: usize, phy: Arc<PhyMem>) -> &mut [DevByte] {
        // 检查范围
        assert!(offset <= self.len && offset + phy.len <= self.len);
        // 查找所在区间
        let (head, region) = self.map.range(..=offset).next_back().unwrap();
        // 获取空闲段长度
        let len = match *region {
            PhyRegion::Mapped(_) => panic!("mem is mapped"),
            PhyRegion::Vacant(len) => len,
        };
        assert!(phy.len <= len);
        // 映射
        {
            let ptr = self.ptr + offset as HCdeviceptr;
            driver!(hcMemMap(ptr as _, phy.len, 0, phy.handle, 0));
            let desc = hcMemAccessDesc {
                location: phy.location,
                flags: hcMemAccessFlags::hcMemAccessFlagsProtReadWrite,
            };
            driver!(hcMemSetAccess(ptr as _, phy.len, &desc, 1));
        }
        // 移除空闲段
        let head = *head;
        self.map.remove(&head);
        // 插入映射段
        let phy_len = phy.len;
        self.map.insert(offset, phy.into());
        // 插入头尾空闲段
        let head_len = offset - head;
        let tail_len = len - head_len - phy_len;
        if head_len > 0 {
            self.map.insert(head, head_len.into());
        }
        if tail_len > 0 {
            let tail = head + head_len + phy_len;
            self.map.insert(tail, tail_len.into());
        }
        unsafe { std::slice::from_raw_parts_mut((self.ptr + offset as HCdeviceptr) as _, phy_len) }
    }

    pub fn unmap(&mut self, offset: usize) -> Arc<PhyMem> {
        let region = self.map.get_mut(&offset).expect("offset is not a boundary");
        let len = match region {
            PhyRegion::Mapped(phy_mem) => phy_mem.len,
            PhyRegion::Vacant(_) => panic!("offset is not mapped"),
        };
        let PhyRegion::Mapped(phy) = std::mem::replace(region, len.into()) else {
            unreachable!()
        };
        let ptr = self.ptr + offset as HCdeviceptr;
        driver!(hcMemUnmap(ptr as _, phy.len));
        phy
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
    let mut virmem = VirMem::new(10 * minimum, 0);
    // 分配一个较小的物理页
    let phymem = prop.create(minimum);
    // 建立映射
    let mapped = virmem.map(minimum, phymem.clone());

    // 通过虚地址操作存储空间
    let host = (0..minimum / size_of::<usize>()).collect::<Box<_>>();
    // 对存储空间的操作仍然需要在上下文中进行
    dev.context().apply(|_| memcpy_h2d(mapped, &host));

    // 分配另一个虚地址区域
    let mut virmem = VirMem::new(2 * minimum, 0);
    // 将同一个物理页映射到虚地址区域
    let mapped = virmem.map(minimum, phymem);
    // 在另一个上下文中读取存储空间
    let mut host_ = vec![0usize; host.len()];
    dev.context().apply(|_| memcpy_d2h(&mut host_, mapped));

    assert_eq!(&*host, &*host_)
}

#[cfg(nvidia)]
#[allow(unused)]
// #[test] // 这个函数会毁坏 context，这会干扰其他线程上的其他 context，在并发情况下导致异常行为
fn test_unmap() {
    use crate::{
        Ptx,
        bindings::{CUresult, cuStreamSynchronize},
        params,
    };
    use std::ptr::null_mut;

    extern "C" fn host_fn(_e: *mut core::ffi::c_void) {
        for _ in 0..5 {
            println!("waiting @ {:?}", std::thread::current());
            std::thread::sleep(std::time::Duration::from_millis(200))
        }
        println!("Host fn finished!")
    }

    const CODE: &str = r#"
extern "C" __global__ void print(int* p) {
    printf("Read params [%d, %d, %d, %d]\n", p[0], p[1], p[2], p[3]);
}

extern "C" __global__ void add(int* p) {
    for (int i = 0; i < 4; ++i) ++p[i];
}"#;

    if let Err(crate::NoDevice) = crate::init() {
        return;
    }

    let dev = Device::new(0);
    let (ptx, _log) = Ptx::compile(CODE, dev.compute_capability());

    let prop = dev.mem_prop();
    let minium = prop.granularity_minimum();

    // 分配虚地址空间
    let mut vir = VirMem::new(minium, 0);

    dev.context().apply(|ctx| {
        println!("host thread = {:?}", std::thread::current());
        let module = ctx.load(&ptx.unwrap());
        let print = module.get_kernel(c"print");
        let add = module.get_kernel(c"add");

        // 映射虚拟内存
        let mem = vir.map(0, prop.create(minium));

        let stream = ctx.stream();
        // 发射 host 函数，用于延时
        driver!(cuLaunchHostFunc(
            stream.as_raw(),
            Some(host_fn as _),
            null_mut()
        ));

        let attrs = ((), (), 0);
        let params = params![mem.as_ptr()];
        stream
            .memcpy_h2d(&mut mem[..size_of::<[i32; 4]>()], &[0i32, 1, 2, 3])
            .launch(&print, attrs, &params.to_ptrs())
            .synchronize();

        // 进行 unmap，这之后对虚地址的访问将失效
        println!("Kernel launched - now unmap virtual memory ");
        vir.unmap(0);

        // 发射 kernel 或者
        stream
            .launch(&add, attrs, &params.to_ptrs())
            .launch(&print, attrs, &params.to_ptrs());

        let result = unsafe { cuStreamSynchronize(stream.as_raw()) };
        assert!(result != CUresult::CUDA_SUCCESS, "result = {result:?}");

        std::mem::forget((stream, module))
    })
}
