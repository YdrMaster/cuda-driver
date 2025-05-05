use crate::{
    MemSize, Module, Version,
    bindings::{
        CUfunction,
        CUfunction_attribute::{self, *},
    },
};
use context_spore::AsRaw;
use std::{
    ffi::{CStr, c_int, c_void},
    fmt,
    marker::PhantomData,
    ops::Deref,
    ptr::null_mut,
};

#[derive(Clone, Copy, Debug)]
#[repr(transparent)]
pub struct KernelFn<'m>(CUfunction, PhantomData<&'m ()>);

impl Module<'_> {
    pub fn get_kernel(&self, name: impl AsRef<CStr>) -> KernelFn {
        let name = name.as_ref();
        let mut kernel = null_mut();
        driver!(cuModuleGetFunction(
            &mut kernel,
            self.as_raw(),
            name.as_ptr().cast(),
        ));
        KernelFn(kernel, PhantomData)
    }
}

impl AsRaw for KernelFn<'_> {
    type Raw = CUfunction;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl KernelFn<'_> {
    #[inline]
    pub fn max_threads_per_block(&self) -> usize {
        self.get_attribute(CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK) as _
    }

    #[inline]
    pub fn static_smem(&self) -> MemSize {
        self.get_attribute(CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
            .into()
    }

    #[inline]
    pub fn max_dyn_smem(&self) -> MemSize {
        self.get_attribute(CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES)
            .into()
    }

    #[inline]
    pub fn local_mem(&self) -> MemSize {
        self.get_attribute(CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES)
            .into()
    }

    #[inline]
    pub fn num_regs(&self) -> MemSize {
        self.get_attribute(CU_FUNC_ATTRIBUTE_NUM_REGS).into()
    }

    #[inline]
    pub fn ptx_version(&self) -> Version {
        let version = self.get_attribute(CU_FUNC_ATTRIBUTE_PTX_VERSION);
        Version {
            major: version / 10,
            minor: version % 10,
        }
    }

    #[inline]
    pub fn binary_version(&self) -> Version {
        let version = self.get_attribute(CU_FUNC_ATTRIBUTE_PTX_VERSION);
        Version {
            major: version / 10,
            minor: version % 10,
        }
    }

    #[inline]
    pub fn info(&self) -> InfoFmt {
        InfoFmt(self)
    }

    #[inline]
    fn get_attribute(&self, attr: CUfunction_attribute) -> c_int {
        let mut value = 0;
        driver!(cuFuncGetAttribute(&mut value, attr, self.0));
        value
    }
}

pub struct InfoFmt<'a>(&'a KernelFn<'a>);

impl fmt::Display for InfoFmt<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(
            f,
            "  ptx = {}
  bin = {}
  max threads/block = {}
  local mem = {}
  static smem = {}
  max dyn smem = {}
  regs = {}",
            self.0.ptx_version(),
            self.0.binary_version(),
            self.0.max_threads_per_block(),
            self.0.local_mem(),
            self.0.static_smem(),
            self.0.max_dyn_smem(),
            self.0.num_regs(),
        )
    }
}

#[macro_export]
macro_rules! params {
    [$( $p:expr ),*] => {{
        let mut params = $crate::KernelParams::new();
        $( params.push($p); )*
        params
    }};
}

pub struct KernelParams {
    size: usize,
    data: Vec<u64>,
    each: Vec<usize>,
}

impl Default for KernelParams {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelParams {
    pub fn new() -> Self {
        Self {
            size: 0,
            data: Vec::with_capacity(2),
            each: Vec::with_capacity(2),
        }
    }

    pub fn push<T: Copy + 'static>(&mut self, param: T) {
        // 计算参数对齐
        let mask = align_of::<T>() - 1;
        assert!(mask < align_of::<u64>());
        // 计算参数偏移
        let cursor = (self.size + mask) & (!mask);
        self.each.push(cursor);
        // 计算长度，扩张缓冲区
        const UNIT: usize = size_of::<u64>();
        self.size = cursor + size_of::<T>();
        self.data.resize(self.size.div_ceil(UNIT) * UNIT, 0);
        // 拷贝参数数据
        unsafe {
            std::ptr::copy_nonoverlapping(
                (&raw const param).cast::<u8>(),
                self.data.as_mut_ptr().cast::<u8>().add(cursor),
                size_of::<T>(),
            )
        }
    }

    pub fn to_ptrs(&self) -> KernelParamPtrs {
        KernelParamPtrs(
            self.each
                .iter()
                .map(|&offset| unsafe { self.data.as_ptr().byte_add(offset) }.cast())
                .collect(),
            PhantomData,
        )
    }
}

#[derive(Clone, Default)]
#[repr(transparent)]
pub struct KernelParamPtrs<'a>(Box<[*const c_void]>, PhantomData<&'a ()>);

impl KernelParamPtrs<'_> {
    pub fn empty() -> Self {
        Default::default()
    }
}

impl Deref for KernelParamPtrs<'_> {
    type Target = [*const c_void];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[test]
fn test_macro() {
    let params = params![1i32, 2i32, 3i32, 4i32, 5i32];
    assert!(params.to_ptrs().0.windows(2).all(|s| !s[0].is_null()
        && !s[1].is_null()
        && s[1] == unsafe { s[0].add(size_of::<i32>()) }));

    let params = params
        .to_ptrs()
        .0
        .iter()
        .map(|&ptr| unsafe { *(ptr as *const i32) })
        .collect::<Vec<_>>();
    assert_eq!(params, [1, 2, 3, 4, 5])
}
