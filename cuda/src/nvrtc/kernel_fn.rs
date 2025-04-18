use crate::{
    DevByte, Dim3, MemSize, Module, Stream, Version,
    bindings::{
        CUfunction,
        CUfunction_attribute_enum::{self, *},
    },
};
use context_spore::AsRaw;
use std::{
    ffi::{CStr, c_int, c_void},
    fmt,
    marker::PhantomData,
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

impl KernelFn<'_> {
    pub fn launch(
        &self,
        grid_dims: impl Into<Dim3>,
        block_dims: impl Into<Dim3>,
        params: *const *const c_void,
        shared_mem: usize,
        stream: Option<&Stream>,
    ) {
        let grid_dims = grid_dims.into();
        let block_dims = block_dims.into();
        driver!(cuLaunchKernel(
            self.0,
            grid_dims.x,
            grid_dims.y,
            grid_dims.z,
            block_dims.x,
            block_dims.y,
            block_dims.z,
            shared_mem as _,
            stream.map_or(null_mut(), |x| x.as_raw()),
            params as _,
            null_mut(),
        ))
    }

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
    fn get_attribute(&self, attr: CUfunction_attribute_enum) -> c_int {
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

pub trait AsParam {
    #[inline(always)]
    fn as_param(&self) -> *const c_void {
        (&raw const *self).cast()
    }
}

impl AsParam for *const DevByte {}
impl AsParam for *mut DevByte {}
impl AsParam for bool {}
impl AsParam for i8 {}
impl AsParam for u8 {}
impl AsParam for i16 {}
impl AsParam for u16 {}
impl AsParam for i32 {}
impl AsParam for u32 {}
impl AsParam for i64 {}
impl AsParam for u64 {}
impl AsParam for f32 {}
impl AsParam for f64 {}
impl AsParam for isize {}
impl AsParam for usize {}
impl AsParam for half::f16 {}
impl AsParam for half::bf16 {}

#[macro_export]
macro_rules! params {
    [$($p:expr),*] => {{
        use $crate::AsParam;
        [$($p.as_param()),*]
    }};
}

#[test]
fn test_macro() {
    let params = params![1i32, 2i32, 3i32, 4i32, 5i32];
    assert!(params.windows(2).all(|s| !s[0].is_null()
        && !s[1].is_null()
        && s[1] == unsafe { s[0].add(size_of::<i32>()) }));

    let params = params
        .into_iter()
        .map(|ptr| unsafe { *(ptr as *const i32) })
        .collect::<Vec<_>>();
    assert_eq!(params, [1, 2, 3, 4, 5])
}
