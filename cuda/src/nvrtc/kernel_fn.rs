use crate::{bindings as cuda, AsRaw, DevByte, Dim3, Module, Stream};
use std::{
    ffi::{c_void, CStr},
    ptr::null_mut,
};

pub struct KernelFn<'m>(cuda::CUfunction, #[allow(unused)] &'m Module<'m>);

impl<'m> Module<'m> {
    pub fn get_kernel(&'m self, name: impl AsRef<CStr>) -> KernelFn<'m> {
        let name = name.as_ref();
        let mut kernel = null_mut();
        driver!(cuModuleGetFunction(
            &mut kernel,
            self.as_raw(),
            name.as_ptr().cast(),
        ));
        KernelFn(kernel, self)
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
        ));
    }
}

pub trait AsParam {
    #[inline(always)]
    fn as_param(&self) -> *const c_void {
        self as *const _ as _
    }
}

macro_rules! impl_as_param_for {
    ($ty:ty) => {
        impl AsParam for $ty {}
    };
}

impl_as_param_for!(*const DevByte);
impl_as_param_for!(*mut DevByte);
impl_as_param_for!(bool);
impl_as_param_for!(i8);
impl_as_param_for!(u8);
impl_as_param_for!(i16);
impl_as_param_for!(u16);
impl_as_param_for!(i32);
impl_as_param_for!(u32);
impl_as_param_for!(i64);
impl_as_param_for!(u64);
impl_as_param_for!(f32);
impl_as_param_for!(f64);
impl_as_param_for!(usize);

#[cfg(feature = "half")]
impl_as_param_for!(half::f16);

#[cfg(feature = "half")]
impl_as_param_for!(half::bf16);

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
        && s[1] == unsafe { s[0].add(std::mem::size_of::<i32>()) }));

    let params = params
        .into_iter()
        .map(|ptr| unsafe { *(ptr as *const i32) })
        .collect::<Vec<_>>();
    assert_eq!(params, [1, 2, 3, 4, 5]);
}
