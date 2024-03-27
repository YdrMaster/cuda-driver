use crate::{bindings as cuda, AsRaw, Module, Stream};
use std::ffi::{c_uint, CStr};
use std::{ffi::c_void, ptr::null_mut};

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

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Dim3 {
    pub x: c_uint,
    pub y: c_uint,
    pub z: c_uint,
}

impl From<()> for Dim3 {
    #[inline]
    fn from(_: ()) -> Self {
        Self { x: 1, y: 1, z: 1 }
    }
}

impl From<c_uint> for Dim3 {
    #[inline]
    fn from(x: c_uint) -> Self {
        Self { x, y: 1, z: 1 }
    }
}

impl From<(c_uint, c_uint)> for Dim3 {
    #[inline]
    fn from((y, x): (c_uint, c_uint)) -> Self {
        Self { x, y, z: 1 }
    }
}

impl From<(c_uint, c_uint, c_uint)> for Dim3 {
    #[inline]
    fn from((z, y, x): (c_uint, c_uint, c_uint)) -> Self {
        Self { x, y, z }
    }
}
