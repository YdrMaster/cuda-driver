use crate::{bindings as cuda, AsRaw, Dim3, Stream};
use std::{ffi::c_void, ptr::null_mut};

#[repr(transparent)]
pub struct KernelFn(pub(crate) cuda::CUfunction);

impl KernelFn {
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
