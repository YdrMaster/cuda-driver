use crate::{bindings as cuda, AsRaw, Stream};
use std::{
    ffi::{c_uint, c_void},
    ptr::null_mut,
};

#[repr(transparent)]
pub struct KernelFn(pub(crate) cuda::CUfunction);

impl KernelFn {
    pub fn launch(
        &self,
        grid_dims: impl Dim3,
        block_dims: impl Dim3,
        params: *const *const c_void,
        shared_mem: usize,
        stream: Option<&Stream>,
    ) {
        driver!(cuLaunchKernel(
            self.0,
            grid_dims.x(),
            grid_dims.y(),
            grid_dims.z(),
            block_dims.x(),
            block_dims.y(),
            block_dims.z(),
            shared_mem as _,
            stream.map_or_else(|| null_mut(), |x| x.as_raw()),
            params as _,
            null_mut(),
        ));
    }
}

pub trait Dim3 {
    #[inline]
    fn x(&self) -> c_uint {
        1
    }
    #[inline]
    fn y(&self) -> c_uint {
        1
    }
    #[inline]
    fn z(&self) -> c_uint {
        1
    }
}

impl Dim3 for () {}
impl Dim3 for c_uint {
    #[inline]
    fn x(&self) -> c_uint {
        *self
    }
}
impl Dim3 for (c_uint, c_uint) {
    #[inline]
    fn x(&self) -> c_uint {
        self.0
    }
    #[inline]
    fn y(&self) -> c_uint {
        self.1
    }
}
impl Dim3 for (c_uint, c_uint, c_uint) {
    #[inline]
    fn x(&self) -> c_uint {
        self.0
    }
    #[inline]
    fn y(&self) -> c_uint {
        self.1
    }
    #[inline]
    fn z(&self) -> c_uint {
        self.2
    }
}
