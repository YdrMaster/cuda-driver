#![cfg(detected_cuda)]

#[macro_use]
pub mod bindings {
    #![allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    #[macro_export]
    macro_rules! driver {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::bindings::*;
            #[allow(unused_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, CUresult::CUDA_SUCCESS);
        }};
    }

    #[macro_export]
    macro_rules! nvrtc {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::bindings::*;
            #[allow(unused_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, nvrtcResult::NVRTC_SUCCESS);
        }};
    }
}

mod context;
mod device;
mod event;
mod launch;
mod memory;
pub mod nvrtc;
mod stream;

pub trait AsRaw {
    type Raw;

    /// # Safety
    ///
    /// The caller must ensure that the returned item is dropped before the original item.
    unsafe fn as_raw(&self) -> Self::Raw;
}

#[inline(always)]
pub fn init() {
    driver!(cuInit(0));
}

pub use context::{Context, ContextGuard};
pub use device::Device;
pub use event::Event;
pub use launch::KernelFn;
pub use memory::LocalDevBlob;
pub use stream::Stream;

#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum CudaDataType {
    half,
    nv_bfloat16,
    float,
    double,
}

impl CudaDataType {
    #[inline]
    pub fn size(self) -> usize {
        match self {
            Self::half => 2,
            Self::nv_bfloat16 => 2,
            Self::float => 4,
            Self::double => 8,
        }
    }

    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::half => "half",
            Self::nv_bfloat16 => "nv_bfloat16",
            Self::float => "float",
            Self::double => "double",
        }
    }
}

use std::ffi::c_uint;

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
