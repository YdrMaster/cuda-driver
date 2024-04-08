#![deny(warnings)]
#![cfg(detected_cuda)]

#[macro_use]
#[allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
pub mod bindings {
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
mod dev_mem;
mod device;
mod event;
mod host_mem;
mod nvrtc;
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

pub use context::{
    ctx_eq, not_owned, owned, Context, ContextGuard, ContextResource, ContextSpore,
    ResourceOwnership,
};
pub use dev_mem::{memcpy_d2h, memcpy_h2d, DevByte, DevMem, DevMemSpore};
pub use device::Device;
pub use event::{Event, EventSpore};
pub use host_mem::{HostMem, HostMemSpore};
pub use nvrtc::{Dim3, KernelFn, Module, ModuleSpore, Ptx, Symbol};
pub use stream::{Stream, StreamSpore};

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
