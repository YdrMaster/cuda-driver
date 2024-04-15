#![deny(warnings)]
#![cfg(detected_cuda)]

#[cfg(feature = "half")]
pub extern crate half_ as half;

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
mod data_type;
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
pub use data_type::{CuTy, CudaDataType};
pub use dev_mem::{memcpy_d2d, memcpy_d2h, memcpy_h2d, DevByte, DevMem, DevMemSpore};
pub use device::Device;
pub use event::{Event, EventSpore};
pub use host_mem::{HostMem, HostMemSpore};
pub use nvrtc::{Dim3, KernelFn, Module, ModuleSpore, Ptx, Symbol};
pub use stream::{Stream, StreamSpore};
