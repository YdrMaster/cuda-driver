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
mod memory;
pub mod nvrtc;
mod stream;

pub trait AsRaw {
    type Raw;
    unsafe fn as_raw(&self) -> Self::Raw;
}

pub trait WithCtx {
    unsafe fn ctx(&self) -> bindings::CUcontext;
}

#[inline(always)]
pub fn init() {
    driver!(cuInit(0));
}

pub use context::{Context, ContextGuard};
pub use device::Device;
pub use event::Event;
pub use memory::DevicePtr;
pub use stream::Stream;
