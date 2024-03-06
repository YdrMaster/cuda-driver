#![deny(warnings)]
#![cfg(detected_cuda)]

#[macro_use]
pub mod bindings {
    #![allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    #[macro_export]
    macro_rules! cublas {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::bindings::*;
            #[allow(unused_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }};
    }
}

mod cublas;
mod cublaslt;

pub use cublas::Cublas;
pub use cublaslt::CublasLt;
