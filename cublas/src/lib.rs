#![cfg(any(detected_cuda, detected_iluvatar))]
#![deny(warnings)]

#[macro_use]
#[allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    #[macro_export]
    macro_rules! cublas {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::bindings::*;
            #[allow(unused_unsafe, clippy::macro_metavars_in_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }};
    }
}

mod cublas;
#[cfg(detected_cuda)]
mod cublaslt;

pub use cublas::{Cublas, CublasSpore};
#[cfg(detected_cuda)]
pub use cublaslt::{
    CublasLt, CublasLtMatMulDescriptor, CublasLtMatrix, CublasLtMatrixLayout, CublasLtSpore,
    MatrixOrder,
};
