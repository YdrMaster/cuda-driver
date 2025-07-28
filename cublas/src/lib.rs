#![cfg(any(nvidia, iluvatar, metax))]
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
            assert_eq!(err, cublasStatus_t::CUBLAS_STATUS_SUCCESS)
        }};
    }
}

mod blas;
pub use blas::{Computation, Cublas, CublasSpore, GemmScheme};

#[cfg(nvidia)]
mod cublaslt;
#[cfg(nvidia)]
pub use cublaslt::{
    CublasLt, CublasLtMatMulDescriptor, CublasLtMatrix, CublasLtMatrixLayout, CublasLtSpore,
    MatrixOrder,
};
