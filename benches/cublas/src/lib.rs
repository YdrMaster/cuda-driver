#![cfg(detected_cuda)]

#[macro_use]
extern crate cuda;

#[macro_use]
pub mod bindings {
    #![allow(warnings)]
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

mod mat_mul;
mod matrix;
#[cfg(test)]
mod test;

pub use matrix::{CublasLtMatrix, CublasLtMatrixLayout, MatrixOrder};
