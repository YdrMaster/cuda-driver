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

#[macro_use]
pub mod macros {
    #[macro_export]
    macro_rules! cublaslt_matmul {
        ($compute_type:ident, $scale_type:ident) => {{
            $crate::CublasLtMatMulDescriptor::new(
                $crate::bindings::cublasComputeType_t::$compute_type,
                $crate::bindings::cudaDataType::$scale_type,
            )
        }};
    }

    #[macro_export]
    macro_rules! matmul {
        (with $handle:expr, on $stream:expr;
         do $matmul:expr, use $algo:expr, use $workspace:expr;
         ($alpha:expr; $a:expr, $a_ptr:expr; $b:expr, $b_ptr:expr)
      => ($beta:expr ; $c:expr, $c_ptr:expr; $d:expr, $d_ptr:expr)
        ) => {{
            $handle.matmul(
                &$matmul,
                $alpha,
                &$a,
                &$a_ptr,
                &$b,
                &$b_ptr,
                $beta,
                &$c,
                &$c_ptr,
                &$d,
                &$d_ptr,
                $algo,
                &$workspace,
                &$stream,
            );
        }};
    }
}

mod handle;
mod matrix;
mod multiply;

pub use handle::CublasLtHandle;
pub use matrix::{CublasLtMatrix, CublasLtMatrixLayout, MatrixOrder};
pub use multiply::CublasLtMatMulDescriptor;

#[cfg(test)]
mod test;
