#![cfg(detected_cuda)]

mod mat_mul;
mod matrix;
#[cfg(test)]
mod test;

mod bindings {
    #![allow(warnings)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    macro_rules! cuda {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::bindings::*;
            #[allow(unused_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, cudaError::cudaSuccess);
        }};
    }

    macro_rules! cublas {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::bindings::*;
            #[allow(unused_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, cublasStatus_t::CUBLAS_STATUS_SUCCESS);
        }};
    }

    pub(crate) use {cublas, cuda};
}
