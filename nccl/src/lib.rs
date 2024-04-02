//! See <https://github.com/NVIDIA/nccl/issues/565>.

#![cfg(detected_nccl)]

#[macro_use]
pub mod bindings {
    #![allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    #[macro_export]
    macro_rules! nccl {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::bindings::*;
            #[allow(unused_unsafe)]
            let err = unsafe { $f };
            assert_eq!(err, ncclResult_t::ncclSuccess);
        }};
    }
}

#[test]
fn test() {
    use std::ptr::null_mut;
    let mut comm = null_mut();
    let devlist = [0];
    nccl!(ncclCommInitAll(&mut comm, 1, devlist.as_ptr()))
}
