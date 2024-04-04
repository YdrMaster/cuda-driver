//! See <https://github.com/NVIDIA/nccl/issues/565>.

#![cfg(detected_nccl)]

#[macro_use]
#[allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
pub mod bindings {
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

mod all_reduce;
mod communicator;
mod group;

pub use bindings::ncclRedOp_t as ReduceType;
pub use communicator::Communicator;
pub use group::CommunicatorGroup;

use bindings::ncclDataType_t;
use cuda::CudaDataType;

fn convert(data_type: cuda::CudaDataType) -> ncclDataType_t {
    match data_type {
        CudaDataType::half => ncclDataType_t::ncclHalf,
        CudaDataType::float => ncclDataType_t::ncclFloat,
        CudaDataType::double => ncclDataType_t::ncclDouble,
        CudaDataType::nv_bfloat16 => panic!("nv_bfloat16 is not supported by NCCL"),
    }
}
