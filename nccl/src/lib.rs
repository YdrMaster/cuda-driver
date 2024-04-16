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

mod all_gather;
mod all_reduce;
mod broadcast;
mod communicator;
mod group;

pub use bindings::ncclRedOp_t as ReduceType;
pub use communicator::Communicator;
pub use group::CommunicatorGroup;

use bindings::ncclDataType_t;
use cuda::CudaDataType;

#[inline]
fn convert(data_type: CudaDataType) -> ncclDataType_t {
    match data_type {
        CudaDataType::i8 => ncclDataType_t::ncclInt8,
        CudaDataType::u8 => ncclDataType_t::ncclUint8,
        CudaDataType::i32 => ncclDataType_t::ncclInt32,
        CudaDataType::u32 => ncclDataType_t::ncclUint32,
        CudaDataType::i64 => ncclDataType_t::ncclInt64,
        CudaDataType::u64 => ncclDataType_t::ncclUint64,
        CudaDataType::f16 => ncclDataType_t::ncclHalf,
        CudaDataType::f32 => ncclDataType_t::ncclFloat,
        CudaDataType::f64 => ncclDataType_t::ncclDouble,
        dt => panic!("\"{}\" is not supported by NCCL", dt.name()),
    }
}
