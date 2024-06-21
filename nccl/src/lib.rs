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
use digit_layout::DigitLayout;

#[inline]
fn convert(data_type: DigitLayout) -> ncclDataType_t {
    use digit_layout::types::*;
    match data_type {
        I8 => ncclDataType_t::ncclInt8,
        U8 => ncclDataType_t::ncclUint8,
        I32 => ncclDataType_t::ncclInt32,
        U32 => ncclDataType_t::ncclUint32,
        I64 => ncclDataType_t::ncclInt64,
        U64 => ncclDataType_t::ncclUint64,
        F16 => ncclDataType_t::ncclHalf,
        F32 => ncclDataType_t::ncclFloat,
        F64 => ncclDataType_t::ncclDouble,
        _ => panic!("Digit layout is not supported by NCCL"),
    }
}
