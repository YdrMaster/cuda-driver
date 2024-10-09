//! See <https://github.com/NVIDIA/nccl/issues/565>.

#![cfg(detected_nccl)]
#![deny(warnings)]

#[macro_use]
#[allow(unused, non_upper_case_globals, non_camel_case_types, non_snake_case)]
pub mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));

    #[macro_export]
    macro_rules! nccl {
        ($f:expr) => {{
            #[allow(unused_imports)]
            use $crate::bindings::*;
            #[allow(unused_unsafe, clippy::macro_metavars_in_unsafe)]
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
    use digit_layout::types as ty;
    use ncclDataType_t::*;
    match data_type {
        ty::I8 => ncclInt8,
        ty::U8 => ncclUint8,
        ty::I32 => ncclInt32,
        ty::U32 => ncclUint32,
        ty::I64 => ncclInt64,
        ty::U64 => ncclUint64,
        ty::F16 => ncclFloat16,
        ty::F32 => ncclFloat32,
        ty::F64 => ncclFloat64,
        _ => panic!("{data_type} is not supported by NCCL"),
    }
}
