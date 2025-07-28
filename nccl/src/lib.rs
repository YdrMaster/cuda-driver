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
            assert_eq!(err, hcclResult_t::hcclSuccess);
        }};
    }
}

mod all_gather;
mod all_reduce;
mod broadcast;
mod communicator;
mod group;

pub use bindings::hcclRedOp_t as ReduceType;
pub use communicator::Communicator;
pub use group::CommunicatorGroup;

use bindings::hcclDataType_t;
use digit_layout::DigitLayout;

#[inline]
fn convert(data_type: DigitLayout) -> hcclDataType_t {
    use digit_layout::types as ty;
    use hcclDataType_t::*;
    match data_type {
        ty::I8 => hcclInt8,
        ty::U8 => hcclUint8,
        ty::I32 => hcclInt32,
        ty::U32 => hcclUint32,
        ty::I64 => hcclInt64,
        ty::U64 => hcclUint64,
        ty::F16 => hcclFloat16,
        ty::F32 => hcclFloat32,
        ty::F64 => hcclFloat64,
        _ => panic!("{data_type} is not supported by NCCL"),
    }
}
