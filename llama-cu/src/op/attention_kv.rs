pub mod bindings {
    #![allow(non_upper_case_globals)]
    #![allow(non_camel_case_types)]
    #![allow(non_snake_case)]
    #![allow(unused)]
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

use super::offset_ptr;
use core::ffi::c_void;
use cuda::{AsRaw, Stream, VirByte, bindings::CUresult::CUDA_SUCCESS};
use nn::Tensor;

/// SAFETY: NineToothedTensor 的生命周期与 tensor 相同
fn to_nine_toothed_tensor<const N: usize>(
    tensor: Tensor<*const VirByte, N>,
) -> bindings::NineToothedTensor {
    bindings::NineToothedTensor {
        data: offset_ptr(&tensor) as *mut c_void,
        shape: tensor.shape().as_ptr() as *mut u64,
        strides: tensor.strides().as_ptr() as *mut i64,
    }
}

fn launch_attention_kv<const N: usize>(
    q: Tensor<*const VirByte, N>,
    k: Tensor<*const VirByte, N>,
    v: Tensor<*const VirByte, N>,
    k_cache: Tensor<*const VirByte, N>,
    k_cache_end: Tensor<*const VirByte, N>,
    v_cache: Tensor<*const VirByte, N>,
    v_cache_end: Tensor<*const VirByte, N>,
    mask: Tensor<*const VirByte, N>,
    o: Tensor<*const VirByte, N>,
    stream: &Stream,
) {
    unsafe {
        let result = bindings::launch_attention_kv(
            stream.as_raw() as *mut c_void,
            to_nine_toothed_tensor(q),
            to_nine_toothed_tensor(k),
            to_nine_toothed_tensor(v),
            to_nine_toothed_tensor(k_cache),
            to_nine_toothed_tensor(k_cache_end),
            to_nine_toothed_tensor(v_cache),
            to_nine_toothed_tensor(v_cache_end),
            to_nine_toothed_tensor(mask),
            to_nine_toothed_tensor(o),
        );
        if result != CUDA_SUCCESS as i32 {
            panic!("launch_attention_kv failed: {}", result);
        }
    }
}
