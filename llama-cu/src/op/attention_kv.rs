use super::offset_ptr;
use core::ffi::c_void;
use cuda::{AsRaw, Stream, VirByte, bindings::CUresult::CUDA_SUCCESS};
use nn::Tensor;

#[allow(non_upper_case_globals, non_camel_case_types, non_snake_case, unused)]
mod bindings {
    include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
}

pub fn launch_attention_kv<const N: usize>(
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
    let q = NTTBuf::from(q);
    let k = NTTBuf::from(k);
    let v = NTTBuf::from(v);
    let k_cache = NTTBuf::from(k_cache);
    let k_cache_end = NTTBuf::from(k_cache_end);
    let v_cache = NTTBuf::from(v_cache);
    let v_cache_end = NTTBuf::from(v_cache_end);
    let mask = NTTBuf::from(mask);
    let o = NTTBuf::from(o);
    unsafe {
        let result = bindings::launch_attention_kv(
            stream.as_raw() as *mut c_void,
            q.to_ntt(),
            k.to_ntt(),
            v.to_ntt(),
            k_cache.to_ntt(),
            k_cache_end.to_ntt(),
            v_cache.to_ntt(),
            v_cache_end.to_ntt(),
            mask.to_ntt(),
            o.to_ntt(),
        );
        if result != CUDA_SUCCESS as i32 {
            panic!("launch_attention_kv failed: {}", result);
        }
    }
}

struct NTTBuf {
    data: *const VirByte,
    shape: Box<[u64]>,
    strides: Box<[i64]>,
}

impl<const N: usize> From<Tensor<*const VirByte, N>> for NTTBuf {
    fn from(value: Tensor<*const VirByte, N>) -> Self {
        let unit = value.dt().nbytes() as isize;
        Self {
            data: offset_ptr(&value),
            shape: value.shape().iter().map(|&d| d as _).collect(),
            strides: value.strides().iter().map(|&s| (s / unit) as _).collect(),
        }
    }
}

impl NTTBuf {
    fn to_ntt(&self) -> bindings::NineToothedTensor {
        bindings::NineToothedTensor {
            data: self.data.cast_mut().cast(),
            shape: self.shape.as_ptr().cast_mut(),
            strides: self.strides.as_ptr().cast_mut(),
        }
    }
}
