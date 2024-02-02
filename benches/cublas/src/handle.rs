use crate::{bindings as cu, CublasLtMatMulDescriptor, CublasLtMatrix};
use cuda::{bindings::CUdeviceptr, AsRaw, Context, ContextGuard, DevSlice, Stream};
use half::{bf16, f16};
use std::{
    ffi::c_void,
    mem::{size_of, size_of_val, MaybeUninit},
    ptr::null_mut,
    sync::Arc,
};

pub struct CublasLtHandle {
    ctx: Arc<Context>,
    handle: cu::cublasLtHandle_t,
}

unsafe impl Send for CublasLtHandle {}
unsafe impl Sync for CublasLtHandle {}

impl AsRaw for CublasLtHandle {
    type Raw = cu::cublasLtHandle_t;

    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.handle
    }
}

impl Drop for CublasLtHandle {
    #[inline]
    fn drop(&mut self) {
        cublas!(cu::cublasLtDestroy(self.handle));
    }
}

impl CublasLtHandle {
    #[inline]
    pub fn create_on(ctx: &ContextGuard) -> Self {
        let mut handle = null_mut();
        cublas!(cublasLtCreate(&mut handle));
        Self {
            ctx: ctx.clone_ctx(),
            handle,
        }
    }

    pub fn tune(
        &self,
        matmul: &CublasLtMatMulDescriptor,
        a: &CublasLtMatrix,
        b: &CublasLtMatrix,
        c: &CublasLtMatrix,
        d: &CublasLtMatrix,
    ) -> (cu::cublasLtMatmulAlgo_t, usize) {
        let mut result = MaybeUninit::uninit();
        self.ctx.apply(|_| {
            let mut device = 0;
            driver!(cuCtxGetDevice(&mut device));

            let mut alignment = 0;
            driver!(cuDeviceGetAttribute(
                &mut alignment,
                CUdevice_attribute_enum::CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT,
                device,
            ));

            let workspace = u64::MAX;
            let alignment = alignment as u32;

            let mut preference = null_mut();
            cublas!(cublasLtMatmulPreferenceCreate(&mut preference));
            cublas!(cublasLtMatmulPreferenceSetAttribute(
                preference,
                cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace as *const _ as _,
                size_of_val(&workspace),
            ));
            cublas!(cublasLtMatmulPreferenceSetAttribute(
                preference,
                cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                &alignment as *const _ as _,
                size_of_val(&alignment),
            ));
            cublas!(cublasLtMatmulPreferenceSetAttribute(
                preference,
                cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                &alignment as *const _ as _,
                size_of_val(&alignment),
            ));
            cublas!(cublasLtMatmulPreferenceSetAttribute(
                preference,
                cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                &alignment as *const _ as _,
                size_of_val(&alignment),
            ));
            cublas!(cublasLtMatmulPreferenceSetAttribute(
                preference,
                cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                &alignment as *const _ as _,
                size_of_val(&alignment),
            ));

            let mut ans_n = 0;
            cublas!(cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                matmul.as_raw(),
                a.as_raw(),
                b.as_raw(),
                c.as_raw(),
                d.as_raw(),
                preference,
                1,
                result.as_mut_ptr(),
                &mut ans_n,
            ));
            assert_eq!(ans_n, 1);
        });

        let result = unsafe { result.assume_init() };
        (result.algo, result.workspaceSize)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &self,
        matmul: &CublasLtMatMulDescriptor,
        alpha: f32,
        a: &CublasLtMatrix,
        a_ptr: &impl AsRaw<Raw = CUdeviceptr>,
        b: &CublasLtMatrix,
        b_ptr: &impl AsRaw<Raw = CUdeviceptr>,
        beta: f32,
        c: &CublasLtMatrix,
        c_ptr: &impl AsRaw<Raw = CUdeviceptr>,
        d: &CublasLtMatrix,
        d_ptr: &impl AsRaw<Raw = CUdeviceptr>,
        algo: cu::cublasLtMatmulAlgo_t,
        workspace: &DevSlice,
        stream: &Stream,
    ) {
        let mut buf = MaybeUninit::<cu::cudaDataType>::uninit();
        let mut written = 0;
        cublas!(cublasLtMatmulDescGetAttribute(
            matmul.as_raw(),
            cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_SCALE_TYPE,
            buf.as_mut_ptr().cast(),
            size_of::<cu::cudaDataType>(),
            &mut written,
        ));
        assert_eq!(written, size_of::<cu::cudaDataType>());
        let alpha = FloatScalar::new(alpha, unsafe { buf.assume_init() });
        let beta = FloatScalar::new(beta, unsafe { buf.assume_init() });
        self.ctx.apply(|_| {
            cublas!(cublasLtMatmul(
                self.handle,
                matmul.as_raw(),
                alpha.as_ptr(),
                a_ptr.as_raw() as _,
                a.as_raw(),
                b_ptr.as_raw() as _,
                b.as_raw(),
                beta.as_ptr(),
                c_ptr.as_raw() as _,
                c.as_raw(),
                d_ptr.as_raw() as _,
                d.as_raw(),
                &algo,
                workspace.as_raw() as _,
                workspace.len(),
                stream.as_raw() as _,
            ))
        });
    }
}

struct FloatScalar([u8; 8]);

impl FloatScalar {
    fn new(f: f32, target: cu::cudaDataType) -> Self {
        let mut ans = Self([0; 8]);
        match target {
            cu::cudaDataType::CUDA_R_16F => unsafe {
                let f = f16::from_f32(f).to_bits().to_ne_bytes();
                std::ptr::copy_nonoverlapping(f.as_ptr(), ans.0.as_mut_ptr(), f.len());
            },
            cu::cudaDataType::CUDA_R_16BF => unsafe {
                let f = bf16::from_f32(f).to_bits().to_ne_bytes();
                std::ptr::copy_nonoverlapping(f.as_ptr(), ans.0.as_mut_ptr(), f.len());
            },
            cu::cudaDataType::CUDA_R_32F => unsafe {
                let f = f.to_bits().to_ne_bytes();
                std::ptr::copy_nonoverlapping(f.as_ptr(), ans.0.as_mut_ptr(), f.len());
            },
            cu::cudaDataType::CUDA_R_64F => unsafe {
                let f = (f as f64).to_bits().to_ne_bytes();
                std::ptr::copy_nonoverlapping(f.as_ptr(), ans.0.as_mut_ptr(), f.len());
            },
            _ => unreachable!(),
        }
        ans
    }

    #[inline]
    fn as_ptr(&self) -> *const c_void {
        self.0.as_ptr().cast()
    }
}
