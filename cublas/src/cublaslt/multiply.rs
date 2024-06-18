use crate::bindings::{cublasComputeType_t, cublasLtMatmulDesc_t, cudaDataType};
use cuda::AsRaw;
use std::ptr::null_mut;

#[repr(transparent)]
pub struct CublasLtMatMulDescriptor(cublasLtMatmulDesc_t);

unsafe impl Send for CublasLtMatMulDescriptor {}
unsafe impl Sync for CublasLtMatMulDescriptor {}

impl AsRaw for CublasLtMatMulDescriptor {
    type Raw = cublasLtMatmulDesc_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Drop for CublasLtMatMulDescriptor {
    #[inline]
    fn drop(&mut self) {
        cublas!(cublasLtMatmulDescDestroy(self.0));
    }
}

impl CublasLtMatMulDescriptor {
    #[inline]
    pub fn new(compute_type: cublasComputeType_t, scale_type: cudaDataType) -> Self {
        let mut desc = null_mut();
        cublas!(cublasLtMatmulDescCreate(
            &mut desc,
            compute_type,
            scale_type
        ));
        Self(desc)
    }
}

#[test]
fn test_behavior() {
    // 据测试，这个类型是 CPU 上的，不需要在上下文中调用。
    let _mat_mul = CublasLtMatMulDescriptor::new(
        cublasComputeType_t::CUBLAS_COMPUTE_16F,
        cudaDataType::CUDA_R_16F,
    );
}
