use crate::bindings::{macaDataType_t, mcblasComputeType_t, mcblasLtMatmulDesc_t};
use cuda::AsRaw;
use std::ptr::null_mut;

#[repr(transparent)]
pub struct CublasLtMatMulDescriptor(mcblasLtMatmulDesc_t);

unsafe impl Send for CublasLtMatMulDescriptor {}
unsafe impl Sync for CublasLtMatMulDescriptor {}

impl AsRaw for CublasLtMatMulDescriptor {
    type Raw = mcblasLtMatmulDesc_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Drop for CublasLtMatMulDescriptor {
    #[inline]
    fn drop(&mut self) {
        cublas!(mcblasLtMatmulDescDestroy(self.0));
    }
}

impl CublasLtMatMulDescriptor {
    #[inline]
    pub fn new(compute_type: mcblasComputeType_t, scale_type: macaDataType_t) -> Self {
        let mut desc = null_mut();
        cublas!(mcblasLtMatmulDescCreate(
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
        mcblasComputeType_t::MCBLAS_COMPUTE_16F,
        macaDataType_t::MACA_R_16F,
    );
}
