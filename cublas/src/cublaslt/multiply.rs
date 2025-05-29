use crate::bindings::{hcblasComputeType_t, hcblasLtMatmulDesc_t, hpccDataType_t};
use cuda::AsRaw;
use std::ptr::null_mut;

#[repr(transparent)]
pub struct CublasLtMatMulDescriptor(hcblasLtMatmulDesc_t);

unsafe impl Send for CublasLtMatMulDescriptor {}
unsafe impl Sync for CublasLtMatMulDescriptor {}

impl AsRaw for CublasLtMatMulDescriptor {
    type Raw = hcblasLtMatmulDesc_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Drop for CublasLtMatMulDescriptor {
    #[inline]
    fn drop(&mut self) {
        cublas!(hcblasLtMatmulDescDestroy(self.0));
    }
}

impl CublasLtMatMulDescriptor {
    #[inline]
    pub fn new(compute_type: hcblasComputeType_t, scale_type: hpccDataType_t) -> Self {
        let mut desc = null_mut();
        cublas!(hcblasLtMatmulDescCreate(
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
        hcblasComputeType_t::HCBLAS_COMPUTE_16F,
        hpccDataType_t::HPCC_R_16F,
    );
}
