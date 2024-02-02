use crate::bindings as cu;
use cuda::AsRaw;
use std::ptr::null_mut;

#[repr(transparent)]
pub struct CublasLtMatMulDescriptor(cu::cublasLtMatmulDesc_t);

unsafe impl Send for CublasLtMatMulDescriptor {}
unsafe impl Sync for CublasLtMatMulDescriptor {}

impl AsRaw for CublasLtMatMulDescriptor {
    type Raw = cu::cublasLtMatmulDesc_t;

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
    pub fn new(compute_type: cu::cublasComputeType_t, scale_type: cu::cudaDataType) -> Self {
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
    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    let desc = dev
        .context()
        .apply(|_| cublaslt_matmul!(CUBLAS_COMPUTE_32F, CUDA_R_32F));
    dev.context().apply(|_| {
        // cublasLtMatmulDesc_t 在一个上下文中创建后，可以在其他上下文中使用
        cublas!(cublasLtMatmulDescSetAttribute(
            desc.as_raw(),
            cu::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
            &cu::cublasOperation_t::CUBLAS_OP_N as *const _ as _,
            std::mem::size_of::<cu::cublasOperation_t>(),
        ));
    });
    // 释放 cublas 对象不需要存在当前上下文
    drop(desc);
}
