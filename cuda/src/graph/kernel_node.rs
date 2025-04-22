use crate::{AsRaw, Dim3, KernelFn, bindings::CUDA_KERNEL_NODE_PARAMS};
use std::{ffi::c_void, mem::MaybeUninit, ptr::null_mut};

impl super::KernelNode<'_> {
    pub(crate) fn get_params(&self) -> CUDA_KERNEL_NODE_PARAMS {
        let mut params = MaybeUninit::uninit();
        driver!(cuGraphKernelNodeGetParams_v2(self.0, params.as_mut_ptr()));
        unsafe { params.assume_init() }
    }

    pub fn set_params(
        &self,
        kernel: &KernelFn,
        grid_dims: impl Into<Dim3>,
        block_dims: impl Into<Dim3>,
        params: *const *const c_void,
        shared_mem: usize,
    ) {
        let grid_dims = grid_dims.into();
        let block_dims = block_dims.into();
        let params = CUDA_KERNEL_NODE_PARAMS {
            func: unsafe { kernel.as_raw() },
            gridDimX: grid_dims.x,
            gridDimY: grid_dims.y,
            gridDimZ: grid_dims.z,
            blockDimX: block_dims.x,
            blockDimY: block_dims.y,
            blockDimZ: block_dims.z,
            sharedMemBytes: shared_mem as _,
            kernelParams: params.cast_mut().cast(),
            extra: null_mut(),
            kern: null_mut(),
            ctx: null_mut(),
        };
        driver!(cuGraphKernelNodeSetParams_v2(self.0, &params))
    }
}
