use super::{Graph, KernelNode};
use crate::{AsRaw, Dim3, KernelFn};
use std::{ffi::c_void, marker::PhantomData, ptr::null_mut};

impl KernelFn<'_> {
    pub fn add_to_graph<'g>(
        &self,
        grid_dims: impl Into<Dim3>,
        block_dims: impl Into<Dim3>,
        params: *const *const c_void,
        shared_mem: usize,
        graph: &'g Graph,
    ) -> KernelNode<'g> {
        let mut node = null_mut();
        let grid_dims = grid_dims.into();
        let block_dims = block_dims.into();
        driver!(cuGraphAddKernelNode_v2(
            &mut node,
            graph.as_raw(),
            null_mut(),
            0,
            &CUDA_KERNEL_NODE_PARAMS {
                func: self.as_raw(),
                gridDimX: grid_dims.x,
                gridDimY: grid_dims.y,
                gridDimZ: grid_dims.z,
                blockDimX: block_dims.x,
                blockDimY: block_dims.y,
                blockDimZ: block_dims.z,
                sharedMemBytes: shared_mem as _,
                kernelParams: params as _,
                extra: null_mut(),
                kern: null_mut(),
                ctx: null_mut(),
            }
        ));
        KernelNode(node, PhantomData)
    }
}
