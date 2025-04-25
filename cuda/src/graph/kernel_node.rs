use super::{Graph, GraphNode, KernelNode};
use crate::{AsRaw, Dim3, KernelFn};
use std::{ffi::c_void, marker::PhantomData, ptr::null_mut};

impl KernelFn<'_> {
    pub fn add_to_graph<'g>(
        &self,
        attributes: (impl Into<Dim3>, impl Into<Dim3>, usize),
        params: *const *const c_void,
        graph: &'g Graph,
        dependencies: &[GraphNode],
    ) -> KernelNode<'g> {
        let mut node = null_mut();
        let dependencies = dependencies
            .iter()
            .map(|n| unsafe { n.as_raw() })
            .collect::<Box<_>>();
        let (grid, block, shared_mem) = attributes;
        let grid = grid.into();
        let block = block.into();
        driver!(cuGraphAddKernelNode_v2(
            &mut node,
            graph.as_raw(),
            dependencies.as_ptr(),
            dependencies.len(),
            &CUDA_KERNEL_NODE_PARAMS {
                func: self.as_raw(),
                gridDimX: grid.x,
                gridDimY: grid.y,
                gridDimZ: grid.z,
                blockDimX: block.x,
                blockDimY: block.y,
                blockDimZ: block.z,
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
