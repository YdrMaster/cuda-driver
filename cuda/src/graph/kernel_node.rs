use super::{Graph, GraphNode, KernelNode};
use crate::{AsRaw, Dim3, KernelFn, bindings::CUDA_KERNEL_NODE_PARAMS};
use std::{ffi::c_void, marker::PhantomData, ptr::null_mut};

impl Graph {
    pub fn add_kernel_call(
        &self,
        f: &KernelFn,
        attributes: (impl Into<Dim3>, impl Into<Dim3>, usize),
        params: *const *const c_void,
        dependencies: &[GraphNode],
    ) -> KernelNode {
        let (grid, block, shared_mem) = attributes;
        let grid = grid.into();
        let block = block.into();
        let params = CUDA_KERNEL_NODE_PARAMS {
            func: unsafe { f.as_raw() },
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
        };

        self.add_kernel_node_with_params(&params, dependencies)
    }

    pub fn add_kernel_node(&self, node: &KernelNode, dependencies: &[GraphNode]) -> KernelNode {
        let mut params = unsafe { std::mem::zeroed() };
        driver!(cuGraphKernelNodeGetParams_v2(node.as_raw(), &mut params));

        self.add_kernel_node_with_params(&params, dependencies)
    }

    pub fn add_kernel_node_with_params(
        &self,
        params: &CUDA_KERNEL_NODE_PARAMS,
        dependencies: &[GraphNode],
    ) -> KernelNode {
        let dependencies = dependencies
            .iter()
            .map(|n| unsafe { n.as_raw() })
            .collect::<Box<_>>();

        let mut node = null_mut();
        driver!(cuGraphAddKernelNode_v2(
            &mut node,
            self.as_raw(),
            dependencies.as_ptr(),
            dependencies.len(),
            params
        ));
        KernelNode(node, PhantomData)
    }
}
