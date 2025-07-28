use super::{Graph, GraphNode, KernelNode};
use crate::{
    Dim3, KernelFn,
    bindings::{dim3, hcKernelNodeParams},
};
use context_spore::AsRaw;
use std::{ffi::c_void, ptr::null_mut};

impl Graph {
    pub fn add_kernel_call<'a>(
        &self,
        f: &KernelFn,
        attrs: (impl Into<Dim3>, impl Into<Dim3>, usize),
        params: &[*const c_void],
        deps: impl IntoIterator<Item = &'a GraphNode<'a>>,
    ) -> KernelNode {
        let (grid, block, shared_mem) = attrs;
        let grid = grid.into();
        let block = block.into();
        let num_blocks: dim3 = dim3 {
            x: grid.x,
            y: grid.y,
            z: grid.z,
        };
        let dim_blocks: dim3 = dim3 {
            x: block.x,
            y: block.y,
            z: block.z,
        };
        let params = hcKernelNodeParams {
            func: unsafe { f.as_raw().cast() },
            gridDim: num_blocks,
            blockDim: dim_blocks,
            sharedMemBytes: shared_mem as _,
            kernelParams: params.as_ptr() as _,
            extra: null_mut(),
            #[cfg(nvidia)]
            kern: null_mut(),
            #[cfg(nvidia)]
            ctx: null_mut(),
        };

        self.add_kernel_node_with_params(&params, deps)
    }

    pub fn add_kernel_node<'a>(
        &self,
        node: &KernelNode,
        deps: impl IntoIterator<Item = &'a GraphNode<'a>>,
    ) -> KernelNode {
        #[cfg(not(iluvatar))]
        {
            let mut params = unsafe { std::mem::zeroed() };
            driver!(hcGraphKernelNodeGetParams(node.as_raw(), &mut params));
            self.add_kernel_node_with_params(&params, deps)
        }
        #[cfg(iluvatar)]
        {
            let _ = (node, deps);
            unimplemented!()
        }
    }

    pub fn add_kernel_node_with_params<'a>(
        &self,
        params: &hcKernelNodeParams,
        deps: impl IntoIterator<Item = &'a GraphNode<'a>>,
    ) -> KernelNode {
        #[cfg(not(iluvatar))]
        {
            let deps = super::collect_dependencies(deps);
            let mut node = null_mut();
            driver!(hcGraphAddKernelNode(
                &mut node,
                self.as_raw(),
                deps.as_ptr(),
                deps.len(),
                params,
            ));
            KernelNode(node, std::marker::PhantomData)
        }
        #[cfg(iluvatar)]
        {
            let _ = (params, deps);
            unimplemented!()
        }
    }
}
