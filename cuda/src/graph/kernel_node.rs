use super::{Graph, GraphNode, HostFnNode, KernelNode};
use crate::{
    AsRaw, Dim3, KernelFn,
    bindings::{CUDA_HOST_NODE_PARAMS, CUDA_KERNEL_NODE_PARAMS, CUhostFn},
};
use std::{ffi::c_void, marker::PhantomData, ptr::null_mut};

impl Graph {
    pub fn add_kernel_call(
        &self,
        f: &KernelFn,
        attrs: (impl Into<Dim3>, impl Into<Dim3>, usize),
        params: &[*const c_void],
        dependencies: &[GraphNode],
    ) -> KernelNode {
        let (grid, block, shared_mem) = attrs;
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
            kernelParams: params.as_ptr() as _,
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

    pub fn add_host_node(
        &self,
        host_fn: CUhostFn,
        user_data: *mut c_void,
        dependencies: &[GraphNode],
    ) -> HostFnNode {
        let dependencies = dependencies
            .iter()
            .map(|n| unsafe { n.as_raw() })
            .collect::<Box<_>>();

        let cuda_host_node_params = CUDA_HOST_NODE_PARAMS {
            fn_: host_fn,
            userData: user_data,
        };
        let mut node = null_mut();
        driver!(cuGraphAddHostNode(
            &mut node,
            self.as_raw(),
            dependencies.as_ptr(),
            dependencies.len(),
            &cuda_host_node_params,
        ));
        HostFnNode(node, PhantomData)
    }

    pub fn add_host_node_with_rust_fn(
        &self,
        host_fn: impl Fn() + Send + Sync + 'static,
        dependencies: &[GraphNode],
    ) -> HostFnNode {
        extern "C" fn c_host_fn(user_data: *mut c_void) {
            let host_fn = unsafe { Box::from_raw(user_data as *mut Box<dyn Fn()>) };
            host_fn();
        }
        let boxed_closure: Box<dyn Fn()> = Box::new(host_fn);

        let user_data = Box::into_raw(Box::new(boxed_closure));
        self.add_host_node(Some(c_host_fn), user_data as *mut c_void, dependencies)
    }
}
