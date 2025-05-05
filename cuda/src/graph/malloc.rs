use super::{Graph, GraphNode, MemAllocNode, collect_dependencies};
use crate::bindings::CUDA_MEM_ALLOC_NODE_PARAMS;
use context_spore::AsRaw;
use std::{marker::PhantomData, ptr::null_mut};

impl Graph {
    pub fn add_alloc_node_with_params<'a>(
        &self,
        params: &mut CUDA_MEM_ALLOC_NODE_PARAMS,
        deps: impl IntoIterator<Item = &'a GraphNode<'a>>,
    ) -> MemAllocNode {
        let deps = collect_dependencies(deps);

        let mut node = null_mut();
        driver!(cuGraphAddMemAllocNode(
            &mut node,
            self.as_raw(),
            deps.as_ptr(),
            deps.len(),
            params,
        ));
        MemAllocNode(node, PhantomData)
    }
}
