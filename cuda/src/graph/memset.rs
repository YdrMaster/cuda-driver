use super::{Graph, GraphNode, MemsetNode};
use crate::{bindings::hcMemsetParams, graph::collect_dependencies};
use context_spore::AsRaw;
use std::{marker::PhantomData, ptr::null_mut};

impl Graph {
    pub fn add_memset_node_with_params<'a>(
        &self,
        params: &hcMemsetParams,
        deps: impl IntoIterator<Item = &'a GraphNode<'a>>,
    ) -> MemsetNode {
        let deps = collect_dependencies(deps);

        let mut node = null_mut();
        driver!(hcGraphAddMemsetNode(
            &mut node,
            self.as_raw(),
            deps.as_ptr(),
            deps.len(),
            params,
            #[cfg(not(metax))]
            null_mut(),
        ));
        MemsetNode(node, PhantomData)
    }
}
