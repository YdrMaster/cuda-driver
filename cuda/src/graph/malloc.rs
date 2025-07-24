use super::{Graph, GraphNode, MemAllocNode, collect_dependencies};
use crate::bindings::mcMemAllocNodeParams;
use context_spore::AsRaw;
use std::{marker::PhantomData, ptr::null_mut};

impl Graph {
    pub fn add_alloc_node_with_params<'a>(
        &self,
        params: &mut mcMemAllocNodeParams,
        deps: impl IntoIterator<Item = &'a GraphNode<'a>>,
    ) -> MemAllocNode {
        let deps = collect_dependencies(deps);

        let mut node = null_mut();
        driver!(mcGraphAddMemAllocNode(
            &mut node,
            self.as_raw(),
            deps.as_ptr(),
            deps.len(),
            params,
        ));
        MemAllocNode(node, PhantomData)
    }
}

#[cfg(all(not(metax), test))]
mod test {
    use crate::{AsRaw, Device, Graph, GraphNode};

    #[test]
    fn test_behavior() {
        if let Err(crate::NoDevice) = crate::init() {
            return;
        }
        // 捕获分配行为并提取分配节点参数
        let mut params = unsafe { std::mem::zeroed() };
        let _graph = Device::new(0).context().apply(|ctx| {
            let stream = ctx.stream().capture();
            let mem = stream.malloc::<u8>(4 << 10);
            stream.free(mem);
            let graph = stream.end();
            for node in graph.nodes() {
                match node {
                    GraphNode::MemAlloc(node) => {
                        driver!(cuGraphMemAllocNodeGetParams(node.as_raw(), &mut params));
                        println!("{params:#x?}")
                    }
                    GraphNode::MemFree(node) => {
                        let mut ptr = 0;
                        driver!(cuGraphMemFreeNodeGetParams(node.as_raw(), &mut ptr));
                        println!("{ptr:#x}")
                    }
                    _ => unreachable!(),
                }
            }
            graph
        });
        // 使用捕获的参数构造另一个图中的分配节点
        let graph = Graph::new();
        let mut params_ = params;
        graph.add_alloc_node_with_params(&mut params_, &[]);
        // 分配的虚地址不受参数控制
        assert_ne!(params.dptr, params_.dptr);
        println!("{params:#x?}")
    }
}
