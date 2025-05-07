mod free;
mod host_fn;
mod kernel;
mod malloc;
mod memcpy;
mod memset;

use crate::{
    CurrentCtx, Stream,
    bindings::{
        CUgraph, CUgraphExec, CUgraphNode,
        CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL as LOCAL,
    },
};
use context_spore::{AsRaw, impl_spore};
use std::{ffi::CString, marker::PhantomData, ops::Deref, path::Path, ptr::null_mut, str::FromStr};

#[repr(transparent)]
pub struct Graph(CUgraph);

impl_spore!(GraphExec and GraphExecSpore by (CurrentCtx, CUgraphExec));

#[repr(transparent)]
pub struct CaptureStream<'ctx>(Stream<'ctx>);

impl<'ctx> Stream<'ctx> {
    pub fn capture(self) -> CaptureStream<'ctx> {
        driver!(cuStreamBeginCapture_v2(self.as_raw(), LOCAL));
        CaptureStream(self)
    }

    pub fn launch_graph(&self, graph: &GraphExec) -> &Self {
        driver!(cuGraphLaunch(graph.0.rss, self.as_raw()));
        self
    }
}

impl CaptureStream<'_> {
    pub fn end(self) -> Graph {
        let mut graph = null_mut();
        driver!(cuStreamEndCapture(self.0.as_raw(), &mut graph));
        Graph(graph)
    }
}

impl<'ctx> Deref for CaptureStream<'ctx> {
    type Target = Stream<'ctx>;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Graph {
    pub fn new() -> Self {
        let mut graph = null_mut();
        driver!(cuGraphCreate(&mut graph, 0));
        Self(graph)
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        driver!(cuGraphDestroy(self.0))
    }
}

impl AsRaw for Graph {
    type Raw = CUgraph;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Graph {
    pub fn save_dot(&self, path: impl AsRef<Path>) {
        let path = CString::from_str(&path.as_ref().display().to_string()).unwrap();
        driver!(cuGraphDebugDotPrint(self.0, path.as_ptr().cast(), u32::MAX))
    }

    pub fn nodes(&self) -> Vec<GraphNode> {
        let mut num = 0;
        driver!(cuGraphGetNodes(self.0, null_mut(), &mut num));
        let mut ans = vec![null_mut(); num];
        driver!(cuGraphGetNodes(self.0, ans.as_mut_ptr(), &mut num));
        assert_eq!(num, ans.len());
        ans.into_iter().map(GraphNode::new).collect()
    }
}

impl CurrentCtx {
    pub fn instantiate<'ctx>(&'ctx self, graph: &Graph) -> GraphExec<'ctx> {
        let mut exec = null_mut();
        driver!(cuGraphInstantiateWithFlags(
            &mut exec,
            graph.0,
            CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as _
        ));
        GraphExec(unsafe { self.wrap_raw(exec) }, PhantomData)
    }
}

impl Drop for GraphExec<'_> {
    fn drop(&mut self) {
        driver!(cuGraphExecDestroy(self.0.rss))
    }
}

impl AsRaw for GraphExec<'_> {
    type Raw = CUgraphExec;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss
    }
}

pub enum GraphNode<'g> {
    Kernel(KernelNode<'g>),
    MemAlloc(MemAllocNode<'g>),
    MemFree(MemFreeNode<'g>),
    Memcpy(MemcpyNode<'g>),
    Memset(MemsetNode<'g>),
    HostFn(HostFnNode<'g>),
    SubGraph(SubGraphNode<'g>),
    Empty(EmptyNode<'g>),
    EventWait(EventWaitNode<'g>),
    EventRecord(EventRecordNode<'g>),
    ExtSemasSignal(ExtSemasSignalNode<'g>),
    ExtSemasWait(ExtSemasWaitNode<'g>),
}

macro_rules! typed_node {
    ($( $name:ident )+) => {
        $(
            #[repr(transparent)]
            pub struct $name<'g>(CUgraphNode, PhantomData<&'g ()>);

            impl AsRaw for $name<'_> {
                type Raw = CUgraphNode;
                #[inline]
                unsafe fn as_raw(&self) -> Self::Raw {
                    self.0
                }
            }

            impl<'a> From<$name<'a>> for GraphNode<'a> {
                fn from(node: $name) -> Self {
                    GraphNode::new(node.0)
                }
            }
        )+
    };
}

typed_node! {
    KernelNode
    MemAllocNode
    MemFreeNode
    MemcpyNode
    MemsetNode
    HostFnNode
    SubGraphNode
    EmptyNode
    EventWaitNode
    EventRecordNode
    ExtSemasSignalNode
    ExtSemasWaitNode
}

impl GraphNode<'_> {
    pub(super) fn new(raw: CUgraphNode) -> Self {
        use crate::bindings::CUgraphNodeType as ty;

        let mut type_ = ty::CU_GRAPH_NODE_TYPE_EMPTY;
        driver!(cuGraphNodeGetType(raw, &mut type_));

        #[rustfmt::skip]
        let ans = match type_ {
            ty::CU_GRAPH_NODE_TYPE_KERNEL           => Self::Kernel        (KernelNode        (raw, PhantomData)),
            ty::CU_GRAPH_NODE_TYPE_MEM_ALLOC        => Self::MemAlloc      (MemAllocNode      (raw, PhantomData)),
            ty::CU_GRAPH_NODE_TYPE_MEM_FREE         => Self::MemFree       (MemFreeNode       (raw, PhantomData)),
            ty::CU_GRAPH_NODE_TYPE_MEMCPY           => Self::Memcpy        (MemcpyNode        (raw, PhantomData)),
            ty::CU_GRAPH_NODE_TYPE_MEMSET           => Self::Memset        (MemsetNode        (raw, PhantomData)),
            ty::CU_GRAPH_NODE_TYPE_HOST             => Self::HostFn        (HostFnNode        (raw, PhantomData)),
            ty::CU_GRAPH_NODE_TYPE_GRAPH            => Self::SubGraph      (SubGraphNode      (raw, PhantomData)),
            ty::CU_GRAPH_NODE_TYPE_EMPTY            => Self::Empty         (EmptyNode         (raw, PhantomData)),
            ty::CU_GRAPH_NODE_TYPE_WAIT_EVENT       => Self::EventWait     (EventWaitNode     (raw, PhantomData)),
            ty::CU_GRAPH_NODE_TYPE_EVENT_RECORD     => Self::EventRecord   (EventRecordNode   (raw, PhantomData)),
            ty::CU_GRAPH_NODE_TYPE_EXT_SEMAS_SIGNAL => Self::ExtSemasSignal(ExtSemasSignalNode(raw, PhantomData)),
            ty::CU_GRAPH_NODE_TYPE_EXT_SEMAS_WAIT   => Self::ExtSemasWait  (ExtSemasWaitNode  (raw, PhantomData)),
            _ => todo!(),
        };
        ans
    }
}

impl AsRaw for GraphNode<'_> {
    type Raw = CUgraphNode;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        macro_rules! case {
            ( $( $variant:ident )+ ) => {
                match self {
                    $(GraphNode::$variant(node) => unsafe { node.as_raw() },)+
                }
            };
        }

        case! {
            Kernel
            MemAlloc
            MemFree
            Memcpy
            Memset
            HostFn
            SubGraph
            Empty
            EventWait
            EventRecord
            ExtSemasSignal
            ExtSemasWait
        }
    }
}

fn collect_dependencies<'a>(
    deps: impl IntoIterator<Item = &'a GraphNode<'a>>,
) -> Box<[CUgraphNode]> {
    deps.into_iter().map(|n| unsafe { n.as_raw() }).collect()
}

#[cfg(test)]
mod test {
    use crate::{Device, Ptx, Symbol, params};
    use context_spore::AsRaw;
    use std::{ffi::CString, ptr::null_mut, str::FromStr};

    #[test]
    fn test_behavior() {
        if let Err(crate::NoDevice) = crate::init() {
            return;
        }
        // 不需要在上下文中创建和释放 cuda graph
        let mut graph = null_mut();
        driver!(cuGraphCreate(&mut graph, 0));
        assert!(!graph.is_null());
        // 但 cuda graph 实例化需要在上下文中
        let mut exec = null_mut();
        driver!(
            CUDA_ERROR_INVALID_CONTEXT,
            cuGraphInstantiateWithFlags(
                &mut exec,
                graph,
                CUgraphInstantiate_flags::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH as _
            )
        );
        // √
        driver!(cuGraphDestroy(graph))
    }

    #[test]
    fn test_save_dot() {
        const CODE: &str = r#"
extern "C" __global__ void add(float *a, float const *b) {
    a[threadIdx.x] += b[threadIdx.x];
}
extern "C" __global__ void sub(float *a, float const *b) {
    a[threadIdx.x] -= b[threadIdx.x];
}
extern "C" __global__ void mul(float *a, float const *b) {
    a[threadIdx.x] *= b[threadIdx.x];
}
"#;

        if let Err(crate::NoDevice) = crate::init() {
            return;
        }

        Device::new(0).context().apply(|ctx| {
            let (ptx, _log) = Ptx::compile(CODE, ctx.dev().compute_capability());
            let module = ctx.load(&ptx.unwrap());

            let mut kernels =
                Symbol::search(CODE).map(|name| module.get_kernel(name.to_c_string()));
            let add = kernels.next().unwrap();
            let sub = kernels.next().unwrap();
            let mul = kernels.next().unwrap();

            let a_host = ctx.malloc_host::<f32>(1024);
            let b_host = ctx.malloc_host::<f32>(1024);
            let c_host = ctx.malloc_host::<f32>(1024);
            let d_host = ctx.malloc_host::<f32>(1024);
            let mut ans = ctx.malloc_host::<f32>(1024);

            let stream = ctx.stream();
            let stream = stream.capture();

            {
                let mut a = stream.from_host(&a_host);
                let mut b = stream.from_host(&b_host);
                let mut c = stream.from_host(&c_host);
                let mut d = stream.from_host(&d_host);

                stream
                    .launch(
                        &add,
                        (1, 1024, 0),
                        &params![a.as_mut_ptr(), b.as_mut_ptr()].to_ptrs(),
                    )
                    .launch(
                        &sub,
                        (1, 1024, 0),
                        &params![c.as_mut_ptr(), d.as_mut_ptr()].to_ptrs(),
                    )
                    .launch(
                        &mul,
                        (1, 1024, 0),
                        &params![a.as_mut_ptr(), c.as_mut_ptr()].to_ptrs(),
                    )
                    .memcpy_d2h(&mut ans, &a)
                    .free(a)
                    .free(b)
                    .free(c)
                    .free(d);
            }

            stream
                .end()
                .save_dot(std::env::current_dir().unwrap().join("graph.dot"))
        })
    }

    #[test]
    fn test_launch() {
        const CODE: &str =
            r#"extern "C" __global__ void print(int n) { printf("Hello, world(%d)!\n", n); }"#;

        if let Err(crate::NoDevice) = crate::init() {
            return;
        }

        Device::new(0).context().apply(|ctx| {
            let (ptx, _log) = Ptx::compile(CODE, ctx.dev().compute_capability());
            let module = ctx.load(&ptx.unwrap());
            let kernel = module.get_kernel(CString::from_str("print").unwrap());

            let stream = ctx.stream().capture();
            // cuda graph 会将 kernel 用到的参数拷贝保存在节点中
            stream.launch(&kernel, ((), (), 0), &params![10].to_ptrs());
            let graph = stream.end();
            let stream = ctx.stream();

            // 执行图可以多次执行

            let exec = ctx.instantiate(&graph);
            stream
                .launch_graph(&exec)
                .launch_graph(&exec)
                .launch_graph(&exec);

            // 释放掉图之后执行图仍能执行

            let _ = graph;
            stream
                .launch_graph(&exec)
                .launch_graph(&exec)
                .launch_graph(&exec);

            let stream = stream.capture();

            // 不能向捕获流上发射 cuda graph
            // 否则会破坏捕获流、实例化图和模块

            driver!(
                CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
                cuGraphLaunch(exec.as_raw(), stream.as_raw())
            );
            std::mem::forget(stream);

            driver!(
                CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
                cuGraphExecDestroy(exec.as_raw())
            );
            std::mem::forget(exec);

            driver!(
                CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED,
                cuModuleUnload(module.as_raw())
            );
            std::mem::forget(module);
        })
    }
}
