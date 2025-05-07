use super::{Graph, GraphNode, HostFnNode, collect_dependencies};
use crate::bindings::{CUDA_HOST_NODE_PARAMS, CUhostFn};
use context_spore::AsRaw;
use std::{ffi::c_void, marker::PhantomData, ptr::null_mut};

impl Graph {
    pub fn add_host_node_with_rust_fn<'a>(
        &self,
        host_fn: impl Fn() + Send + Sync + 'static,
        deps: impl IntoIterator<Item = &'a GraphNode<'a>>,
    ) -> HostFnNode {
        extern "C" fn c_host_fn(user_data: *mut c_void) {
            let host_fn = unsafe { Box::from_raw(user_data as *mut Box<dyn Fn()>) };
            host_fn();
        }
        let boxed_closure: Box<dyn Fn()> = Box::new(host_fn);

        let user_data = Box::into_raw(Box::new(boxed_closure));
        self.add_host_node(Some(c_host_fn), user_data as *mut c_void, deps)
    }

    pub fn add_host_node<'a>(
        &self,
        host_fn: CUhostFn,
        user_data: *mut c_void,
        deps: impl IntoIterator<Item = &'a GraphNode<'a>>,
    ) -> HostFnNode {
        let deps = collect_dependencies(deps);

        let cuda_host_node_params = CUDA_HOST_NODE_PARAMS {
            fn_: host_fn,
            userData: user_data,
        };
        let mut node = null_mut();
        driver!(cuGraphAddHostNode(
            &mut node,
            self.as_raw(),
            deps.as_ptr(),
            deps.len(),
            &cuda_host_node_params,
        ));
        HostFnNode(node, PhantomData)
    }
}

#[cfg(test)]
mod test {
    use crate::{AsRaw, Ptx, bindings::CUDA_HOST_NODE_PARAMS, graph::Graph, params};
    use std::ptr::{null, null_mut};

    #[test]
    fn test_launch_host_fn() {
        const CODE: &str = r#"extern "C" __global__ void print(int n) { printf("Hello, world(%d)! from GPU\n", n); }"#;

        use context_spore::AsRaw;
        // 主机函数不得执行任何依赖于未完成 CUDA 工作的同步操作
        // 主机函数内部不得调用任何 CUDA API，否则可能返回 CUDA_ERROR_NOT_PERMITTED（但非强制要求）。
        extern "C" fn host_fn(e: *mut core::ffi::c_void) {
            let num: usize = unsafe { *e.cast::<usize>() };
            println!("Hello World from CPU! num: {}", num);
        }

        let one: usize = 1;
        let two: usize = 2;
        if let Err(crate::NoDevice) = crate::init() {
            return;
        }
        crate::Device::new(0).context().apply(|ctx| {
            let (ptx, _log) = Ptx::compile(CODE, ctx.dev().compute_capability());
            let module = ctx.load(&ptx.unwrap());
            let kernel = module.get_kernel(c"print");

            let stream = ctx.stream();
            driver!(cuLaunchHostFunc(
                stream.as_raw(),
                Some(host_fn as unsafe extern "C" fn(*mut core::ffi::c_void)),
                &one as *const usize as *mut core::ffi::c_void
            ));

            stream.launch(&kernel, (1, 1, 0), &params![1].to_ptrs());

            driver!(cuLaunchHostFunc(
                stream.as_raw(),
                Some(host_fn as unsafe extern "C" fn(*mut core::ffi::c_void)),
                &two as *const usize as *mut core::ffi::c_void
            ));

            stream.launch(&kernel, (1, 1, 0), &params![2].to_ptrs());
        });
    }

    #[test]
    fn test_host_graph_dot() {
        const CODE: &str = r#"extern "C" __global__ void print(int n) { printf("Hello, world(%d)! from GPU\n", n); }"#;

        use context_spore::AsRaw;
        extern "C" fn host_fn(e: *mut core::ffi::c_void) {
            let num: usize = e as usize;
            println!("Hello World from CPU! num: {}", num);
        }

        if let Err(crate::NoDevice) = crate::init() {
            return;
        }
        crate::Device::new(0).context().apply(|ctx| {
            let (ptx, _log) = Ptx::compile(CODE, ctx.dev().compute_capability());
            let module = ctx.load(&ptx.unwrap());
            let kernel = module.get_kernel(c"print");

            let stream = ctx.stream();
            let stream = stream.capture();
            driver!(cuLaunchHostFunc(
                stream.as_raw(),
                Some(host_fn as unsafe extern "C" fn(*mut core::ffi::c_void)),
                1 as *mut core::ffi::c_void
            ));

            stream.launch(&kernel, (1, 1, 0), &params![1].to_ptrs());

            driver!(cuLaunchHostFunc(
                stream.as_raw(),
                Some(host_fn as unsafe extern "C" fn(*mut core::ffi::c_void)),
                2 as *mut core::ffi::c_void
            ));

            stream.launch(&kernel, (1, 1, 0), &params![2].to_ptrs());

            stream
                .end()
                .save_dot(std::env::current_dir().unwrap().join("host_graph.dot"))
        });
    }

    #[test]
    fn test_host_graph_node() {
        const CODE: &str = r#"extern "C" __global__ void print(int n) { printf("Hello, world(%d)! from GPU\n", n); }"#;

        extern "C" fn host_fn(e: *mut core::ffi::c_void) {
            let num: usize = e as usize;
            println!("Hello World from CPU! num: {}", num);
        }

        if let Err(crate::NoDevice) = crate::init() {
            return;
        }
        crate::Device::new(0).context().apply(|ctx| {
            let (ptx, _log) = Ptx::compile(CODE, ctx.dev().compute_capability());
            let module = ctx.load(&ptx.unwrap());
            let kernel = module.get_kernel(c"print");

            let graph = Graph::new();

            let one: usize = 1;

            let cuda_host_node_params = CUDA_HOST_NODE_PARAMS {
                fn_: Some(host_fn),
                userData: one as *mut core::ffi::c_void,
            };
            let mut node = null_mut();
            driver!(cuGraphAddHostNode(
                &mut node,
                graph.as_raw(),
                null(),
                0,
                &cuda_host_node_params,
            ));

            graph.add_kernel_call(&kernel, (1, 1, 0), &params![1].to_ptrs(), &[]);

            // graph.save_dot(std::env::current_dir().unwrap().join("host_graph.dot"));
            let stream = ctx.stream();
            let exec = ctx.instantiate(&graph);

            graph.add_host_node_with_rust_fn(
                || {
                    println!("Captured x = {}", 1);
                },
                &[],
            );
            //两者不排序不保证执行顺序
            stream.launch_graph(&exec);
        });
    }

    #[test]
    fn test_host_node_with_rust_fn() {
        if let Err(crate::NoDevice) = crate::init() {
            return;
        }
        crate::Device::new(0).context().apply(|ctx| {
            let graph = Graph::new();
            graph.add_host_node_with_rust_fn(
                || {
                    println!("hello from rust fn");
                },
                &[],
            );
            // graph.save_dot(std::env::current_dir().unwrap().join("host_graph.dot"));
            let stream = ctx.stream();
            let exec = ctx.instantiate(&graph);

            //两者不排序不保证执行顺序
            stream.launch_graph(&exec);
        });
    }
}
