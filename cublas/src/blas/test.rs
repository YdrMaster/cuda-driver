use super::{Cublas, GemmScheme};
use cuda::{DevByte, Device, Graph, GraphNode, Stream, driver};

#[test]
fn test_compute() {
    if let Err(cuda::NoDevice) = cuda::init() {
        return;
    }

    Device::new(0).context().apply(|ctx| {
        // |10 10|    |1 1 1 1|   |1 1|
        // |20 20| <- |2 2 2 2| · |2 2|
        // |30 30|    |3 3 3 3|   |3 3|
        //                        |4 4|
        let a: [f32; 12] = std::array::from_fn(|i| (i % 3 + 1) as _);
        let a = ctx.from_host(&a);
        let b: [f32; 8] = std::array::from_fn(|i| (i % 4 + 1) as _);
        let b = ctx.from_host(&b);

        let check = |stream: &Stream, result: &[DevByte]| {
            // 拷贝数据
            let mut host = ctx.malloc_host::<u8>(result.len());
            stream.memcpy_d2h(&mut host, result);
            stream.synchronize();
            // 验证结果
            let mut data = vec![0.0f32; host.len() / size_of::<f32>()];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host.as_mut_ptr(),
                    data.as_mut_ptr().cast(),
                    host.len(),
                )
            };
            assert_eq!(data, &[10., 20., 30., 10., 20., 30.]);
        };

        let mut c = ctx.from_host(&[0.0f32; 6]);
        let mut blas = Cublas::new(ctx);

        let stream = ctx.stream();
        blas.set_stream(&stream);
        unsafe {
            blas.gemm(
                3,
                2,
                4,
                GemmScheme::<f32, f32>::new(1., 1.),
                a.as_ptr(),
                false,
                3,
                b.as_ptr(),
                false,
                4,
                c.as_mut_ptr(),
                3,
            )
        }
        check(&stream, &c);
        // ----------------------------------------------------
        // 捕获计算图
        let stream = stream.capture();
        blas.set_stream(&stream);
        unsafe {
            blas.gemm(
                3,
                2,
                4,
                GemmScheme::<f32, f32>::new(1., 1.),
                a.as_ptr(),
                false,
                3,
                b.as_ptr(),
                false,
                4,
                c.as_mut_ptr(),
                3,
            )
        }
        let graph = stream.end();
        graph.save_dot(std::env::current_dir().unwrap().join("cublas.dot"));
        // 在 cuda graph 捕获期间不能释放捕获流上的 handle
        drop(blas);
        // 清空输出区域
        driver!(cuMemsetD8_v2(c.as_mut_ptr() as _, 0, c.len()));
        // 执行计算图
        let stream = ctx.stream();
        stream.launch_graph(&ctx.instantiate(&graph));
        // 测试计算正确
        check(&stream, &c);
        // ----------------------------------------------------
        // 已知：当且仅当 b 转置，生成的计算图中不是单个 kernel
        let mut nodes = graph.nodes().into_iter();
        let Some(GraphNode::Kernel(kernel)) = nodes.next() else {
            panic!("this should be a kernel node")
        };
        let None = nodes.next() else {
            panic!("this graph should have a single node")
        };
        // 构造新的计算图并导入节点
        let graph = Graph::new();
        graph.add_kernel_node(&kernel, &[]);
        // 清空输出区域
        driver!(cuMemsetD8_v2(c.as_mut_ptr() as _, 0, c.len()));
        // 执行计算图
        let stream = ctx.stream();
        stream.launch_graph(&ctx.instantiate(&graph));
        // 测试计算正确
        check(&stream, &c);
    })
}

#[test]
fn test_compute_batched() {
    if let Err(cuda::NoDevice) = cuda::init() {
        return;
    }

    Device::new(0).context().apply(|ctx| {
        // |10| |26|    |1 1 1 1|   |1| |5|
        // |20| |52| <- |2 2 2 2| · |2| |6|
        // |30| |78|    |3 3 3 3|   |3| |7|
        //                          |4| |8|
        let a: [f32; 12] = std::array::from_fn(|i| (i % 3 + 1) as _);
        let a = ctx.from_host(&a);
        let b: [f32; 8] = std::array::from_fn(|i| (i + 1) as _);
        let b = ctx.from_host(&b);

        let check = |stream: &Stream, result: &[DevByte]| {
            // 拷贝数据
            let mut host = ctx.malloc_host::<u8>(result.len());
            stream.memcpy_d2h(&mut host, result);
            stream.synchronize();
            // 验证结果
            let mut data = vec![0.0f32; host.len() / size_of::<f32>()];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host.as_mut_ptr(),
                    data.as_mut_ptr().cast(),
                    host.len(),
                )
            };
            assert_eq!(data, &[10., 20., 30., 26., 52., 78.]);
        };

        let mut c = ctx.from_host(&[0.0f32; 6]);
        let mut blas = Cublas::new(ctx);

        let stream = ctx.stream();
        blas.set_stream(&stream);
        unsafe {
            blas.gemm_batched(
                3,
                1,
                4,
                GemmScheme::<f32, f32>::new(1., 1.),
                a.as_ptr(),
                false,
                3,
                b.as_ptr(),
                false,
                4,
                c.as_mut_ptr(),
                3,
                2,
                3,
                0,
                4,
            )
        }
        check(&stream, &c);
        // ----------------------------------------------------
        // 捕获计算图
        let stream = stream.capture();
        blas.set_stream(&stream);
        unsafe {
            blas.gemm_batched(
                3,
                1,
                4,
                GemmScheme::<f32, f32>::new(1., 1.),
                a.as_ptr(),
                false,
                3,
                b.as_ptr(),
                false,
                4,
                c.as_mut_ptr(),
                3,
                2,
                3,
                0,
                4,
            )
        }
        let graph = stream.end();
        graph.save_dot(std::env::current_dir().unwrap().join("cublas.dot"));
        // 在 cuda graph 捕获期间不能释放捕获流上的 handle
        drop(blas);
        // 清空输出区域
        driver!(cuMemsetD8_v2(c.as_mut_ptr() as _, 0, c.len()));
        // 执行计算图
        let stream = ctx.stream();
        stream.launch_graph(&ctx.instantiate(&graph));
        // 测试计算正确
        check(&stream, &c);
        // ----------------------------------------------------
        // 已知：当且仅当 b 转置，生成的计算图中不是单个 kernel
        let mut nodes = graph.nodes().into_iter();
        let Some(GraphNode::Kernel(kernel)) = nodes.next() else {
            panic!("this should be a kernel node")
        };
        let None = nodes.next() else {
            panic!("this graph should have a single node")
        };
        // 构造新的计算图并导入节点
        let graph = Graph::new();
        graph.add_kernel_node(&kernel, &[]);
        // 清空输出区域
        driver!(cuMemsetD8_v2(c.as_mut_ptr() as _, 0, c.len()));
        // 执行计算图
        let stream = ctx.stream();
        stream.launch_graph(&ctx.instantiate(&graph));
        // 测试计算正确
        check(&stream, &c);
    })
}
