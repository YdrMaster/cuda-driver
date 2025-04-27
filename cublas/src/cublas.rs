use crate::bindings::cublasHandle_t;
use cuda::{AsRaw, CurrentCtx, Stream, impl_spore};
use std::{marker::PhantomData, ptr::null_mut};

impl_spore!(Cublas and CublasSpore by (CurrentCtx, cublasHandle_t));

impl Drop for Cublas<'_> {
    #[inline]
    fn drop(&mut self) {
        cublas!(cublasDestroy_v2(self.0.rss))
    }
}

impl AsRaw for Cublas<'_> {
    type Raw = cublasHandle_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss
    }
}

impl Cublas<'_> {
    #[inline]
    pub fn new(ctx: &CurrentCtx) -> Self {
        let mut handle = null_mut();
        cublas!(cublasCreate_v2(&mut handle));
        Self(unsafe { ctx.wrap_raw(handle) }, PhantomData)
    }

    #[inline]
    pub fn bind(stream: &Stream) -> Self {
        let mut ans = Self::new(stream.ctx());
        ans.set_stream(stream);
        ans
    }

    #[inline]
    pub fn set_stream(&mut self, stream: &Stream) {
        cublas!(cublasSetStream_v2(self.0.rss, stream.as_raw().cast()));
    }
}

#[test]
fn test_behavior() {
    use cuda::{DevByte, Device, Graph, GraphNode, driver};

    if !cuda::init().is_ok() {
        return;
    }

    Device::new(0).context().apply(|ctx| {
        // [3, 2] <- [3, 4] x [4, 2]
        let a: [f32; 12] = std::array::from_fn(|i| (i % 3 + 1) as _);
        let a = ctx.from_host(&a);
        let b: [f32; 8] = std::array::from_fn(|i| (i % 4 + 1) as _);
        let b = ctx.from_host(&b);

        let alpha = 1.0f32;
        let beta = 1.0f32;

        let check = |graph: &Graph, result: &mut [DevByte]| {
            // 清空输出区域
            driver!(cuMemsetD8_v2(result.as_mut_ptr() as _, 0, result.len()));
            // 执行计算图
            let stream = ctx.stream();
            ctx.instantiate(&graph).launch(&stream);
            // 拷贝数据
            let mut host = ctx.malloc_host::<u8>(result.len());
            stream.memcpy_d2h(&mut host, &result);
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

        let stream = ctx.stream();
        let mut blas = Cublas::new(ctx);
        let mut c = ctx.from_host(&[0.0f32; 6]);
        // 捕获计算图
        let stream = stream.capture();
        blas.set_stream(&stream);
        cublas!(cublasGemmEx(
            blas.as_raw(),
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_N,
            3,
            2,
            4,
            (&raw const alpha).cast(),
            a.as_ptr().cast(),
            cudaDataType::CUDA_R_32F,
            3,
            b.as_ptr().cast(),
            cudaDataType::CUDA_R_32F,
            4,
            (&raw const beta).cast(),
            c.as_mut_ptr().cast(),
            cudaDataType::CUDA_R_32F,
            3,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
        ));
        let graph = stream.end();
        // 在 cuda graph 捕获期间不能释放捕获流上的 handle
        drop(blas);
        // 测试计算正确
        check(&graph, &mut c);
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
        // 测试计算正确
        check(&graph, &mut c)
    })
}
