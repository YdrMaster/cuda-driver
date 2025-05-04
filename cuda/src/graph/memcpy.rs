use super::{Graph, GraphNode, MemcpyNode};
use crate::{
    AsRaw, CurrentCtx, DevByte,
    bindings::{CUDA_MEMCPY3D, CUmemorytype},
};
use std::{
    marker::PhantomData,
    mem::MaybeUninit,
    ptr::{null, null_mut},
};

const CFG: CUDA_MEMCPY3D = CUDA_MEMCPY3D {
    srcXInBytes: 0,
    srcY: 0,
    srcZ: 0,
    srcLOD: 0,
    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
    srcHost: null(),
    srcDevice: 0,
    srcArray: null_mut(),
    reserved0: null_mut(),
    srcPitch: 0,
    srcHeight: 0,
    dstXInBytes: 0,
    dstY: 0,
    dstZ: 0,
    dstLOD: 0,
    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
    dstHost: null_mut(),
    dstDevice: 0,
    dstArray: null_mut(),
    reserved1: null_mut(),
    dstPitch: 0,
    dstHeight: 0,
    WidthInBytes: 0,
    Height: 1,
    Depth: 1,
};

impl Graph {
    pub fn add_memcpy_d2d(
        &self,
        dst: &mut [DevByte],
        src: &[DevByte],
        dependencies: &[GraphNode],
    ) -> MemcpyNode {
        assert_eq!(size_of_val(dst), size_of_val(src));
        self.add_memcpy_d2d_ptr(
            dst.as_mut_ptr(),
            src.as_ptr(),
            size_of_val(dst),
            dependencies,
        )
    }

    pub fn add_memcpy_d2d_ptr(
        &self,
        dst: *mut DevByte,
        src: *const DevByte,
        len: usize,
        dependencies: &[GraphNode],
    ) -> MemcpyNode {
        CurrentCtx::apply_current(|ctx| {
            self.add_memcpy_node_with_params(
                ctx,
                &CUDA_MEMCPY3D {
                    srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    srcDevice: src as _,
                    dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                    dstDevice: dst as _,
                    WidthInBytes: len,
                    ..CFG
                },
                dependencies,
            )
        })
        .unwrap()
    }

    pub fn add_memcpy_node(
        &self,
        ctx: &CurrentCtx,
        node: &MemcpyNode,
        dependencies: &[GraphNode],
    ) -> MemcpyNode {
        let mut params = MaybeUninit::uninit();
        driver!(cuGraphMemcpyNodeGetParams(
            node.as_raw(),
            params.as_mut_ptr()
        ));

        self.add_memcpy_node_with_params(ctx, unsafe { params.assume_init_ref() }, dependencies)
    }

    pub fn add_memcpy_node_with_params(
        &self,
        ctx: &CurrentCtx,
        params: &CUDA_MEMCPY3D,
        dependencies: &[GraphNode],
    ) -> MemcpyNode {
        let dependencies = dependencies
            .iter()
            .map(|n| unsafe { n.as_raw() })
            .collect::<Box<_>>();

        let mut node = null_mut();
        driver!(cuGraphAddMemcpyNode(
            &mut node,
            self.as_raw(),
            dependencies.as_ptr(),
            dependencies.len(),
            params,
            ctx.as_raw(),
        ));
        MemcpyNode(node, PhantomData)
    }
}

#[cfg(test)]
mod test {
    use crate::{Device, Graph, GraphNode, memcpy_d2h};
    use context_spore::AsRaw;
    use std::mem::MaybeUninit;

    #[test]
    fn test_capture() {
        if let Err(crate::NoDevice) = crate::init() {
            return;
        }

        Device::new(0).context().apply(|ctx| {
            let mut dst = ctx.malloc::<u8>(1 << 10);
            let src = ctx.malloc::<u8>(1 << 10);

            let stream = ctx.stream().capture();
            stream.memcpy_d2d(&mut dst, &src);
            let graph = stream.end();
            let [GraphNode::Memcpy(node)] = &*graph.nodes() else {
                panic!()
            };
            let mut params = MaybeUninit::uninit();
            driver!(cuGraphMemcpyNodeGetParams(
                node.as_raw(),
                params.as_mut_ptr()
            ));
            let params = unsafe { params.assume_init() };
            assert_eq!(params.WidthInBytes, 1 << 10);
            println!("{params:#x?}")
        })
    }

    #[test]
    fn test_d2d() {
        if let Err(crate::NoDevice) = crate::init() {
            return;
        }

        Device::new(0).context().apply(|ctx| {
            let origin = (0..128u64).collect::<Box<_>>();
            let src = ctx.from_host(&origin);
            let mut dst = ctx.malloc::<u8>(src.len());

            let graph = Graph::new();
            graph.add_memcpy_d2d(&mut dst, &src, &[]);
            ctx.instantiate(&graph).launch(&ctx.stream());

            let mut host = vec![0u64; origin.len()];
            memcpy_d2h(&mut host, &dst);
            assert_eq!(host, &*origin)
        })
    }
}
