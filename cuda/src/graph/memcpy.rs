use super::{Graph, GraphNode, MemcpyNode};
use crate::{
    DevByte,
    bindings::{CUDA_MEMCPY3D, CUmemorytype},
};
use context_spore::AsRaw;
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
        self.add_memcpy_node_with_params(
            &CUDA_MEMCPY3D {
                srcMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                srcDevice: src.as_ptr() as _,
                dstMemoryType: CUmemorytype::CU_MEMORYTYPE_DEVICE,
                dstDevice: dst.as_mut_ptr() as _,
                WidthInBytes: size_of_val(dst),
                ..CFG
            },
            dependencies,
        )
    }

    pub fn add_memcpy_node(&self, node: &MemcpyNode, dependencies: &[GraphNode]) -> MemcpyNode {
        let mut params = MaybeUninit::uninit();
        driver!(cuGraphMemcpyNodeGetParams(
            node.as_raw(),
            params.as_mut_ptr()
        ));

        self.add_memcpy_node_with_params(unsafe { params.assume_init_ref() }, dependencies)
    }

    pub fn add_memcpy_node_with_params(
        &self,
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
            null_mut(),
        ));
        MemcpyNode(node, PhantomData)
    }
}

#[cfg(test)]
mod test {
    use crate::{AsRaw, DevByte, Device, Graph, GraphNode, VirMem, memcpy_d2h, memcpy_h2d};
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
            let mut src = ctx.malloc::<u8>(2 << 10);
            let mut dst = ctx.malloc::<u8>(2 << 10);

            let graph = Graph::new();
            graph.add_memcpy_d2d(&mut dst, &src, &[]);
            test_memcpy_in_graph(&ctx.dev(), &graph, &mut dst, &mut src, 0..u64::MAX);
        })
    }

    #[test]
    fn test_vm() {
        if let Err(crate::NoDevice) = crate::init() {
            return;
        }

        let dev = Device::new(0);
        let prop = dev.mem_prop();
        let minium = prop.granularity_minimum();

        let mut dst = VirMem::new(minium, 0).map_on(&dev);
        let src = VirMem::new(minium, 0).map_on(&dev);

        let graph = Graph::new();
        // 虚存不能直接传入 memcpy node，当时必须是已映射状态
        graph.add_memcpy_d2d(&mut dst, &src, &[]);

        let phy0 = prop.create(minium);
        let phy1 = prop.create(minium);
        let phy2 = prop.create(minium);
        let phy3 = prop.create(minium);

        let (vir_dst, _) = dst.unmap();
        let (vir_src, _) = src.unmap();
        let mut dst = vir_dst.map(phy0);
        let mut src = vir_src.map(phy1);
        test_memcpy_in_graph(&dev, &graph, &mut dst, &mut src, 0..);

        let (vir_dst, _phy0) = dst.unmap();
        let (vir_src, _phy1) = src.unmap();
        let mut dst = vir_dst.map(phy2);
        let mut src = vir_src.map(phy3);
        test_memcpy_in_graph(&dev, &graph, &mut dst, &mut src, (0..u64::MAX).rev());
    }

    fn test_memcpy_in_graph(
        dev: &Device,
        graph: &Graph,
        dst: &mut [DevByte],
        src: &mut [DevByte],
        origin: impl IntoIterator<Item = u64>,
    ) {
        assert_eq!(dst.len(), src.len());
        let origin = origin
            .into_iter()
            .take(dst.len() / size_of::<u64>())
            .collect::<Box<_>>();
        dev.context().apply(|ctx| {
            memcpy_h2d(src, &origin);

            ctx.instantiate(&graph).launch(&ctx.stream());

            let mut host = vec![0u64; origin.len()];
            memcpy_d2h(&mut host, &dst);
            assert_eq!(host, &*origin)
        })
    }
}
