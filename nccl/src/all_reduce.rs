use crate::{convert, Communicator, ReduceType};
use cuda::{AsRaw, CudaDataType, DevByte, Stream};

impl Communicator {
    pub fn all_reduce(
        &self,
        dst: &mut [DevByte],
        src: Option<&[DevByte]>,
        dt: CudaDataType,
        op: ReduceType,
        stream: &Stream,
    ) {
        let size = dst.len();
        let recvbuff = dst.as_mut_ptr().cast();
        nccl!(ncclAllReduce(
            if let Some(src) = src {
                assert_eq!(src.len(), size);
                src.as_ptr() as _
            } else {
                recvbuff
            },
            recvbuff,
            size / dt.size(),
            convert(dt),
            op,
            self.as_raw(),
            stream.as_raw() as _,
        ));
    }
}

#[test]
fn test() {
    use cuda::{ContextResource, ContextSpore};
    use std::iter::zip;

    const N: usize = 12 << 10; // 10K * sizeof::<f32>() = 40K bytes

    cuda::init();

    let mut array = [1.0f32; N];
    let group = crate::CommunicatorGroup::new(&[0, 1]);
    let contexts = group.contexts().collect::<Vec<_>>();
    let streams = contexts
        .iter()
        .map(|context| context.apply(|ctx| ctx.stream().sporulate()))
        .collect::<Vec<_>>();
    let mem = group
        .call()
        .iter()
        .enumerate()
        .map(|(i, comm)| {
            contexts[i].apply(|ctx| {
                let stream = streams[i].sprout_ref(ctx);

                let mut mem = ctx.malloc::<f32>(N);
                // let mut mem = stream.malloc::<f32>(N); // stream ordered memory allocation is not allowed in NCCL

                stream.memcpy_h2d(&mut mem, &array);
                comm.all_reduce(
                    &mut mem,
                    None,
                    CudaDataType::f32,
                    ReduceType::ncclSum,
                    &stream,
                );
                mem.sporulate()
            })
        })
        .collect::<Vec<_>>();

    for (context, (stream, mem)) in zip(contexts, zip(streams, mem)) {
        context.apply(|ctx| {
            ctx.synchronize();
            cuda::memcpy_d2h(&mut array, &*mem.sprout(ctx));
            assert_eq!(array, [2.; N]);
            stream.sprout(ctx);
        });
    }
}
