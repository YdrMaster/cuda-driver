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
    use cuda::ContextResource;
    use std::iter::zip;

    const N: usize = 10 << 10; // 10K * sizeof::<f32>() = 40K bytes

    cuda::init();

    let mut array = [1.0f32; N];
    let group = crate::CommunicatorGroup::new(&[0, 1]);
    let contexts = group.contexts().collect::<Vec<_>>();
    let mut streams = contexts
        .iter()
        .map(|context| context.apply(|ctx| ctx.stream().sporulate()))
        .collect::<Vec<_>>();
    let mem = group
        .call()
        .iter()
        .enumerate()
        .map(|(i, comm)| {
            contexts[i].apply(|ctx| {
                let stream = unsafe { ctx.sprout(&streams[i]) };
                let mut mem = stream.from_host(&array);
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

    for (i, (context, mut mem)) in zip(contexts, mem).enumerate() {
        context.apply(|ctx| {
            ctx.synchronize();
            cuda::memcpy_d2h(&mut array, unsafe { &*ctx.sprout(&mem) });
            assert_eq!(array, [2.; N]);

            unsafe { ctx.kill(&mut mem) };
            unsafe { ctx.kill(&mut streams[i]) };
        });
    }
}
