use crate::{Communicator, ReduceType, convert};
use cuda::{AsRaw, DevByte, Stream};
use digit_layout::DigitLayout;

impl Communicator {
    pub fn all_reduce(
        &self,
        dst: &mut [DevByte],
        src: Option<&[DevByte]>,
        dt: DigitLayout,
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
                // 如果 src 不存在，原地执行
                recvbuff
            },
            recvbuff,
            size / dt.nbytes(),
            convert(dt),
            op,
            self.as_raw(),
            stream.as_raw() as _,
        ));
    }
}

#[cfg(test)]
mod test {
    use super::ReduceType;
    use crate::CommunicatorGroup;
    use cuda::{ContextResource, ContextSpore};
    use digit_layout::types::{self, F32};
    use std::iter::zip;

    const N: usize = 2 << 20;

    #[test]
    fn test() {
        let group = match cuda::init() {
            Ok(()) if cuda::Device::count() >= 2 => CommunicatorGroup::new(&[0, 1]),
            _ => return,
        };

        let mut array = vec![1.0f32; N];
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
                    let mut mem = stream.malloc::<f32>(N);
                    stream.memcpy_h2d(&mut mem, &array);
                    comm.all_reduce(&mut mem, None, F32, ReduceType::ncclSum, stream);
                    mem.sporulate()
                })
            })
            .collect::<Vec<_>>();

        for (context, (stream, mem)) in zip(contexts, zip(streams, mem)) {
            context.apply(|ctx| {
                ctx.synchronize();
                cuda::memcpy_d2h(&mut array, &mem.sprout(ctx));
                assert_eq!(array, [2.; N]);
                stream.sprout(ctx);
            })
        }
    }

    #[test]
    fn test_capture() {
        let group = match cuda::init() {
            Ok(()) if cuda::Device::count() >= 2 => CommunicatorGroup::new(&[0, 1]),
            _ => return,
        };

        let array = vec![1.0f32; N];
        std::thread::scope(|s| {
            group
                .into_vec()
                .into_iter()
                .map(|comm| {
                    let array = array.clone();
                    s.spawn(move || {
                        let device = comm.device();
                        let graph = device.retain_primary().apply(|ctx| {
                            let stream = ctx.stream();
                            let mut mem = stream.from_host(&array);
                            let stream = stream.capture();
                            comm.all_reduce(
                                &mut mem,
                                None,
                                types::F32,
                                ReduceType::ncclSum,
                                &stream,
                            );
                            stream.end()
                        });
                        graph.save_dot(
                            std::env::current_dir()
                                .unwrap()
                                .join(format!("comm{}.dot", device.index())),
                        );
                        // 捕获了 communicator 操作的 graph 必须先于 communicator 释放
                        // 如果打开 ↓ 这个释放操作会在此阻塞
                        // drop(comm)
                    })
                })
                .collect::<Vec<_>>()
                .into_iter()
                .for_each(|t| t.join().unwrap())
        });
    }
}
