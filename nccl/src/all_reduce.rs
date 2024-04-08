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
