use crate::{convert, Communicator, ReduceType};
use cuda::{AsRaw, CudaDataType, DevSlice, Stream};
use std::ffi::c_void;

impl Communicator {
    pub fn all_reduce(
        &self,
        dst: &mut DevSlice,
        src: Option<&DevSlice>,
        dt: CudaDataType,
        op: ReduceType,
        stream: &Stream,
    ) {
        let recvbuff = unsafe { dst.as_raw() as *mut c_void };
        let count = dst.len();
        let sendbuff = if let Some(src) = src {
            assert_eq!(src.len(), count);
            unsafe { src.as_raw() as _ }
        } else {
            recvbuff
        };
        nccl!(ncclAllReduce(
            sendbuff,
            recvbuff,
            count,
            convert(dt),
            op,
            self.0,
            stream.as_raw() as _,
        ));
    }
}
