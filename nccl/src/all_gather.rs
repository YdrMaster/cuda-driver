use crate::{convert, Communicator};
use cuda::{AsRaw, CudaDataType, DevSlice, Stream};
use std::ffi::c_void;

impl Communicator {
    #[inline]
    pub fn all_gather(
        &self,
        dst: &mut DevSlice,
        src: Option<&DevSlice>,
        dt: CudaDataType,
        stream: &Stream,
    ) {
        let size = {
            let count = self.count();
            assert_eq!(dst.len() % count, 0);
            dst.len() / count
        };
        let recvbuff = unsafe { dst.as_raw() as *mut c_void };
        nccl!(ncclAllGather(
            if let Some(src) = src {
                assert_eq!(src.len(), size);
                unsafe { src.as_raw() as *mut c_void }
            } else {
                unsafe { recvbuff.add(self.rank() * size) }
            },
            recvbuff,
            size / dt.size(),
            convert(dt),
            self.as_raw(),
            stream.as_raw() as _,
        ));
    }
}
