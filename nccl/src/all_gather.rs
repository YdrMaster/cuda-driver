use crate::{convert, Communicator};
use cuda::{AsRaw, CudaDataType, DevByte, Stream};
use std::ffi::c_void;

impl Communicator {
    #[inline]
    pub fn all_gather(
        &self,
        dst: &mut [DevByte],
        src: Option<&[DevByte]>,
        dt: CudaDataType,
        stream: &Stream,
    ) {
        let size = {
            let count = self.count();
            assert_eq!(dst.len() % count, 0);
            dst.len() / count
        };
        let recvbuff = dst.as_mut_ptr().cast::<c_void>();
        nccl!(ncclAllGather(
            if let Some(src) = src {
                assert_eq!(src.len(), size);
                src.as_ptr() as *mut c_void
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
