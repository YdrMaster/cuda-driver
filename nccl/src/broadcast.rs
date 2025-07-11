use crate::Communicator;
use cuda::{AsRaw, DevByte, Stream};
use std::ffi::{c_int, c_void};

impl Communicator {
    #[inline]
    pub fn broadcast(
        &self,
        dst: &mut [DevByte],
        src: Option<&[DevByte]>,
        root: c_int,
        stream: &Stream,
    ) {
        let size = dst.len();
        let recvbuff = dst.as_mut_ptr().cast::<c_void>();
        nccl!(mcclBroadcast(
            if let Some(src) = src {
                assert_eq!(src.len(), size);
                src.as_ptr().cast()
            } else {
                recvbuff
            },
            recvbuff,
            size,
            mcclDataType_t::mcclUint8,
            root,
            self.as_raw(),
            stream.as_raw() as _,
        ));
    }
}
