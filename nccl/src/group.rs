use crate::Communicator;
use cuda::{ContextGuard, Device};
use std::{ffi::c_int, iter::zip, ptr::null_mut};

#[repr(transparent)]
pub struct CommunicatorGroup(Vec<(c_int, Communicator)>);

impl CommunicatorGroup {
    pub fn new(devlist: &[c_int]) -> Self {
        let mut comms = vec![null_mut(); devlist.len()];
        nccl!(ncclCommInitAll(
            comms.as_mut_ptr(),
            devlist.len() as _,
            devlist.as_ptr()
        ));
        Self(
            zip(devlist, comms)
                .map(|(&dev, comm)| (dev, Communicator(comm)))
                .collect(),
        )
    }

    #[inline]
    pub fn apply(&self, mut f: impl FnMut(&Communicator, &ContextGuard)) {
        nccl!(ncclGroupStart());
        for (dev, comm) in &self.0 {
            Device::new(*dev).retain_primary().apply(|ctx| f(comm, ctx));
        }
        nccl!(ncclGroupEnd());
    }
}
