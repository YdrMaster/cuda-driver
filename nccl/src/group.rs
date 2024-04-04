use crate::Communicator;
use cuda::{Context, ContextGuard};
use std::{ffi::c_int, ptr::null_mut};

#[repr(transparent)]
pub struct CommunicatorGroup(Vec<Communicator>);

impl CommunicatorGroup {
    pub fn new(devlist: &[c_int]) -> Self {
        let mut comms = vec![null_mut(); devlist.len()];
        nccl!(ncclCommInitAll(
            comms.as_mut_ptr(),
            devlist.len() as _,
            devlist.as_ptr()
        ));
        Self(comms.into_iter().map(From::from).collect())
    }

    #[inline]
    pub fn context_iter(&self) -> impl Iterator<Item = Context> + '_ {
        self.0.iter().map(|comm| comm.device().retain_primary())
    }

    #[inline]
    pub fn apply(&self, mut f: impl FnMut(&Communicator, &ContextGuard)) {
        nccl!(ncclGroupStart());
        for comm in &self.0 {
            comm.device().retain_primary().apply(|ctx| f(comm, ctx));
        }
        nccl!(ncclGroupEnd());
    }
}
