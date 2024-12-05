use crate::Communicator;
use cuda::Context;
use std::{ffi::c_int, ops::Deref, ptr::null_mut};

#[repr(transparent)]
pub struct CommunicatorGroup(Vec<Communicator>);

impl Deref for CommunicatorGroup {
    type Target = [Communicator];
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

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
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline]
    pub fn contexts(&self) -> impl Iterator<Item = Context> + '_ {
        self.0.iter().map(|comm| comm.device().retain_primary())
    }

    #[inline]
    pub fn call(&self) -> GroupGuard {
        nccl!(ncclGroupStart());
        GroupGuard(&self.0)
    }

    #[inline]
    pub fn into_vec(self) -> Vec<Communicator> {
        self.0
    }
}

#[repr(transparent)]
pub struct GroupGuard<'a>(&'a [Communicator]);

impl Deref for GroupGuard<'_> {
    type Target = [Communicator];
    #[inline]
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl Drop for GroupGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        nccl!(ncclGroupEnd());
    }
}
