use crate::bindings::ncclComm_t;

#[repr(transparent)]
pub struct Communicator(pub(crate) ncclComm_t);

impl Communicator {
    #[inline]
    pub fn abort(&self) {
        nccl!(ncclCommAbort(self.0));
    }
}

impl Drop for Communicator {
    #[inline]
    fn drop(&mut self) {
        nccl!(ncclCommDestroy(self.0));
    }
}
