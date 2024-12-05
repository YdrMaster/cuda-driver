use crate::bindings::ncclComm_t;
use cuda::{AsRaw, Device};

#[repr(transparent)]
pub struct Communicator(ncclComm_t);

unsafe impl Send for Communicator {}
unsafe impl Sync for Communicator {}

impl Communicator {
    #[inline]
    pub fn count(&self) -> usize {
        let mut count = 0;
        nccl!(ncclCommCount(self.0, &mut count));
        count as _
    }

    #[inline]
    pub fn device(&self) -> Device {
        let mut index = 0;
        nccl!(ncclCommCuDevice(self.0, &mut index));
        Device::new(index)
    }

    #[inline]
    pub fn rank(&self) -> usize {
        let mut rank = 0;
        nccl!(ncclCommUserRank(self.0, &mut rank));
        rank as _
    }

    #[inline]
    pub fn abort(&self) {
        nccl!(ncclCommAbort(self.0));
    }
}

impl From<ncclComm_t> for Communicator {
    #[inline]
    fn from(comm: ncclComm_t) -> Self {
        Self(comm)
    }
}

impl AsRaw for Communicator {
    type Raw = ncclComm_t;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Drop for Communicator {
    #[inline]
    fn drop(&mut self) {
        nccl!(ncclCommDestroy(self.0));
    }
}

#[test]
fn test_behavior() {
    use cuda::NoDevice;
    use std::{ptr::null_mut, time::Instant};

    let gpu = match cuda::init() {
        Ok(()) => cuda::Device::new(0),
        Err(NoDevice) => return,
    };

    let devlist = [unsafe { gpu.as_raw() }];
    let time = Instant::now();
    let mut comm = null_mut();
    nccl!(ncclCommInitAll(&mut comm, 1, devlist.as_ptr()));
    println!("init: {:?}", time.elapsed());
    assert!(!comm.is_null());

    let time = Instant::now();
    let mut count = 0;
    nccl!(ncclCommCount(comm, &mut count));
    println!("get count: {:?}", time.elapsed());
    assert_eq!(count, 1);

    let time = Instant::now();
    let mut device = 0;
    nccl!(ncclCommCuDevice(comm, &mut device));
    println!("get device: {:?}", time.elapsed());
    assert_eq!(device, devlist[0]);

    let time = Instant::now();
    let mut rank = 0;
    nccl!(ncclCommUserRank(comm, &mut rank));
    println!("get rank: {:?}", time.elapsed());
    assert_eq!(rank, 0);
}
