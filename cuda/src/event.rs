use crate::AsRaw;

use super::{bindings as cuda, stream::Stream};
use std::{ptr::null_mut, time::Duration};

#[repr(transparent)]
pub struct Event(cuda::CUevent);

impl Drop for Event {
    fn drop(&mut self) {
        driver!(cuEventDestroy_v2(self.0));
    }
}

impl Stream<'_> {
    pub fn record(&self) -> Event {
        let mut event = null_mut();
        driver!(cuEventCreate(
            &mut event,
            CUstream_flags::CU_STREAM_DEFAULT as _
        ));
        driver!(cuEventRecord(event, self.as_raw()));
        Event(event)
    }

    pub fn bench(&self, f: impl Fn(usize, &Self), times: usize, warm_up: usize) -> Duration {
        for i in 0..warm_up {
            f(i, self);
        }
        let start = self.record();
        for i in 0..times {
            f(i, self);
        }
        let end = self.record();
        end.synchronize();
        end.elapse_from(&start).div_f32(times as _)
    }
}

impl Event {
    pub fn synchronize(&self) {
        driver!(cuEventSynchronize(self.0));
    }

    pub fn elapse_from(&self, start: &Self) -> Duration {
        let mut ms = 0.0;
        driver!(cuEventElapsedTime(&mut ms, start.0, self.0));
        Duration::from_secs_f32(ms / 1000.0)
    }
}
