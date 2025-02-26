use crate::{CurrentCtx, Stream, bindings::CUevent};
use context_spore::{AsRaw, impl_spore};
use std::{marker::PhantomData, ptr::null_mut, time::Duration};

impl_spore!(Event and EventSpore by (CurrentCtx, CUevent));

impl<'ctx> Stream<'ctx> {
    pub fn record(&self) -> Event<'ctx> {
        let mut event = null_mut();
        driver!(cuEventCreate(
            &mut event,
            CUstream_flags::CU_STREAM_DEFAULT as _
        ));
        driver!(cuEventRecord(event, self.as_raw()));
        Event(unsafe { self.ctx().wrap_raw(event) }, PhantomData)
    }
}

impl Drop for Event<'_> {
    #[inline]
    fn drop(&mut self) {
        driver!(cuEventDestroy_v2(self.0.rss));
    }
}

impl AsRaw for Event<'_> {
    type Raw = CUevent;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss
    }
}

impl Stream<'_> {
    #[inline]
    pub fn wait_for(&self, event: &Event) {
        driver!(cuStreamWaitEvent(self.as_raw(), event.0.rss, 0));
    }

    pub fn bench(&self, mut f: impl FnMut(usize, &Self), times: usize, warm_up: usize) -> Duration {
        for i in 0..warm_up {
            f(i, self);
        }
        let mut time = Duration::ZERO;
        for i in 0..times {
            let start = self.record();
            f(i, self);
            let end = self.record();
            end.synchronize();
            time += end.elapse_from(&start)
        }
        time.div_f64(times as _)
    }
}

impl Event<'_> {
    #[inline]
    pub fn is_complete(&self) -> bool {
        use crate::bindings::{cuEventQuery, cudaError_enum as R};
        match unsafe { cuEventQuery(self.0.rss) } {
            R::CUDA_SUCCESS => true,
            R::CUDA_ERROR_NOT_READY => false,
            err => panic!("Unexpected error: {err:?}"),
        }
    }

    #[inline]
    pub fn synchronize(&self) {
        driver!(cuEventSynchronize(self.0.rss))
    }

    #[inline]
    pub fn elapse_from(&self, start: &Self) -> Duration {
        let mut ms = 0.0;
        driver!(cuEventElapsedTime(&mut ms, start.0.rss, self.0.rss));
        Duration::from_secs_f32(ms * 1e-3)
    }
}
