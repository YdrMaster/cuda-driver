use crate::{CurrentCtx, Stream, bindings::HCevent};
use context_spore::{AsRaw, impl_spore};
use std::{marker::PhantomData, ptr::null_mut, time::Duration};

impl_spore!(Event and EventSpore by (CurrentCtx, HCevent));

impl<'ctx> Stream<'ctx> {
    pub fn record(&self) -> Event<'ctx> {
        let mut event = null_mut();
        driver!(hcEventCreate(&mut event));
        driver!(hcEventRecord(event, self.as_raw()));
        Event(unsafe { self.ctx().wrap_raw(event) }, PhantomData)
    }
}

impl Drop for Event<'_> {
    #[inline]
    fn drop(&mut self) {
        driver!(hcEventDestroy(self.0.rss));
    }
}

impl AsRaw for Event<'_> {
    type Raw = HCevent;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss
    }
}

impl Stream<'_> {
    #[inline]
    pub fn wait_for(&self, event: &Event) -> &Self {
        driver!(hcStreamWaitEvent(self.as_raw(), event.0.rss, 0));
        self
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
        use crate::bindings::{hcError_t as E, hcEventQuery};
        match unsafe { hcEventQuery(self.0.rss) } {
            E::hcSuccess => true,
            E::hcErrorNotReady => false,
            err => panic!("Unexpected error: {err:?}"),
        }
    }

    #[inline]
    pub fn synchronize(&self) {
        driver!(hcEventSynchronize(self.0.rss))
    }

    #[inline]
    pub fn elapse_from(&self, start: &Self) -> Duration {
        let mut ms = 0.;
        driver!(hcEventElapsedTime(&mut ms, start.0.rss, self.0.rss));
        Duration::from_secs_f32(ms * 1e-3)
    }
}
