use crate::{
    bindings as cuda, context::ResourceOwnership, not_owned, owned, spore_convention, AsRaw,
    ContextGuard, ContextResource, ContextSpore, Stream,
};
use std::{
    mem::{forget, replace},
    ptr::null_mut,
    time::Duration,
};

pub struct Event<'ctx>(cuda::CUevent, ResourceOwnership<'ctx>);

impl<'ctx> Stream<'ctx> {
    pub fn record(&self) -> Event<'ctx> {
        let mut event = null_mut();
        driver!(cuEventCreate(
            &mut event,
            CUstream_flags::CU_STREAM_DEFAULT as _
        ));
        driver!(cuEventRecord(event, self.as_raw()));
        Event(event, owned(self.ctx()))
    }
}

impl Drop for Event<'_> {
    #[inline]
    fn drop(&mut self) {
        if self.1.is_owned() {
            driver!(cuEventDestroy_v2(self.0));
        }
    }
}

impl AsRaw for Event<'_> {
    type Raw = cuda::CUevent;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Stream<'_> {
    #[inline]
    pub fn wait_for(&self, event: &Event) {
        driver!(cuStreamWaitEvent(self.as_raw(), event.0, 0));
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

impl Event<'_> {
    #[inline]
    pub fn synchronize(&self) {
        driver!(cuEventSynchronize(self.0));
    }

    #[inline]
    pub fn elapse_from(&self, start: &Self) -> Duration {
        let mut ms = 0.0;
        driver!(cuEventElapsedTime(&mut ms, start.0, self.0));
        Duration::from_secs_f32(ms / 1000.0)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub struct EventSpore(cuda::CUevent);

spore_convention!(EventSpore);

impl ContextSpore for EventSpore {
    type Resource<'ctx> = Event<'ctx>;

    #[inline]
    unsafe fn sprout<'ctx>(&self, ctx: &'ctx ContextGuard) -> Self::Resource<'ctx> {
        Event(self.0, not_owned(ctx))
    }

    #[inline]
    unsafe fn kill(&mut self, ctx: &ContextGuard) {
        drop(Event(replace(&mut self.0, null_mut()), owned(ctx)));
    }

    #[inline]
    fn is_alive(&self) -> bool {
        !self.0.is_null()
    }
}

impl<'ctx> ContextResource<'ctx> for Event<'ctx> {
    type Spore = EventSpore;

    #[inline]
    fn sporulate(self) -> Self::Spore {
        let e = self.0;
        forget(self);
        EventSpore(e)
    }
}
