use crate::{Blob, CurrentCtx};
use context_spore::{impl_spore, AsRaw};
use std::{
    alloc::Layout,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    os::raw::c_void,
    ptr::null_mut,
    slice::{from_raw_parts, from_raw_parts_mut},
};

impl_spore!(HostMem and HostMemSpore by (CurrentCtx, Blob<*mut c_void>));

impl CurrentCtx {
    pub fn malloc_host<T: Copy>(&self, len: usize) -> HostMem {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = null_mut();
        driver!(cuMemHostAlloc(&mut ptr, len, 0));
        HostMem(unsafe { self.wrap_raw(Blob { ptr, len }) }, PhantomData)
    }
}

impl Drop for HostMem<'_> {
    #[inline]
    fn drop(&mut self) {
        driver!(cuMemFreeHost(self.0.rss.ptr));
    }
}

impl AsRaw for HostMem<'_> {
    type Raw = *mut c_void;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.rss.ptr
    }
}

impl Deref for HostMem<'_> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { from_raw_parts(self.0.rss.ptr.cast(), self.0.rss.len) }
    }
}

impl DerefMut for HostMem<'_> {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { from_raw_parts_mut(self.0.rss.ptr.cast(), self.0.rss.len) }
    }
}

impl Deref for HostMemSpore {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { from_raw_parts(self.0.rss.ptr.cast(), self.0.rss.len) }
    }
}

impl DerefMut for HostMemSpore {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { from_raw_parts_mut(self.0.rss.ptr.cast(), self.0.rss.len) }
    }
}

#[test]
fn test_behavior() {
    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    let mut ptr = null_mut();
    crate::Device::new(0).context().apply(|_| {
        driver!(cuMemHostAlloc(&mut ptr, 128, 0));
        driver!(cuMemFreeHost(ptr));
    });
    ptr = null_mut();
    driver!(cuMemFreeHost(ptr));
}

#[test]
fn bench() {
    use rand::Rng;
    use std::time::{Duration, Instant};

    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    crate::Device::new(0).context().apply(|ctx| {
        let mut pagable = vec![0.0f32; 256 << 20];
        rand::thread_rng().fill(&mut *pagable);
        let pagable = unsafe {
            from_raw_parts(
                pagable.as_ptr().cast::<u8>() as *const u8,
                size_of_val(&*pagable),
            )
        };

        let size = pagable.len();

        let mut locked = ctx.malloc_host::<u8>(size);
        locked.copy_from_slice(&pagable);

        let stream = ctx.stream();
        let mut dev = ctx.malloc::<u8>(size);

        fn bench_memcpy(
            host: &[u8],
            dev: &mut [crate::DevByte],
            stream: &crate::Stream,
        ) -> (Duration, Duration) {
            let time = Instant::now();
            let start = stream.record();
            stream.memcpy_h2d(dev, host);
            let end = stream.record();
            let host = time.elapsed();
            end.synchronize();
            let dev = end.elapse_from(&start);
            (host, dev)
        }
        fn format_bw(gb: f32, dur: Duration) -> String {
            format!("{:.3}gb/s", gb / dur.as_secs_f32())
        }

        println!(
            "{:^10} | {:^10} | {:^10} | {:^10} | {:^10} | {:^10}",
            "sync host", "sync dev", "sync bw", "async host", "async dev", "async bw"
        );
        let gb = size as f32 / (1 << 30) as f32;
        for _ in 0..10 {
            let (sync_host, sync_dev) = bench_memcpy(&pagable, &mut dev, &stream);
            let (async_host, async_dev) = bench_memcpy(&locked, &mut dev, &stream);
            println!(
                "{:^10.3?} | {:^10.3?} | {:^10} | {:^10.3?} | {:^10.3?} | {:^10}",
                sync_host,
                sync_dev,
                format_bw(gb, sync_dev),
                async_host,
                async_dev,
                format_bw(gb, async_dev),
            );
        }
    });
}
