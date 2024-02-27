use crate::{bindings as cuda, AsRaw, Context, ContextGuard, Stream};
use std::{alloc::Layout, marker::PhantomData, ops::Range, ptr::null_mut, sync::Arc};

#[derive(Debug)]
pub struct DevBlob {
    ctx: Arc<Context>,
    ptr: cuda::CUdeviceptr,
}

impl AsRaw for DevBlob {
    type Raw = cuda::CUdeviceptr;

    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.ptr
    }
}

impl Drop for DevBlob {
    #[inline]
    fn drop(&mut self) {
        self.ctx.apply(|_| driver!(cuMemFree_v2(self.ptr)));
    }
}

impl ContextGuard<'_> {
    #[inline]
    pub fn malloc(&self, size: usize) -> DevBlob {
        let mut ptr = 0;
        driver!(cuMemAlloc_v2(&mut ptr, size));
        DevBlob {
            ctx: self.clone_ctx(),
            ptr,
        }
    }

    #[inline]
    pub fn malloc_for<T: Copy>(&self, len: usize) -> DevBlob {
        self.malloc(Layout::array::<T>(len).unwrap().size())
    }

    #[inline]
    pub fn from_slice<T: Copy>(&self, slice: &[T]) -> DevBlob {
        let ans = self.malloc_for::<T>(slice.len());
        driver!(cuMemcpyHtoD_v2(
            ans.ptr,
            slice.as_ptr().cast(),
            Layout::array::<T>(slice.len()).unwrap().size()
        ));
        ans
    }
}

impl Stream<'_> {
    #[inline]
    pub fn malloc(&self, size: usize) -> DevBlob {
        let mut ptr = 0;
        driver!(cuMemAllocAsync(&mut ptr, size, self.as_raw()));
        DevBlob {
            ctx: self.ctx().clone_ctx(),
            ptr,
        }
    }

    #[inline]
    pub fn malloc_for<T: Copy>(&self, len: usize) -> DevBlob {
        self.malloc(Layout::array::<T>(len).unwrap().size())
    }

    #[inline]
    pub fn from_slice<T: Copy>(&self, slice: &[T]) -> DevBlob {
        let ans = self.malloc_for::<T>(slice.len());
        driver!(cuMemcpyHtoDAsync_v2(
            ans.ptr,
            slice.as_ptr().cast(),
            Layout::array::<T>(slice.len()).unwrap().size(),
            self.as_raw(),
        ));
        ans
    }
}

impl DevBlob {
    #[inline]
    pub fn ctx(&self) -> &Arc<Context> {
        &self.ctx
    }
}

pub struct DevSlice<'a> {
    ptr: cuda::CUdeviceptr,
    len: usize,
    _phantom: PhantomData<&'a ()>,
}

impl AsRaw for DevSlice<'_> {
    type Raw = cuda::CUdeviceptr;

    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.ptr
    }
}

impl DevBlob {
    #[inline]
    pub fn as_slice(&self, ctx: &ContextGuard) -> DevSlice {
        assert!(Context::check_eq(ctx, &*self.ctx));
        DevSlice {
            ptr: self.ptr,
            len: {
                let mut size = 0;
                driver!(cuMemGetAddressRange_v2(null_mut(), &mut size, self.ptr));
                size
            },
            _phantom: PhantomData,
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that `len` is less than or equal to the actual length of the memory,
    /// and the context bound to this memory is currently loaded to the current thread.
    #[inline]
    pub unsafe fn as_slice_unchecked(&self, len: usize) -> DevSlice {
        DevSlice {
            ptr: self.ptr,
            len,
            _phantom: PhantomData,
        }
    }
}

impl DevSlice<'_> {
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    #[inline]
    pub fn slice(self, range: &Range<usize>) -> Self {
        assert!(range.len() <= self.len);
        Self {
            ptr: self.ptr + range.start as cuda::CUdeviceptr,
            len: range.len(),
            _phantom: self._phantom,
        }
    }

    #[inline]
    pub fn split_at(self, idx: usize) -> (Self, Self) {
        let idx = idx.min(self.len);
        (
            Self {
                ptr: self.ptr,
                len: idx,
                _phantom: self._phantom,
            },
            Self {
                ptr: self.ptr + idx as cuda::CUdeviceptr,
                len: self.len - idx,
                _phantom: self._phantom,
            },
        )
    }

    #[inline]
    pub fn copy_in<T: Copy>(&self, src: &[T]) {
        let size = Layout::array::<T>(src.len()).unwrap().size();
        assert!(size <= self.len);
        driver!(cuMemcpyHtoD_v2(self.ptr, src.as_ptr().cast(), size));
    }

    #[inline]
    pub fn copy_out<T: Copy>(&self, dst: &mut [T]) {
        let size = Layout::array::<T>(dst.len()).unwrap().size();
        assert!(size <= self.len);
        driver!(cuMemcpyDtoH_v2(dst.as_mut_ptr().cast(), self.ptr, size));
    }
}

#[test]
fn bench() {
    use std::time::Instant;

    crate::init();
    let Some(dev) = crate::Device::fetch() else {
        return;
    };

    // let d = 2048;
    // let nh = 32;
    // let nkvh = 4;
    // let dh = d / nh;
    // let di = 5632;
    // let nv = 32000;
    // let dt = 2;

    let d = 4096;
    let nh = 32;
    let nkvh = 32;
    let dh = d / nh;
    let di = 12288;
    let nv = 119696;
    let dt = 2;

    println!("model: {}", dt * d * (nv + nv + 1));
    println!("layer: {}", (((nh + nkvh) * dh + 1) * 2 + di * 3) * d);
    println!();

    let m0 = vec![1u8; dt * nv * d];
    let m1 = vec![1u8; dt * d];
    let m2 = vec![1u8; dt * nv * d];
    let m3 = vec![1u8; (((nh + nkvh) * dh + 1) * 2 + di * 3) * d];

    dev.set_mempool_threshold(u64::MAX);
    dev.context().apply(|ctx| {
        {
            let t0 = Instant::now();
            ctx.lock_page(&m0);
            ctx.lock_page(&m1);
            ctx.lock_page(&m2);
            ctx.lock_page(&m3);
            let t1 = Instant::now();
            println!("register = {:?}", t1 - t0);
        }

        let stream = ctx.stream();
        for i in 0..4 {
            let e0 = stream.record();
            let t0 = Instant::now();
            let _dev0 = stream.from_slice(&m0);
            let _dev1 = stream.from_slice(&m1);
            let _dev2 = stream.from_slice(&m2);
            let t1 = Instant::now();
            let e1 = stream.record();
            e1.synchronize();
            let t2 = Instant::now();
            println!();
            println!("model loop {i} total = {:?}", t2 - t0);
            println!(
                "host = {:?}, sync = {:?}, dev = {:?}",
                t1 - t0,
                t2 - t1,
                e1.elapse_from(&e0),
            );
        }
        for i in 0..4 {
            let e0 = stream.record();
            let t0 = Instant::now();
            let _dev3 = stream.from_slice(&m3);
            let t1 = Instant::now();
            let e1 = stream.record();
            e1.synchronize();
            let t2 = Instant::now();
            println!();
            println!("layer loop {i} total = {:?}", t2 - t0);
            println!(
                "host = {:?}, sync = {:?}, dev = {:?}",
                t1 - t0,
                t2 - t1,
                e1.elapse_from(&e0),
            );
        }
    });
}
