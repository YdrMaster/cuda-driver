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
