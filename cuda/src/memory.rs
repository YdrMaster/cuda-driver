use crate::{bindings as cuda, AsRaw, ContextGuard, Stream};
use std::{alloc::Layout, mem::size_of_val};

#[derive(Clone)]
pub struct LocalDevBlob<'a> {
    ptr: cuda::CUdeviceptr,
    len: usize,
    _ctx: &'a ContextGuard<'a>,
}

impl<'a> Stream<'a> {
    pub fn malloc<T: Copy>(&'a self, len: usize) -> LocalDevBlob<'a> {
        let len = Layout::array::<T>(len).unwrap().size();
        let mut ptr = 0;
        driver!(cuMemAllocAsync(&mut ptr, len, self.as_raw()));
        LocalDevBlob {
            ptr,
            len,
            _ctx: self.ctx(),
        }
    }

    pub fn from_host<T: Copy>(&'a self, slice: &[T]) -> LocalDevBlob<'a> {
        let stream = unsafe { self.as_raw() };
        let len = size_of_val(slice);
        let src = slice.as_ptr().cast();
        let mut ptr = 0;
        driver!(cuMemAllocAsync(&mut ptr, len, stream));
        driver!(cuMemcpyHtoDAsync_v2(ptr, src, len, stream));
        LocalDevBlob {
            ptr,
            len,
            _ctx: self.ctx(),
        }
    }
}

impl LocalDevBlob<'_> {
    #[inline]
    pub const fn len(&self) -> usize {
        self.len
    }

    pub fn copy_in_async<T: Copy>(&mut self, slice: &[T], stream: &Stream) {
        let len = size_of_val(slice);
        let src = slice.as_ptr().cast();
        assert_eq!(len, self.len);
        driver!(cuMemcpyHtoDAsync_v2(self.ptr, src, len, stream.as_raw()));
    }

    pub fn copy_out<T: Copy>(&self, slice: &mut [T]) {
        let len = size_of_val(slice);
        let dst = slice.as_mut_ptr().cast();
        assert_eq!(len, self.len);
        driver!(cuMemcpyDtoH_v2(dst, self.ptr, len));
    }
}

impl AsRaw for LocalDevBlob<'_> {
    type Raw = cuda::CUdeviceptr;

    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.ptr
    }
}
