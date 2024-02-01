use crate::{bindings as cuda, AsRaw, ContextGuard, Stream};
use std::alloc::Layout;

#[derive(Default, Debug)]
#[repr(transparent)]
pub struct DevicePtr(cuda::CUdeviceptr);

impl ContextGuard<'_> {
    #[inline]
    pub fn malloc(&self, size: usize) -> DevicePtr {
        let mut ptr = 0;
        driver!(cuMemAlloc_v2(&mut ptr, size));
        DevicePtr(ptr)
    }

    #[inline]
    pub fn free(&self, ptr: DevicePtr) {
        driver!(cuMemFree_v2(ptr.0));
    }
}

impl Stream<'_> {
    #[inline]
    pub fn malloc(&self, size: usize) -> DevicePtr {
        let mut ptr = 0;
        driver!(cuMemAllocAsync(&mut ptr, size, self.as_raw()));
        DevicePtr(ptr)
    }

    #[inline]
    pub fn free(&self, ptr: DevicePtr) {
        driver!(cuMemFreeAsync(ptr.0, self.as_raw()));
    }
}

impl AsRaw for DevicePtr {
    type Raw = cuda::CUdeviceptr;

    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Drop for DevicePtr {
    #[inline]
    fn drop(&mut self) {
        driver!(cuMemFree_v2(self.0));
    }
}

impl DevicePtr {
    #[inline]
    pub unsafe fn copy_in<T>(&mut self, data: &[T], _ctx: &ContextGuard) {
        driver!(cuMemcpyHtoD_v2(
            self.0,
            data.as_ptr().cast(),
            Layout::array::<T>(data.len()).unwrap().size()
        ));
    }

    #[inline]
    pub unsafe fn copy_out<T>(&self, data: &mut [T], _ctx: &ContextGuard) {
        driver!(cuMemcpyDtoH_v2(
            data.as_mut_ptr().cast(),
            self.0,
            Layout::array::<T>(data.len()).unwrap().size()
        ));
    }

    #[inline]
    pub unsafe fn copy_in_async<T>(&mut self, data: &[T], stream: &Stream) {
        driver!(cuMemcpyHtoDAsync_v2(
            self.0,
            data.as_ptr().cast(),
            Layout::array::<T>(data.len()).unwrap().size(),
            stream.as_raw(),
        ));
    }

    #[inline]
    pub unsafe fn copy_out_async<T>(&self, data: &mut [T], stream: &Stream) {
        driver!(cuMemcpyDtoHAsync_v2(
            data.as_mut_ptr().cast(),
            self.0,
            Layout::array::<T>(data.len()).unwrap().size(),
            stream.as_raw(),
        ));
    }
}
