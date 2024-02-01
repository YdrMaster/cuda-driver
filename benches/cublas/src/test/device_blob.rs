// use crate::{bindings::cuda, stream::Stream, AsRaw};
// use std::{alloc::Layout, ffi::c_void, ptr::null_mut};

// #[repr(transparent)]
// pub struct DeviceBlob(*mut c_void);

// impl AsRaw for DeviceBlob {
//     type Output = *mut c_void;

//     #[inline]
//     fn as_raw(&self) -> Self::Output {
//         self.0
//     }
// }

// impl Drop for DeviceBlob {
//     fn drop(&mut self) {
//         cuda!(cudaFree(self.0));
//     }
// }

// impl DeviceBlob {
//     pub fn uninit<T: Copy>(len: usize, stream: &Stream) -> Self {
//         let len = Layout::array::<T>(len).unwrap().size();
//         let stream = stream.as_raw();

//         let mut ptr = null_mut();
//         cuda!(cudaMallocAsync(&mut ptr, len, stream));
//         Self(ptr)
//     }

//     pub fn from_slice<T: Copy>(src: &[T], stream: &Stream) -> Self {
//         let len = Layout::array::<T>(src.len()).unwrap().size();
//         let stream = stream.as_raw();

//         let mut ptr = null_mut();
//         cuda!(cudaMallocAsync(&mut ptr, len, stream));
//         cuda!(cudaMemcpyAsync(
//             ptr,
//             src.as_ptr().cast(),
//             len,
//             cudaMemcpyKind::cudaMemcpyHostToDevice,
//             stream,
//         ));
//         Self(ptr)
//     }

//     pub fn copy_out<T: Copy>(&self, dst: &mut [T]) {
//         cuda!(cudaMemcpy(
//             dst.as_mut_ptr().cast(),
//             self.0,
//             Layout::array::<T>(dst.len()).unwrap().size(),
//             cudaMemcpyKind::cudaMemcpyDeviceToHost
//         ));
//     }
// }
