use std::{
    alloc::{Layout, alloc, dealloc},
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

pub enum Data<'a> {
    Mapped(&'a [u8]),
    Owned(Blob),
}

impl<'a> From<&'a [u8]> for Data<'a> {
    fn from(value: &'a [u8]) -> Self {
        Self::Mapped(value)
    }
}

impl From<Blob> for Data<'_> {
    fn from(value: Blob) -> Self {
        Self::Owned(value)
    }
}

impl Deref for Data<'_> {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        match self {
            Self::Mapped(slice) => slice,
            Self::Owned(blob) => blob,
        }
    }
}

pub struct Blob {
    ptr: NonNull<u8>,
    len: usize,
}

impl Blob {
    pub fn new(len: usize) -> Self {
        Self {
            ptr: match len {
                0 => NonNull::dangling(),
                _ => NonNull::new(unsafe { alloc(layout(len)) }).unwrap(),
            },
            len,
        }
    }
}

impl Drop for Blob {
    fn drop(&mut self) {
        match self.len {
            0 => {}
            len => unsafe { dealloc(self.ptr.as_ptr(), layout(len)) },
        }
    }
}

impl Clone for Blob {
    fn clone(&self) -> Self {
        let mut ans = Self::new(self.len);
        ans.copy_from_slice(self);
        ans
    }
}

impl Deref for Blob {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &Self::Target {
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for Blob {
    #[inline]
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

#[inline]
fn layout(len: usize) -> Layout {
    Layout::from_size_align(len, align_of::<usize>()).unwrap()
}
