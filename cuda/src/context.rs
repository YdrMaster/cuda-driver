use crate::{bindings as cuda, AsRaw, Device};
use std::{ptr::null_mut, sync::Arc};

#[derive(PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Context(cuda::CUcontext);

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl Device {
    #[inline]
    pub fn context(&self) -> Arc<Context> {
        let mut context = null_mut();
        driver!(cuCtxCreate_v2(&mut context, 0, self.as_raw()));
        driver!(cuCtxPopCurrent_v2(null_mut()));
        Arc::new(Context(context))
    }
}

impl Drop for Context {
    #[inline]
    fn drop(&mut self) {
        driver!(cuCtxDestroy_v2(self.0));
    }
}

impl AsRaw for Context {
    type Raw = cuda::CUcontext;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl Context {
    #[inline]
    pub fn apply<T>(self: &Arc<Self>, f: impl FnOnce(&ContextGuard) -> T) -> T {
        f(&self.push())
    }

    #[inline]
    pub fn check_eq(
        a: &impl AsRaw<Raw = cuda::CUcontext>,
        b: &impl AsRaw<Raw = cuda::CUcontext>,
    ) -> bool {
        unsafe { a.as_raw() == b.as_raw() }
    }
}

pub struct ContextGuard<'a>(&'a Arc<Context>);

impl Context {
    #[inline]
    fn push<'a>(self: &'a Arc<Context>) -> ContextGuard<'a> {
        driver!(cuCtxPushCurrent_v2(self.0));
        ContextGuard(self)
    }
}

impl Drop for ContextGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        let mut top = null_mut();
        driver!(cuCtxPopCurrent_v2(&mut top));
        debug_assert_eq!(top, self.0 .0)
    }
}

impl AsRaw for ContextGuard<'_> {
    type Raw = cuda::CUcontext;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0 .0
    }
}

impl ContextGuard<'_> {
    #[inline]
    pub fn clone_ctx(&self) -> Arc<Context> {
        self.0.clone()
    }

    /// 将一段 host 存储空间注册为锁页内存，以允许从这个上下文直接访问。
    pub fn lock_page<T>(&self, slice: &[T]) {
        let ptrs = slice.as_ptr_range();
        driver!(cuMemHostRegister_v2(
            ptrs.start as _,
            ptrs.end as usize - ptrs.start as usize,
            0,
        ));
    }

    /// 将一段 host 存储空间从锁页内存注销。
    pub fn unlock_page<T>(&self, slice: &[T]) {
        driver!(cuMemHostUnregister(slice.as_ptr() as _));
    }

    #[inline]
    pub fn synchronize(&self) {
        driver!(cuCtxSynchronize());
    }
}
