use crate::{
    bindings::{CUcontext, CUdevice},
    AsRaw, Device, RawContainer,
};
use std::{
    marker::PhantomData,
    mem::{align_of, size_of},
    ptr::null_mut,
};

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct Context {
    ctx: CUcontext,
    dev: CUdevice,
    primary: bool,
}

impl Device {
    #[inline]
    pub fn context(&self) -> Context {
        const { assert!(size_of::<Context>() == size_of::<[usize; 2]>()) }
        const { assert!(align_of::<Context>() == align_of::<usize>()) }

        let dev = unsafe { self.as_raw() };
        let mut ctx = null_mut();
        driver!(cuCtxCreate_v2(&mut ctx, 0, dev));
        driver!(cuCtxPopCurrent_v2(null_mut()));
        Context {
            ctx,
            dev,
            primary: false,
        }
    }

    #[inline]
    pub fn retain_primary(&self) -> Context {
        let dev = unsafe { self.as_raw() };
        let mut ctx = null_mut();
        driver!(cuDevicePrimaryCtxRetain(&mut ctx, dev));
        Context {
            ctx,
            dev,
            primary: true,
        }
    }
}

impl Drop for Context {
    #[inline]
    fn drop(&mut self) {
        if self.primary {
            driver!(cuDevicePrimaryCtxRelease_v2(self.dev));
        } else {
            driver!(cuCtxDestroy_v2(self.ctx));
        }
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl AsRaw for Context {
    type Raw = CUcontext;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.ctx
    }
}

impl Context {
    #[inline]
    pub fn device(&self) -> Device {
        Device::new(self.dev)
    }

    #[inline]
    pub fn apply<T>(&self, f: impl FnOnce(&ContextGuard) -> T) -> T {
        f(&self.push())
    }
}

#[repr(transparent)]
pub struct ContextGuard<'a>(CUcontext, PhantomData<&'a ()>);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NoCtxError;

impl Context {
    #[inline]
    fn push(&self) -> ContextGuard {
        driver!(cuCtxPushCurrent_v2(self.ctx));
        ContextGuard(self.ctx, PhantomData)
    }
}

impl Drop for ContextGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        let mut top = null_mut();
        driver!(cuCtxPopCurrent_v2(&mut top));
        assert_eq!(top, self.0)
    }
}

impl AsRaw for ContextGuard<'_> {
    type Raw = CUcontext;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl ContextGuard<'_> {
    /// 如果存在当前上下文，在当前上下文上执行依赖上下文的操作。
    #[inline]
    pub fn apply_current<T>(f: impl FnOnce(&Self) -> T) -> Result<T, NoCtxError> {
        let mut raw = null_mut();
        driver!(cuCtxGetCurrent(&mut raw));
        if !raw.is_null() {
            Ok(f(&Self(raw, PhantomData)))
        } else {
            Err(NoCtxError)
        }
    }

    #[inline]
    pub fn dev(&self) -> Device {
        let mut dev = 0;
        driver!(cuCtxGetDevice(&mut dev));
        Device::new(dev)
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

    /// Wrap a raw object in a `RawContainer`.
    ///
    /// # Safety
    ///
    /// The raw object must be created in this [`Context`].
    #[inline]
    pub unsafe fn wrap_raw<T>(&self, raw: T) -> RawContainer<T> {
        RawContainer { ctx: self.0, raw }
    }
}

#[test]
fn test_primary() {
    crate::init();
    let Some(dev) = crate::Device::fetch() else {
        return;
    };
    let mut flags = 0;
    let mut active = 0;
    driver!(cuDevicePrimaryCtxGetState(
        dev.as_raw(),
        &mut flags,
        &mut active
    ));
    assert_eq!(flags, 0);
    assert_eq!(active, 0);

    let mut pctx = null_mut();
    driver!(cuDevicePrimaryCtxRetain(&mut pctx, dev.as_raw()));
    assert!(!pctx.is_null());

    driver!(cuDevicePrimaryCtxGetState(
        dev.as_raw(),
        &mut flags,
        &mut active
    ));
    assert_eq!(flags, 0);
    assert_ne!(active, 0);

    driver!(cuCtxGetCurrent(&mut pctx));
    assert!(pctx.is_null());
}
