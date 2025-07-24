﻿use crate::{
    Device, MemSize,
    bindings::{MCcontext, MCdevice},
};
use context_spore::{AsRaw, RawContainer};
use std::{
    mem::{align_of, size_of},
    ptr::null_mut,
};

#[derive(PartialEq, Eq, Hash, Debug)]
pub struct Context {
    ctx: MCcontext,
    dev: MCdevice,
    primary: bool,
}

impl Device {
    #[inline]
    pub fn context(&self) -> Context {
        const { assert!(size_of::<Context>() == size_of::<[usize; 2]>()) }
        const { assert!(align_of::<Context>() == align_of::<usize>()) }

        let dev = unsafe { self.as_raw() };
        let mut ctx = null_mut();
        driver!(mcCtxCreate(&mut ctx, 0, dev));
        driver!(mcCtxPopCurrent(null_mut()));
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
        driver!(mcDevicePrimaryCtxRetain(&mut ctx, dev));
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
            #[cfg(nvidia)]
            {
                driver!(cuDevicePrimaryCtxRelease_v2(self.dev))
            }
            #[cfg(iluvatar)]
            {
                driver!(cuDevicePrimaryCtxRelease(self.dev))
            }
        } else {
            driver!(mcCtxDestroy(self.ctx))
        }
    }
}

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

impl AsRaw for Context {
    type Raw = MCcontext;
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
    pub fn apply<T>(&self, f: impl FnOnce(&CurrentCtx) -> T) -> T {
        driver!(mcCtxPushCurrent(self.ctx));
        let ans = f(&CurrentCtx(self.ctx));
        let mut top = null_mut();
        driver!(mcCtxPopCurrent(&mut top));
        assert_eq!(top, self.ctx);
        ans
    }
}

#[repr(transparent)]
pub struct CurrentCtx(MCcontext);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct NoCtxError;

impl AsRaw for CurrentCtx {
    type Raw = MCcontext;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0
    }
}

impl CurrentCtx {
    #[inline]
    pub fn dev(&self) -> Device {
        let mut dev = 0;
        driver!(mcCtxGetDevice(&mut dev));
        Device::new(dev)
    }

    /// 上下文同步。
    #[inline]
    pub fn synchronize(&self) {
        driver!(mcCtxSynchronize())
    }

    /// 获取驱动反馈的当前上下文可用存储空间和总存储空间。
    #[inline]
    pub fn mem_info(&self) -> (MemSize, MemSize) {
        let mut free = 0;
        let mut total = 0;
        driver!(mcMemGetInfo(&mut free, &mut total));
        (free.into(), total.into())
    }

    /// 如果存在当前上下文，在当前上下文上执行依赖上下文的操作。
    #[inline]
    pub fn apply_current<T>(f: impl FnOnce(&Self) -> T) -> Result<T, NoCtxError> {
        let mut raw = null_mut();
        driver!(mcCtxGetCurrent(&mut raw));
        if !raw.is_null() {
            Ok(f(&Self(raw)))
        } else {
            Err(NoCtxError)
        }
    }

    /// 直接指定当前上下文，并执行依赖上下文的操作。
    ///
    /// # Safety
    ///
    /// The `raw` context must be the current pushed context.
    #[inline]
    pub unsafe fn apply_current_unchecked<T>(raw: MCcontext, f: impl FnOnce(&Self) -> T) -> T {
        f(&Self(raw))
    }

    /// Designates `raw` as the current context.
    ///
    /// # Safety
    ///
    /// The `raw` context must be the current pushed context.
    /// Generally, this method only used for [`RawContainer::ctx`] with limited lifetime.
    #[inline]
    pub unsafe fn from_raw<'ctx>(raw: &MCcontext) -> &'ctx Self {
        let ptr = (&raw const *raw).cast::<Self>();
        unsafe { &*ptr }
    }

    /// Wrap a raw object in a `RawContainer`.
    ///
    /// # Safety
    ///
    /// The raw object must be created in this [`Context`].
    #[inline]
    pub unsafe fn wrap_raw<T: Unpin + 'static>(&self, rss: T) -> RawContainer<MCcontext, T> {
        RawContainer { ctx: self.0, rss }
    }
}

impl CurrentCtx {
    /// 将一段 host 存储空间注册为锁页内存，以允许从这个上下文直接访问。
    pub fn lock_page<T>(&self, slice: &[T]) {
        let ptrs = slice.as_ptr_range();
        driver!(mcHostRegister(
            ptrs.start as _,
            ptrs.end as usize - ptrs.start as usize,
            0,
        ));
    }

    /// 将一段 host 存储空间从锁页内存注销。
    pub fn unlock_page<T>(&self, slice: &[T]) {
        driver!(mcHostUnregister(slice.as_ptr() as _));
    }
}

#[test]
fn test_primary() {
    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    let dev = crate::Device::new(0);
    let mut flags = 0;
    let mut active = 0;
    driver!(mcDevicePrimaryCtxGetState(
        dev.as_raw(),
        &mut flags,
        &mut active
    ));
    assert_eq!(flags, 0);
    assert_eq!(active, 0);

    let mut pctx = null_mut();
    driver!(mcDevicePrimaryCtxRetain(&mut pctx, dev.as_raw()));
    assert!(!pctx.is_null());

    driver!(mcDevicePrimaryCtxGetState(
        dev.as_raw(),
        &mut flags,
        &mut active
    ));
    assert_eq!(flags, 0);
    assert_ne!(active, 0);

    driver!(mcCtxGetCurrent(&mut pctx));
    assert!(pctx.is_null());
}

#[test]
fn test_mem_info() {
    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    let dev = crate::Device::new(0);
    dev.set_mempool_threshold(u64::MAX);
    dev.context().apply(|ctx| {
        let (free, total) = ctx.mem_info();
        println!("mem info: {free}/{total}");
        // 从池中分配空间
        let stream = ctx.stream();
        let mem = stream.malloc::<u8>(free.0 * 3 / 4);
        let (free, total) = ctx.mem_info();
        println!("mem info: {free}/{total}");
        // 释放的存储只会回到池中，驱动不可见
        #[cfg(iluvatar)] // 天数的同步 API 不带自动同步，释放异步分配的内存会报错
        stream.synchronize();
        drop(mem);
        let (free, total) = ctx.mem_info();
        println!("mem info: {free}/{total}")
    })
}
