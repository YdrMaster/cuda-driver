use crate::{bindings as cuda, AsRaw, Device};
use std::ptr::null_mut;

#[derive(PartialEq, Eq, Debug)]
pub struct Context {
    ctx: cuda::CUcontext,
    dev: cuda::CUdevice,
    primary: bool,
}

static_assertions::assert_eq_size!(Context, [usize; 2]);
static_assertions::assert_eq_align!(Context, usize);

unsafe impl Send for Context {}
unsafe impl Sync for Context {}

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

impl Device {
    #[inline]
    pub fn context(&self) -> Context {
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

impl AsRaw for Context {
    type Raw = cuda::CUcontext;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.ctx
    }
}

impl Context {
    pub const DROP: &'static str = "Context spore must be killed manually.";

    #[inline]
    pub fn device(&self) -> Device {
        Device::new(self.dev)
    }

    #[inline]
    pub fn apply<T>(&self, f: impl FnOnce(&ContextGuard) -> T) -> T {
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

#[repr(transparent)]
pub struct ContextGuard<'a>(&'a Context);

impl Context {
    #[inline]
    fn push(&self) -> ContextGuard {
        driver!(cuCtxPushCurrent_v2(self.ctx));
        ContextGuard(self)
    }
}

impl Drop for ContextGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        let mut top = null_mut();
        driver!(cuCtxPopCurrent_v2(&mut top));
        assert_eq!(top, self.0.ctx)
    }
}

impl AsRaw for ContextGuard<'_> {
    type Raw = cuda::CUcontext;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0.ctx
    }
}

#[inline]
pub fn ctx_eq(a: &ContextGuard, b: &ContextGuard) -> bool {
    a.0.ctx == b.0.ctx
}

impl ContextGuard<'_> {
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
}

pub trait ContextResource<'ctx> {
    type Spore: ContextSpore<Resource<'ctx> = Self>;

    fn sporulate(self) -> Self::Spore;
}

pub trait ContextSpore: 'static + Send + Sync {
    type Resource<'ctx>: ContextResource<'ctx, Spore = Self>;

    /// # Safety
    ///
    /// This function must be called in the same context as the one that created the resource.
    unsafe fn sprout<'ctx>(&self, ctx: &'ctx ContextGuard) -> Self::Resource<'ctx>;
    /// # Safety
    ///
    /// This function must be called in the same context as the one that created the resource.
    unsafe fn kill(&mut self, ctx: &ContextGuard);
    fn is_alive(&self) -> bool;
}

pub struct ResourceOwnership<'ctx>(bool, &'ctx ContextGuard<'ctx>);

#[inline(always)]
pub const fn owned<'ctx>(ctx: &'ctx ContextGuard) -> ResourceOwnership<'ctx> {
    ResourceOwnership(true, ctx)
}

#[inline(always)]
pub const fn not_owned<'ctx>(ctx: &'ctx ContextGuard) -> ResourceOwnership<'ctx> {
    ResourceOwnership(false, ctx)
}

impl<'ctx> ResourceOwnership<'ctx> {
    #[inline]
    pub const fn is_owned(&self) -> bool {
        self.0
    }

    #[inline]
    pub const fn ctx(&self) -> &'ctx ContextGuard<'ctx> {
        self.1
    }
}

#[macro_export]
macro_rules! spore_convention {
    ($spore:ty) => {
        unsafe impl Send for $spore {}
        unsafe impl Sync for $spore {}
        impl Drop for $spore {
            #[inline]
            fn drop(&mut self) {
                if self.is_alive() {
                    unreachable!("Context spore must be killed manually.");
                }
            }
        }
    };
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
