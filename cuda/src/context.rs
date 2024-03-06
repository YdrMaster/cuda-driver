use crate::{bindings as cuda, AsRaw, Device};
use std::ptr::null_mut;

#[derive(PartialEq, Eq, Debug)]
#[repr(transparent)]
pub struct Context(cuda::CUcontext);

unsafe impl Send for Context {}
unsafe impl Sync for Context {}
impl Drop for Context {
    #[inline]
    fn drop(&mut self) {
        driver!(cuCtxDestroy_v2(self.0));
    }
}

impl Device {
    #[inline]
    pub fn context(&self) -> Context {
        let mut context = null_mut();
        driver!(cuCtxCreate_v2(&mut context, 0, self.as_raw()));
        driver!(cuCtxPopCurrent_v2(null_mut()));
        Context(context)
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
    pub const DROP: &'static str = "Context spore must be killed manually.";

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

pub struct ContextGuard<'a>(&'a Context);

impl Context {
    #[inline]
    fn push(&self) -> ContextGuard {
        driver!(cuCtxPushCurrent_v2(self.0));
        ContextGuard(self)
    }
}

impl Drop for ContextGuard<'_> {
    #[inline]
    fn drop(&mut self) {
        let mut top = null_mut();
        driver!(cuCtxPopCurrent_v2(&mut top));
        assert_eq!(top, self.0 .0)
    }
}

impl AsRaw for ContextGuard<'_> {
    type Raw = cuda::CUcontext;
    #[inline]
    unsafe fn as_raw(&self) -> Self::Raw {
        self.0 .0
    }
}

#[inline]
pub fn ctx_eq(a: &ContextGuard, b: &ContextGuard) -> bool {
    a.0 .0 == b.0 .0
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
    unsafe fn sprout<'ctx>(&'ctx self, ctx: &'ctx ContextGuard) -> Self::Resource<'ctx>;
    /// # Safety
    ///
    /// This function must be called in the same context as the one that created the resource.
    unsafe fn kill(&mut self, ctx: &ContextGuard);
    fn is_alive(&self) -> bool;
}

pub struct ResourceOwnership<'ctx>(bool, &'ctx ContextGuard<'ctx>);

#[inline(always)]
pub const fn owned<'ctx>(ctx: &'ctx ContextGuard<'ctx>) -> ResourceOwnership<'ctx> {
    ResourceOwnership(true, ctx)
}

#[inline(always)]
pub const fn not_owned<'ctx>(ctx: &'ctx ContextGuard<'ctx>) -> ResourceOwnership<'ctx> {
    ResourceOwnership(true, ctx)
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
