use crate::{bindings::CUcontext, ContextGuard};

pub trait ContextResource<'ctx> {
    type Spore: ContextSpore<Resource<'ctx> = Self>;

    fn sporulate(self) -> Self::Spore;
}

pub trait ContextSpore: 'static + Send + Sync {
    type Resource<'ctx>: ContextResource<'ctx, Spore = Self>;

    fn sprout<'ctx>(self, ctx: &'ctx ContextGuard) -> Self::Resource<'ctx>;
    fn sprout_ref<'ctx>(&'ctx self, ctx: &'ctx ContextGuard) -> &Self::Resource<'ctx>;
    fn sprout_mut<'ctx>(&'ctx mut self, ctx: &'ctx ContextGuard) -> &mut Self::Resource<'ctx>;
}

#[macro_export]
macro_rules! spore_convention {
    ($spore:ty) => {
        unsafe impl Send for $spore {}
        unsafe impl Sync for $spore {}
        impl Drop for $spore {
            #[inline]
            fn drop(&mut self) {
                unreachable!("Never drop ContextSpore");
            }
        }
    };
}

pub struct RawContainer<T> {
    pub ctx: CUcontext,
    pub raw: T,
}

#[macro_export]
macro_rules! impl_spore {
    ($resource:ident and $spore:ident by $kernel:ty) => {
        pub struct $resource<'ctx>(
            $crate::RawContainer<$kernel>,
            std::marker::PhantomData<&'ctx ()>,
        );

        #[repr(transparent)]
        pub struct $spore($crate::RawContainer<$kernel>);

        $crate::spore_convention!($spore);

        impl $crate::ContextSpore for $spore {
            type Resource<'ctx> = $resource<'ctx>;

            #[inline]
            fn sprout<'ctx>(self, ctx: &'ctx $crate::ContextGuard) -> Self::Resource<'ctx> {
                assert_eq!(self.0.ctx, unsafe {
                    <$crate::ContextGuard as $crate::AsRaw>::as_raw(ctx)
                });
                let ans = unsafe { std::mem::transmute_copy(&self.0) };
                std::mem::forget(self);
                ans
            }

            #[inline]
            fn sprout_ref<'ctx>(
                &'ctx self,
                ctx: &'ctx $crate::ContextGuard,
            ) -> &Self::Resource<'ctx> {
                assert_eq!(self.0.ctx, unsafe {
                    <$crate::ContextGuard as $crate::AsRaw>::as_raw(ctx)
                });
                unsafe { std::mem::transmute(&self.0) }
            }

            #[inline]
            fn sprout_mut<'ctx>(
                &'ctx mut self,
                ctx: &'ctx $crate::ContextGuard,
            ) -> &mut Self::Resource<'ctx> {
                assert_eq!(self.0.ctx, unsafe {
                    <$crate::ContextGuard as $crate::AsRaw>::as_raw(ctx)
                });
                unsafe { std::mem::transmute(&mut self.0) }
            }
        }

        impl<'ctx> $crate::ContextResource<'ctx> for $resource<'ctx> {
            type Spore = $spore;

            #[inline]
            fn sporulate(self) -> Self::Spore {
                let s = unsafe { std::mem::transmute_copy(&self.0) };
                std::mem::forget(self);
                $spore(s)
            }
        }
    };
}
