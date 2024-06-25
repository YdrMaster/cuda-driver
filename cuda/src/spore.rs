use crate::{bindings::CUcontext, CurrentCtx};

pub trait ContextResource<'ctx> {
    type Spore: ContextSpore<Resource<'ctx> = Self>;

    fn sporulate(self) -> Self::Spore;
}

pub trait ContextSpore: 'static + Send + Sync {
    type Resource<'ctx>: ContextResource<'ctx, Spore = Self>;

    fn sprout(self, ctx: &CurrentCtx) -> Self::Resource<'_>;
    fn sprout_ref<'ctx>(&'ctx self, ctx: &'ctx CurrentCtx) -> &Self::Resource<'_>;
    fn sprout_mut<'ctx>(&'ctx mut self, ctx: &'ctx CurrentCtx) -> &mut Self::Resource<'_>;
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

        impl<'ctx> $resource<'ctx> {
            pub fn ctx(&self) -> &$crate::CurrentCtx {
                unsafe { $crate::CurrentCtx::from_raw(&self.0.ctx) }
            }
        }

        #[repr(transparent)]
        pub struct $spore($crate::RawContainer<$kernel>);

        $crate::spore_convention!($spore);

        impl $crate::ContextSpore for $spore {
            type Resource<'ctx> = $resource<'ctx>;

            #[inline]
            fn sprout(self, ctx: &$crate::CurrentCtx) -> Self::Resource<'_> {
                assert_eq!(self.0.ctx, unsafe {
                    <$crate::CurrentCtx as $crate::AsRaw>::as_raw(ctx)
                });
                let ans = unsafe { std::mem::transmute_copy(&self.0) };
                std::mem::forget(self);
                ans
            }

            #[inline]
            fn sprout_ref<'ctx>(&'ctx self, ctx: &'ctx $crate::CurrentCtx) -> &Self::Resource<'_> {
                assert_eq!(self.0.ctx, unsafe {
                    <$crate::CurrentCtx as $crate::AsRaw>::as_raw(ctx)
                });
                unsafe { std::mem::transmute(&self.0) }
            }

            #[inline]
            fn sprout_mut<'ctx>(
                &'ctx mut self,
                ctx: &'ctx $crate::CurrentCtx,
            ) -> &mut Self::Resource<'_> {
                assert_eq!(self.0.ctx, unsafe {
                    <$crate::CurrentCtx as $crate::AsRaw>::as_raw(ctx)
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
