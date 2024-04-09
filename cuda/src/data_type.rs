#[allow(non_camel_case_types)]
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
pub enum CudaDataType {
    u8,
    i8,
    u16,
    i16,
    u32,
    i32,
    u64,
    i64,
    f32,
    f64,
    #[cfg(feature = "half")]
    f16,
    #[cfg(feature = "half")]
    bf16,
}

impl CudaDataType {
    #[inline]
    pub fn size(self) -> usize {
        match self {
            Self::u8 => <u8 as CuTy>::SIZE,
            Self::i8 => <i8 as CuTy>::SIZE,
            Self::u16 => <u16 as CuTy>::SIZE,
            Self::i16 => <i16 as CuTy>::SIZE,
            Self::u32 => <u32 as CuTy>::SIZE,
            Self::i32 => <i32 as CuTy>::SIZE,
            Self::u64 => <u64 as CuTy>::SIZE,
            Self::i64 => <i64 as CuTy>::SIZE,
            Self::f32 => <f32 as CuTy>::SIZE,
            Self::f64 => <f64 as CuTy>::SIZE,
            #[cfg(feature = "half")]
            Self::f16 => <half_::f16 as CuTy>::SIZE,
            #[cfg(feature = "half")]
            Self::bf16 => <half_::bf16 as CuTy>::SIZE,
        }
    }

    #[inline]
    pub fn name(self) -> &'static str {
        match self {
            Self::u8 => <u8 as CuTy>::NAME,
            Self::i8 => <i8 as CuTy>::NAME,
            Self::u16 => <u16 as CuTy>::NAME,
            Self::i16 => <i16 as CuTy>::NAME,
            Self::u32 => <u32 as CuTy>::NAME,
            Self::i32 => <i32 as CuTy>::NAME,
            Self::u64 => <u64 as CuTy>::NAME,
            Self::i64 => <i64 as CuTy>::NAME,
            Self::f32 => <f32 as CuTy>::NAME,
            Self::f64 => <f64 as CuTy>::NAME,
            #[cfg(feature = "half")]
            Self::f16 => <half_::f16 as CuTy>::NAME,
            #[cfg(feature = "half")]
            Self::bf16 => <half_::bf16 as CuTy>::NAME,
        }
    }
}

pub trait CuTy: Sized {
    const NAME: &'static str;
    const SIZE: usize = std::mem::size_of::<Self>();
}

impl CuTy for u8 {
    const NAME: &'static str = "unsigned char";
}

impl CuTy for i8 {
    const NAME: &'static str = "char";
}

impl CuTy for u16 {
    const NAME: &'static str = "unsigned short";
}

impl CuTy for i16 {
    const NAME: &'static str = "short";
}

impl CuTy for u32 {
    const NAME: &'static str = "unsigned int";
}

impl CuTy for i32 {
    const NAME: &'static str = "int";
}

impl CuTy for u64 {
    const NAME: &'static str = "unsigned long long";
}

impl CuTy for i64 {
    const NAME: &'static str = "long long";
}

impl CuTy for f32 {
    const NAME: &'static str = "float";
}

impl CuTy for f64 {
    const NAME: &'static str = "double";
}

#[cfg(feature = "half")]
impl CuTy for half_::f16 {
    const NAME: &'static str = "half";
}

#[cfg(feature = "half")]
impl CuTy for half_::bf16 {
    const NAME: &'static str = "nv_bfloat16";
}
