use crate::blob::Blob;
use cuda::{CurrentCtx, DevByte, VirByte, memcpy_d2h};
use ggus::ggml_quants::f16;
use nn::Tensor;
use std::fmt;
use tensor::digit_layout::types;

pub fn fmt<const N: usize>(tensor: &Tensor<*const VirByte, N>, _ctx: &CurrentCtx) {
    let mem_range = tensor.layout().data_range();
    let ptr = tensor.get().cast::<DevByte>();
    let len = *mem_range.end() as usize + tensor.dt().nbytes();
    let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
    let mut host = Blob::new(len);
    memcpy_d2h(&mut host, &slice);
    println!("{}", Fmt(tensor.as_ref().map(|_| host).as_deref()))
}

pub struct Fmt<'a, const N: usize>(Tensor<&'a [u8], N>);

impl<const N: usize> fmt::Display for Fmt<'_, N> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let layout = self.0.layout();
        let ptr = self.0.get().as_ptr();
        macro_rules! display {
            ($ty:ty) => {
                unsafe { layout.write_array(f, ptr.cast::<DataFmt<$ty>>()) }
            };
        }
        match self.0.dt() {
            types::F16 => display!(f16),
            types::F32 => display!(f32),
            types::U32 => display!(u32),
            types::U64 => display!(u64),
            _ => todo!(),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
struct DataFmt<T>(T);

impl fmt::Display for DataFmt<f16> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == f16::ZERO {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self.0.to_f32())
        }
    }
}

impl fmt::Display for DataFmt<f32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0. {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self.0)
        }
    }
}

impl fmt::Display for DataFmt<u32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            write!(f, " ________")
        } else {
            write!(f, "{:>6}", self.0)
        }
    }
}

impl fmt::Display for DataFmt<u64> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            write!(f, " ________")
        } else {
            write!(f, "{:>6}", self.0)
        }
    }
}
