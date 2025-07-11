use crate::bindings::macaDataType_t;
use half::{bf16, f16};
use std::{ffi::c_void, marker::PhantomData};

#[cfg(not(iluvatar))]
type CublasComputeType = crate::bindings::mcblasComputeType_t;
#[cfg(iluvatar)]
type CublasComputeType = crate::bindings::cudaDataType;

pub trait Computation {
    fn a_type(&self) -> macaDataType_t;
    fn b_type(&self) -> macaDataType_t;
    fn c_type(&self) -> macaDataType_t;
    fn compute_type(&self) -> CublasComputeType;
    fn alpha(&self) -> &c_void;
    fn beta(&self) -> &c_void;
}

#[derive(Clone, Copy)]
pub struct GemmScheme<T, Compute> {
    alpha: Compute,
    beta: Compute,
    _phantom: PhantomData<T>,
}

impl<T, Compute> GemmScheme<T, Compute>
where
    Compute: Copy,
    Self: Computation,
{
    pub fn to_value(&self) -> ComputationValue {
        ComputationValue {
            a: self.a_type(),
            b: self.b_type(),
            c: self.c_type(),
            compute: self.compute_type(),
            data: {
                let mut scalar = [0u64; 2];
                fn copy<T: Copy>(data: T, ptr: &mut [u64]) {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            (&raw const data).cast::<u8>(),
                            ptr.as_mut_ptr().cast(),
                            size_of_val(&data),
                        )
                    }
                }
                copy(self.alpha, &mut scalar[0..]);
                copy(self.beta, &mut scalar[1..]);
                scalar
            },
        }
    }
}

macro_rules! impl_gemm_scheme {
    ($ty:ty => $f:expr) => {
        impl<T> GemmScheme<T, $ty> {
            pub fn new(alpha: f64, beta: f64) -> Self {
                Self {
                    alpha: $f(alpha),
                    beta: $f(beta),
                    _phantom: PhantomData,
                }
            }
        }
    };
}

impl_gemm_scheme!( f16 =>  f16::from_f64);
impl_gemm_scheme!(bf16 => bf16::from_f64);
impl_gemm_scheme!( f32 => |x| x as _    );
impl_gemm_scheme!( f64 => |x| x         );

macro_rules! impl_computation {
    ($data:ty => $mc_ty:ident; $compute:ty => $mcblas_ty:ident) => {
        impl Computation for GemmScheme<$data, $compute> {
            fn a_type(&self) -> macaDataType_t {
                macaDataType_t::$mc_ty
            }
            fn b_type(&self) -> macaDataType_t {
                macaDataType_t::$mc_ty
            }
            fn c_type(&self) -> macaDataType_t {
                macaDataType_t::$mc_ty
            }
            fn compute_type(&self) -> CublasComputeType {
                CublasComputeType::$mcblas_ty
            }
            fn alpha(&self) -> &c_void {
                unsafe { &*(&raw const self.alpha).cast() }
            }
            fn beta(&self) -> &c_void {
                unsafe { &*(&raw const self.beta).cast() }
            }
        }
    };
}

#[cfg(not(iluvatar))]
impl_computation!( f16 => MACA_R_16F ; f16 => MCBLAS_COMPUTE_16F);
#[cfg(not(iluvatar))]
impl_computation!( f16 => MACA_R_16F ; f32 => MCBLAS_COMPUTE_32F);
#[cfg(not(iluvatar))]
impl_computation!(bf16 => MACA_R_16BF; f32 => MCBLAS_COMPUTE_32F);
#[cfg(not(iluvatar))]
impl_computation!( f32 => MACA_R_32F ; f32 => MCBLAS_COMPUTE_32F);
#[cfg(not(iluvatar))]
impl_computation!( f64 => MACA_R_64F ; f64 => MCBLAS_COMPUTE_64F);

#[cfg(iluvatar)]
impl_computation!( f16 => CUDA_R_16F ; f16 => CUDA_R_16F);
#[cfg(iluvatar)]
impl_computation!( f16 => CUDA_R_16F ; f32 => CUDA_R_32F);
#[cfg(iluvatar)]
impl_computation!(bf16 => CUDA_R_16BF; f32 => CUDA_R_32F);
#[cfg(iluvatar)]
impl_computation!( f32 => CUDA_R_32F ; f32 => CUDA_R_32F);
#[cfg(iluvatar)]
impl_computation!( f64 => CUDA_R_64F ; f64 => CUDA_R_64F);
pub struct ComputationValue {
    a: macaDataType_t,
    b: macaDataType_t,
    c: macaDataType_t,
    compute: CublasComputeType,
    data: [u64; 2],
}

impl Computation for ComputationValue {
    fn a_type(&self) -> macaDataType_t {
        self.a
    }
    fn b_type(&self) -> macaDataType_t {
        self.b
    }
    fn c_type(&self) -> macaDataType_t {
        self.c
    }
    fn compute_type(&self) -> CublasComputeType {
        self.compute
    }
    fn alpha(&self) -> &c_void {
        unsafe { &*self.data.as_ptr().cast() }
    }
    fn beta(&self) -> &c_void {
        unsafe { &*self.data[1..].as_ptr().cast() }
    }
}
