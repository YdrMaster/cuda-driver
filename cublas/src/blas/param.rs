use crate::bindings::{cublasComputeType_t, cudaDataType};
use half::{bf16, f16};
use std::{ffi::c_void, marker::PhantomData};

#[derive(Clone, Copy)]
pub struct GemmScheme<T, Compute> {
    alpha: T,
    beta: T,
    _compute: PhantomData<Compute>,
}

macro_rules! impl_gemm_scheme {
    ($ty:ty => $f:expr) => {
        impl<Compute> GemmScheme<$ty, Compute> {
            pub fn new(alpha: f64, beta: f64) -> Self {
                Self {
                    alpha: $f(alpha),
                    beta: $f(beta),
                    _compute: PhantomData,
                }
            }
        }
    };
}

impl_gemm_scheme!( f16 =>  f16::from_f64);
impl_gemm_scheme!(bf16 => bf16::from_f64);
impl_gemm_scheme!( f32 => |x| x as _    );
impl_gemm_scheme!( f64 => |x| x         );

pub trait Computation {
    fn a_type(&self) -> cudaDataType;
    fn b_type(&self) -> cudaDataType;
    fn c_type(&self) -> cudaDataType;
    fn compute_type(&self) -> cublasComputeType_t;
    fn alpha(&self) -> &c_void;
    fn beta(&self) -> &c_void;
}

macro_rules! impl_computation {
    ($data:ty => $cuda_ty:ident; $compute:ty => $cublas_ty:ident) => {
        impl Computation for GemmScheme<$data, $compute> {
            fn a_type(&self) -> cudaDataType {
                cudaDataType::$cuda_ty
            }
            fn b_type(&self) -> cudaDataType {
                cudaDataType::$cuda_ty
            }
            fn c_type(&self) -> cudaDataType {
                cudaDataType::$cuda_ty
            }
            fn compute_type(&self) -> cublasComputeType_t {
                cublasComputeType_t::$cublas_ty
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

impl_computation!( f16 => CUDA_R_16F ; f16 => CUBLAS_COMPUTE_16F);
impl_computation!( f16 => CUDA_R_16F ; f32 => CUBLAS_COMPUTE_32F);
impl_computation!(bf16 => CUDA_R_16BF; f32 => CUBLAS_COMPUTE_32F);
impl_computation!( f32 => CUDA_R_32F ; f32 => CUBLAS_COMPUTE_32F);
impl_computation!( f64 => CUDA_R_64F ; f64 => CUBLAS_COMPUTE_64F);
