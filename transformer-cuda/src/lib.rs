#![cfg(detected_cuda)]

mod fused_softmax;
mod rms_normalization;

pub use fused_softmax::FusedSoftmax;
pub use rms_normalization::RmsNormalization;
