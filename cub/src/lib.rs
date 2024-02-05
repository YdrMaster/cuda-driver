#![cfg(detected_cuda)]

#[cfg(test)]
mod reduce_mean;
mod rms_normalization;

pub use rms_normalization::RmsNormalization;
