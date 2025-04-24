mod activation;
mod attention;
mod embedding;
mod ffn;
mod linear;
mod linear_residual;
mod llama;
mod normalization;
mod rope;
mod self_attn;
mod transformer;

pub use activation::SwiGLU;
pub use attention::Attention;
pub use embedding::Embedding;
pub use ffn::Ffn;
pub use linear::Linear;
pub use linear_residual::LinearResidual;
pub use llama::Llama;
pub use normalization::RmsNorm;
pub use rope::RoPE;
pub use self_attn::SelfAttn;
pub use transformer::TransformerBlk;

pub trait NeuralNetwork<T> {}
