use super::NeuralNetwork;
use std::marker::PhantomData;

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct SwiGLU<T>(pub PhantomData<T>);

impl<T> NeuralNetwork<T> for SwiGLU<T> {}

impl<T> SwiGLU<T> {
    pub fn map<U>(self, _f: impl FnOnce(T) -> U) -> SwiGLU<U> {
        SwiGLU(PhantomData)
    }

    pub fn as_ref(&self) -> SwiGLU<&T> {
        SwiGLU(PhantomData)
    }
}
