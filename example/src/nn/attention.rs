use super::NeuralNetwork;
use std::marker::PhantomData;

#[repr(transparent)]
pub struct Attention<T>(pub PhantomData<T>);

impl<T> NeuralNetwork<T> for Attention<T> {}

impl<T> Attention<T> {
    pub fn map<U>(self, _f: impl FnOnce(T) -> U) -> Attention<U> {
        Attention(PhantomData)
    }

    pub fn as_ref(&self) -> Attention<&T> {
        Attention(PhantomData)
    }
}
