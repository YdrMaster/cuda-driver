use super::NeuralNetwork;

pub struct RoPE<T> {
    pub sin: T,
    pub cos: T,
}

impl<T> NeuralNetwork<T> for RoPE<T> {}

impl<T> RoPE<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> RoPE<U> {
        RoPE {
            sin: f(self.sin),
            cos: f(self.cos),
        }
    }

    pub fn as_ref(&self) -> RoPE<&T> {
        RoPE {
            sin: &self.sin,
            cos: &self.cos,
        }
    }
}
