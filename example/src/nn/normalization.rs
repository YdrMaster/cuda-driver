use super::NeuralNetwork;

pub struct RmsNorm<T> {
    pub weight: T,
}

impl<T> NeuralNetwork<T> for RmsNorm<T> {}

impl<T> RmsNorm<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> RmsNorm<U> {
        RmsNorm {
            weight: f(self.weight),
        }
    }

    pub fn as_ref(&self) -> RmsNorm<&T> {
        RmsNorm {
            weight: &self.weight,
        }
    }
}
