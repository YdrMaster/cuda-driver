use super::NeuralNetwork;

pub struct LinearResidual<T> {
    pub weight: T,
}

impl<T> NeuralNetwork<T> for LinearResidual<T> {}

impl<T> LinearResidual<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> LinearResidual<U> {
        LinearResidual {
            weight: f(self.weight),
        }
    }

    pub fn as_ref(&self) -> LinearResidual<&T> {
        LinearResidual {
            weight: &self.weight,
        }
    }
}
