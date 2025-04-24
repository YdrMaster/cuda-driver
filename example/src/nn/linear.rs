use super::NeuralNetwork;

pub struct Linear<T> {
    pub weight: T,
}

impl<T> NeuralNetwork<T> for Linear<T> {}

impl<T> Linear<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Linear<U> {
        Linear {
            weight: f(self.weight),
        }
    }

    pub fn as_ref(&self) -> Linear<&T> {
        Linear {
            weight: &self.weight,
        }
    }
}
