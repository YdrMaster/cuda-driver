pub struct Weight<T> {
    weight: T,
    bias: Option<T>,
}

impl<T> Weight<T> {
    pub fn new(weight: T, bias: Option<T>) -> Self {
        Self { weight, bias }
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Weight<U> {
        Weight {
            weight: f(self.weight),
            bias: self.bias.map(f),
        }
    }

    pub fn as_ref(&self) -> Weight<&T> {
        Weight {
            weight: &self.weight,
            bias: self.bias.as_ref(),
        }
    }
}
