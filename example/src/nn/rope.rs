pub struct Weight<T> {
    sin: T,
    cos: T,
}

impl<T> Weight<T> {
    pub fn new(sin: T, cos: T) -> Self {
        Self { sin, cos }
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Weight<U> {
        Weight {
            sin: f(self.sin),
            cos: f(self.cos),
        }
    }

    pub fn as_ref(&self) -> Weight<&T> {
        Weight {
            sin: &self.sin,
            cos: &self.cos,
        }
    }
}
