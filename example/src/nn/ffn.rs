use super::linear;

pub struct Weight<T> {
    pub up: linear::Weight<T>,
    pub down: linear::Weight<T>,
}

impl<T> Weight<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Weight<U> {
        Weight {
            up: self.up.map(&mut f),
            down: self.down.map(&mut f),
        }
    }

    pub fn as_ref(&self) -> Weight<&T> {
        Weight {
            up: self.up.as_ref(),
            down: self.down.as_ref(),
        }
    }
}
