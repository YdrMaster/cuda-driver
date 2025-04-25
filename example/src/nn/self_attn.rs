use super::{linear, rope};

pub struct Weight<T> {
    pub qkv: linear::Weight<T>,
    pub rope: rope::Weight<T>,
    pub output: linear::Weight<T>,
}

impl<T> Weight<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Weight<U> {
        Weight {
            qkv: self.qkv.map(&mut f),
            rope: self.rope.map(&mut f),
            output: self.output.map(&mut f),
        }
    }

    pub fn as_ref(&self) -> Weight<&T> {
        Weight {
            qkv: self.qkv.as_ref(),
            rope: self.rope.as_ref(),
            output: self.output.as_ref(),
        }
    }
}
