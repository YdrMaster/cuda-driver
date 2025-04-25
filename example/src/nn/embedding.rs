pub struct Weight<T> {
    pub token_embd: T,
}

impl<T> Weight<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Weight<U> {
        Weight {
            token_embd: f(self.token_embd),
        }
    }

    pub fn as_ref(&self) -> Weight<&T> {
        Weight {
            token_embd: &self.token_embd,
        }
    }
}
