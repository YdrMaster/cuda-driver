use super::NeuralNetwork;

pub struct Embedding<T> {
    pub token_embd: T,
}

impl<T> NeuralNetwork<T> for Embedding<T> {}

impl<T> Embedding<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Embedding<U> {
        Embedding {
            token_embd: f(self.token_embd),
        }
    }

    pub fn as_ref(&self) -> Embedding<&T> {
        Embedding {
            token_embd: &self.token_embd,
        }
    }
}
