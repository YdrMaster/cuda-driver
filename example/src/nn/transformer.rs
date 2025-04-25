use super::{ffn, normalization, self_attn};

pub struct Weight<T> {
    pub attn_norm: normalization::Weight<T>,
    pub attn: self_attn::Weight<T>,
    pub ffn_norm: normalization::Weight<T>,
    pub ffn: ffn::Weight<T>,
}

impl<T> Weight<T> {
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Weight<U> {
        Weight {
            attn_norm: self.attn_norm.map(&mut f),
            attn: self.attn.map(&mut f),
            ffn_norm: self.ffn_norm.map(&mut f),
            ffn: self.ffn.map(&mut f),
        }
    }

    pub fn as_ref(&self) -> Weight<&T> {
        Weight {
            attn_norm: self.attn_norm.as_ref(),
            attn: self.attn.as_ref(),
            ffn_norm: self.ffn_norm.as_ref(),
            ffn: self.ffn.as_ref(),
        }
    }
}
