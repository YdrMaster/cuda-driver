pub struct Weight<T> {
    ty: NormType,
    weights: Vec<T>,
}

#[derive(Clone, Copy)]
pub enum NormType {
    RmsNorm,
    LayerNorm,
}

impl<T> Weight<T> {
    pub fn rms_norm(weight: T) -> Self {
        Self {
            ty: NormType::RmsNorm,
            weights: vec![weight],
        }
    }

    pub fn layer_norm(weight: T, bias: T) -> Self {
        Self {
            ty: NormType::LayerNorm,
            weights: vec![weight, bias],
        }
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> Weight<U> {
        Weight {
            ty: self.ty,
            weights: self.weights.into_iter().map(&mut f).collect(),
        }
    }

    pub fn as_ref(&self) -> Weight<&T> {
        Weight {
            ty: self.ty,
            weights: self.weights.iter().collect(),
        }
    }
}
