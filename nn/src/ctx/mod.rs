mod internal;
mod nn_ctx;
mod tensor;

use internal::Internal;
use std::{cell::RefCell, rc::Rc};

pub use nn_ctx::NNCtx;
pub use tensor::{Tensor, TensorMeta};

#[repr(transparent)]
pub struct Context<T>(Rc<RefCell<Internal<T>>>);

impl<T> Default for Context<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}
