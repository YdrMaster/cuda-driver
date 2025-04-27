use crate::Dim;

use super::internal::Internal;
use digit_layout::DigitLayout;
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

/// 计算图层张量
pub struct Tensor<T> {
    pub(super) idx: usize,
    pub(super) ctx: Weak<RefCell<Internal<T>>>,
}

impl<T> Clone for Tensor<T> {
    fn clone(&self) -> Self {
        Self {
            idx: self.idx,
            ctx: self.ctx.clone(),
        }
    }
}

impl<T> Tensor<T> {
    #[inline]
    pub fn dt(&self) -> DigitLayout {
        self.meta().dt
    }

    #[inline]
    pub fn shape(&self) -> Rc<[Dim]> {
        self.meta().shape.clone()
    }

    fn meta(&self) -> TensorMeta {
        self.ctx.upgrade().unwrap().borrow().tensor(self.idx)
    }
}

#[derive(Clone)]
pub struct TensorMeta {
    pub dt: DigitLayout,
    pub shape: Rc<[Dim]>,
}

impl TensorMeta {
    pub fn new(dt: DigitLayout, shape: impl IntoIterator<Item = Dim>) -> Self {
        Self {
            dt,
            shape: shape.into_iter().collect(),
        }
    }

    #[inline]
    pub const fn dt(&self) -> DigitLayout {
        self.dt
    }

    #[inline]
    pub fn shape(&self) -> &[Dim] {
        &self.shape
    }
}
