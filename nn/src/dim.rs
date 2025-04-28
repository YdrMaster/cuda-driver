//! 简单的符号运算系统，用于将形状符号化。
//!
//! 考虑到形状运算的实际情况，只支持多项式的运算。

use std::{
    collections::{HashMap, VecDeque},
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Sub},
};

/// 形状的一个维度，或参与维度运算的值。
///
/// ```rust
/// # use std::collections::HashMap;
/// # use nn::Dim;
/// let a = Dim::var("a");
/// let b = Dim::var("b");
/// let _1 = Dim::from(1);
/// let expr = (a + _1 - 2) * 3 / (b + 1);
/// assert_eq!(expr.substitute(&HashMap::from([("a", 8), ("b", 6)])), 3);
/// ```
#[derive(Clone, Debug)]
pub enum Dim {
    /// 常量
    Constant(usize),
    /// 变量
    Variable(String),
    /// 和式
    Sum(VecDeque<Operand>),
    /// 积式
    Product(VecDeque<Operand>),
}

impl Dim {
    /// 变量。
    pub fn var(symbol: impl Display) -> Self {
        Self::Variable(symbol.to_string())
    }

    /// 维度作为正操作数。
    pub fn positive(self) -> Operand {
        Operand {
            ty: Type::Positive,
            dim: self,
        }
    }

    /// 维度作为负操作数。
    pub fn negative(self) -> Operand {
        Operand {
            ty: Type::Negative,
            dim: self,
        }
    }

    pub fn substitute(self, value: &HashMap<&str, usize>) -> usize {
        match self {
            Self::Constant(value) => value,
            Self::Variable(name) => *value.get(&*name).unwrap(),
            Self::Sum(operands) => operands.into_iter().fold(0, |acc, Operand { ty, dim }| {
                let value = dim.substitute(value);
                match ty {
                    Type::Positive => acc + value,
                    Type::Negative => acc.checked_sub(value).unwrap(),
                }
            }),
            Self::Product(operands) => operands.into_iter().fold(1, |acc, Operand { ty, dim }| {
                let value = dim.substitute(value);
                match ty {
                    Type::Positive => acc * value,
                    Type::Negative => {
                        assert_eq!(acc % value, 0);
                        acc / value
                    }
                }
            }),
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum Type {
    Positive,
    Negative,
}

impl Type {
    pub fn rev(self) -> Self {
        match self {
            Self::Positive => Self::Negative,
            Self::Negative => Self::Positive,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Operand {
    ty: Type,
    dim: Dim,
}

impl Operand {
    pub fn rev_assign(&mut self) {
        self.ty = self.ty.rev()
    }
}

impl Neg for Operand {
    type Output = Self;

    fn neg(self) -> Self::Output {
        let Self { ty, dim } = self;
        Self { ty: ty.rev(), dim }
    }
}

impl From<usize> for Dim {
    fn from(value: usize) -> Self {
        Dim::Constant(value)
    }
}

impl From<String> for Dim {
    fn from(value: String) -> Self {
        Dim::Variable(value)
    }
}

macro_rules! impl_op {
    ($op:ty; $fn:ident; positive: $variant: ident) => {
        impl $op for Dim {
            type Output = Self;
            fn $fn(self, rhs: Self) -> Self::Output {
                match self {
                    Dim::$variant(mut l) => match rhs {
                        Self::$variant(r) => {
                            l.extend(r);
                            Self::$variant(l)
                        }
                        r => {
                            l.push_back(r.positive());
                            Self::$variant(l)
                        }
                    },
                    l => match rhs {
                        Self::$variant(mut r) => {
                            r.push_front(l.positive());
                            Self::$variant(r)
                        }
                        r => Self::$variant([l.positive(), r.positive()].into()),
                    },
                }
            }
        }
    };

    ($op:ty; $fn:ident; negative: $variant: ident) => {
        impl $op for Dim {
            type Output = Self;
            fn $fn(self, rhs: Self) -> Self::Output {
                match self {
                    Dim::$variant(mut l) => match rhs {
                        Self::$variant(r) => {
                            l.extend(r.into_iter().map(Neg::neg));
                            Self::$variant(l)
                        }
                        r => {
                            l.push_back(r.negative());
                            Self::$variant(l)
                        }
                    },
                    l => match rhs {
                        Self::$variant(mut r) => {
                            r.iter_mut().for_each(Operand::rev_assign);
                            r.push_front(l.positive());
                            Self::$variant(r)
                        }
                        r => Self::$variant([l.positive(), r.negative()].into()),
                    },
                }
            }
        }
    };

    ($op:ident; $fn:ident; usize) => {
        impl $op<usize> for Dim {
            type Output = Self;
            fn $fn(self, rhs: usize) -> Self::Output {
                self.$fn(Self::Constant(rhs))
            }
        }
    };
}

impl_op!(Add; add; positive: Sum    );
impl_op!(Sub; sub; negative: Sum    );
impl_op!(Mul; mul; positive: Product);
impl_op!(Div; div; negative: Product);

impl_op!(Add; add; usize);
impl_op!(Sub; sub; usize);
impl_op!(Mul; mul; usize);
impl_op!(Div; div; usize);
