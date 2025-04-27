use std::{
    collections::VecDeque,
    fmt::Display,
    ops::{Add, Div, Mul, Neg, Sub},
};

#[derive(Clone)]
pub enum Dim {
    Const(usize),
    Variable(String),
    Sum(VecDeque<Operand>),
    Product(VecDeque<Operand>),
}

impl Dim {
    pub fn var(symbol: impl Display) -> Self {
        Self::Variable(symbol.to_string())
    }

    pub fn positive(self) -> Operand {
        Operand {
            ty: Type::Positive,
            dim: self,
        }
    }

    pub fn negative(self) -> Operand {
        Operand {
            ty: Type::Negative,
            dim: self,
        }
    }
}

#[derive(Clone, Copy)]
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

#[derive(Clone)]
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
        Dim::Const(value)
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
                self.$fn(Self::Const(rhs))
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
