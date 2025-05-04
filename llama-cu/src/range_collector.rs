use std::{collections::HashMap, hash::Hash, ops::Range};

pub struct RangeCollector<T> {
    calculator: RangeCalculator,
    ranges: HashMap<*const T, Range<usize>>,
    sizes: NumCollector<usize>,
}

impl<T> RangeCollector<T> {
    pub fn new(alignment: usize) -> Self {
        Self {
            calculator: RangeCalculator {
                align: alignment,
                size: 0,
            },
            ranges: Default::default(),
            sizes: Default::default(),
        }
    }

    pub fn insert(&mut self, slice: &[T]) {
        use std::collections::hash_map::Entry::{Occupied, Vacant};
        let len = size_of_val(slice);
        match self.ranges.entry(slice.as_ptr()) {
            Occupied(entry) => {
                assert_eq!(entry.get().len(), len)
            }
            Vacant(entry) => {
                entry.insert(self.calculator.push(len));
                self.sizes.insert(len)
            }
        }
    }

    #[inline]
    pub const fn size(&self) -> usize {
        self.calculator.size
    }

    #[inline]
    pub fn get(&self, ptr: *const T) -> Option<&Range<usize>> {
        self.ranges.get(&ptr)
    }

    #[inline]
    pub fn sizes(&self) -> impl Iterator<Item = (usize, usize)> {
        self.sizes.0.iter().map(|(&a, &b)| (a, b))
    }
}

struct RangeCalculator {
    align: usize,
    size: usize,
}

impl RangeCalculator {
    #[inline]
    pub fn push(&mut self, size: usize) -> Range<usize> {
        let start = self.size.div_ceil(self.align) * self.align;
        self.size = start + size;
        start..self.size
    }
}

/// 统计指定类型的参数出现的次数。
#[derive(Debug)]
#[repr(transparent)]
pub struct NumCollector<T>(HashMap<T, usize>);

impl<T> Default for NumCollector<T> {
    fn default() -> Self {
        Self(Default::default())
    }
}

impl<T: Eq + Hash> NumCollector<T> {
    pub fn insert(&mut self, t: T) {
        *self.0.entry(t).or_insert(0) += 1
    }
}
