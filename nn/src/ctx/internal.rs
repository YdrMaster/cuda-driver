use super::{Context, Tensor, TensorMeta};
use crate::{Arg, Dim, OpError, Operator};
use digit_layout::DigitLayout;
use std::{collections::HashMap, ops::Range, rc::Rc};

pub(super) struct Internal<T> {
    ops: HashMap<String, Box<dyn Operator>>,
    tensors: Vec<TensorMeta>,
    weights: Vec<Weight<T>>,
    global_inputs: Vec<usize>,
    graph_nodes: Vec<Node>,
}

impl<T> Default for Internal<T> {
    fn default() -> Self {
        Self {
            ops: Default::default(),
            tensors: Default::default(),
            weights: Default::default(),
            global_inputs: Default::default(),
            graph_nodes: Default::default(),
        }
    }
}

impl<T> Internal<T> {
    pub fn tensor(&self, idx: usize) -> TensorMeta {
        self.tensors[idx].clone()
    }
}

#[allow(unused)]
struct Node {
    name: String,
    op: String,
    arg: Option<Arg>,
    inputs: Box<[usize]>,
    outputs: Range<usize>,
}

#[allow(unused)]
struct Weight<T> {
    name: String,
    idx: usize,
    item: T,
}

impl<T> Context<T> {
    pub fn register_op(&mut self, name: impl Into<String>, op: Box<dyn Operator>) {
        let mut internal = self.0.borrow_mut();

        assert!(internal.ops.insert(name.into(), op).is_none())
    }

    pub fn global_input(&self, dt: DigitLayout, shape: impl IntoIterator<Item = Dim>) -> Tensor<T> {
        let mut internal = self.0.borrow_mut();

        let idx = internal.tensors.len();
        internal.tensors.push(TensorMeta::new(dt, shape));
        internal.global_inputs.push(idx);
        self.tensor(idx)
    }

    pub fn weight(
        &self,
        name: String,
        dt: DigitLayout,
        shape: impl IntoIterator<Item = Dim>,
        item: T,
    ) -> Tensor<T> {
        let mut internal = self.0.borrow_mut();

        let idx = internal.tensors.len();
        internal.tensors.push(TensorMeta::new(dt, shape));
        internal.weights.push(Weight { name, idx, item });
        self.tensor(idx)
    }

    pub fn call<'ctx>(
        &self,
        name: String,
        op: impl AsRef<str>,
        inputs: impl IntoIterator<Item = Tensor<T>>,
        arg: Option<Arg>,
    ) -> Result<Vec<Tensor<T>>, OpError>
    where
        T: 'ctx,
    {
        let mut internal = self.0.borrow_mut();

        let op = op.as_ref();
        let Some(infer) = internal.ops.get(op) else {
            return Err(OpError::NotExist);
        };

        let inputs = inputs.into_iter().map(|t| t.idx).collect::<Box<_>>();
        let input_meta = inputs
            .iter()
            .map(|&i| internal.tensor(i))
            .collect::<Vec<_>>();

        let outputs = infer.infer(&input_meta, arg.as_ref())?;
        let start = internal.tensors.len();
        let range = start..start + outputs.len();
        internal.tensors.extend(outputs);

        internal.graph_nodes.push(Node {
            name,
            op: op.into(),
            arg,
            inputs,
            outputs: range.clone(),
        });

        Ok(range.clone().map(|idx| self.tensor(idx)).collect())
    }

    fn tensor(&self, idx: usize) -> Tensor<T> {
        Tensor {
            idx,
            ctx: Rc::downgrade(&self.0),
        }
    }
}
