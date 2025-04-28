use super::{Edge, GraphBuilder, Node, OpLib, Tensor, TensorMeta, WeightInfo};
use crate::{Arg, Dim, Graph, GraphTopo, OpError, TopoNode};
use digit_layout::DigitLayout;
use std::{cell::RefCell, collections::HashMap, ops::Range, rc::Rc, usize};

pub(super) struct GraphContext<T>(Rc<RefCell<Internal<T>>>);

impl GraphBuilder {
    pub(super) fn new_context<T>(
        &self,
        global_inputs: impl IntoIterator<Item = TensorMeta>,
    ) -> (GraphContext<T>, Vec<Tensor<T>>) {
        let tensors = global_inputs.into_iter().collect::<Vec<_>>();
        let n_inputs = tensors.len();
        let rc = Rc::new(RefCell::new(Internal {
            op_lib: self.op_lib.clone(),
            n_inputs,
            tensors,
            op_nodes: Default::default(),
            weights: Default::default(),
        }));

        let tensors = (0..n_inputs)
            .map(|idx| Tensor {
                idx,
                ctx: Rc::downgrade(&rc),
            })
            .collect();

        (GraphContext(rc), tensors)
    }
}

pub(super) struct Internal<T> {
    op_lib: Rc<OpLib>,
    op_nodes: Vec<Node_>,
    tensors: Vec<TensorMeta>,
    weights: HashMap<usize, WeightInfo<T>>,
    n_inputs: usize,
}

impl<T> Internal<T> {
    pub fn tensor(&self, idx: usize) -> TensorMeta {
        self.tensors[idx].clone()
    }

    pub fn into_graph(self, global_outputs: Vec<Tensor<T>>) -> Graph<Node, Edge<T>> {
        let Self {
            tensors,
            mut weights,
            op_nodes,
            n_inputs,
            ..
        } = self;
        let global_outputs = global_outputs
            .into_iter()
            .map(|t| t.idx)
            .collect::<Vec<_>>();
        let n_outputs = global_outputs.len();

        let mut nodes = Vec::with_capacity(op_nodes.len());
        let mut topo_nodes = Vec::with_capacity(op_nodes.len());
        let mut edges = Vec::with_capacity(tensors.len());
        let mut connections =
            Vec::with_capacity(n_outputs + op_nodes.iter().map(|n| n.inputs.len()).sum::<usize>());

        let mut edge_map = vec![usize::MAX; edges.len()];

        // 填入全图输入
        for i in 0..n_inputs {
            edge_map[i] = i;
            edges.push(Edge {
                meta: tensors[i].clone(),
                weight_info: None,
            });
        }
        // 预留全图输出的空间
        connections.extend(std::iter::repeat_n(usize::MAX, n_outputs));
        // 遍历节点
        for op in op_nodes {
            let Node_ {
                name,
                op,
                arg,
                inputs,
                outputs,
            } = op;
            // 记录输入
            let n_inputs = inputs.len();
            let n_outputs = outputs.len();
            let mut n_local = 0;
            connections.extend(inputs.into_iter().map(|i| match edge_map[i] {
                usize::MAX => {
                    // 未映射，应该是权重
                    let j = edges.len();
                    edge_map[i] = j;
                    n_local += 1;
                    edges.push(Edge {
                        meta: tensors[i].clone(),
                        weight_info: weights.remove(&i),
                    });
                    j
                }
                j => j,
            }));
            // 记录输出
            for i in outputs {
                assert_eq!(edge_map[i], usize::MAX);
                edge_map[i] = edges.len();
                edges.push(Edge {
                    meta: tensors[i].clone(),
                    weight_info: None,
                });
            }
            // 记录节点拓扑
            topo_nodes.push(TopoNode {
                n_local,
                n_inputs,
                n_outputs,
            });
            // 记录节点
            nodes.push(Node { name, op, arg })
        }
        // 回填全图输出
        for (i, j) in global_outputs.into_iter().enumerate() {
            connections[i] = edge_map[j]
        }
        Graph {
            topo: unsafe {
                GraphTopo::from_raw_parts(
                    n_inputs,
                    n_outputs,
                    connections.into(),
                    topo_nodes.into(),
                )
            },
            nodes: nodes.into(),
            edges: edges.into(),
        }
    }
}

#[allow(unused)]
struct Node_ {
    name: String,
    op: String,
    arg: Option<Arg>,
    inputs: Box<[usize]>,
    outputs: Range<usize>,
}

impl<T> GraphContext<T> {
    pub fn take(self) -> Internal<T> {
        let mut internal = self.0.borrow_mut();

        Internal {
            op_lib: internal.op_lib.clone(),
            n_inputs: internal.n_inputs,
            op_nodes: std::mem::take(&mut internal.op_nodes),
            tensors: std::mem::take(&mut internal.tensors),
            weights: std::mem::take(&mut internal.weights),
        }
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
        assert!(
            internal
                .weights
                .insert(idx, WeightInfo { name, item })
                .is_none()
        );
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
        let Some(infer) = internal.op_lib.get(op) else {
            return Err(OpError::NotExist);
        };

        let inputs = inputs.into_iter().map(|t| t.idx).collect::<Box<_>>();
        let input_meta = inputs
            .iter()
            .map(|&i| internal.tensor(i))
            .collect::<Vec<_>>();

        let outputs = infer.infer(&input_meta, arg.as_ref())?;
        let start = internal.tensors.len();
        internal.tensors.extend(outputs);
        let end = internal.tensors.len();

        internal.op_nodes.push(Node_ {
            name,
            op: op.into(),
            arg,
            inputs,
            outputs: start..end,
        });

        Ok((start..end).map(|idx| self.tensor(idx)).collect())
    }

    fn tensor(&self, idx: usize) -> Tensor<T> {
        Tensor {
            idx,
            ctx: Rc::downgrade(&self.0),
        }
    }
}
