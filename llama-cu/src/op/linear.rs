use super::{Modules, Operator};
use cuda::{Graph, GraphNode, VirByte};
use tensor::Tensor;

pub struct Linear;

impl Operator for Linear {
    fn add_to_graph<'a, const N: usize>(
        _graph: &'a Graph,
        _deps: &[GraphNode<'a>],
        _modules: &mut Modules,
        _arg: Option<nn::Arg>,
        _inputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
        _outputs: impl IntoIterator<Item = Tensor<*const VirByte, N>>,
    ) -> Vec<GraphNode<'a>> {
        todo!()
    }
}
