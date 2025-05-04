mod blob;
mod gguf;
mod llama;
mod loader;
mod op;
mod range_collector;

use cuda::{Device, VirByte, VirMem};
use gguf::{GGufModel, map_files};
use ggus::ggml_quants::digit_layout::types;
use loader::WeightLoader;
use nn::{Dim, Exec, GraphBuilder, Node, TensorMeta, op as nn_op};
use op::Operator;
use range_collector::RangeCollector;
use std::{fmt, time::Instant};

// cargo run --release -- ../TinyStory-5M-v0.0-F32.gguf
fn main() {
    let mut times = TimeCollector::default();
    let maps = map_files(std::env::args().nth(1).unwrap());
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    let llama = llama::init(&mut gguf);
    times.push("load");

    let nn::Graph(graph::Graph { topo, nodes, edges }) = GraphBuilder::default()
        .register_op("embedding", nn_op::embedding::Embedding)
        .register_op("rms-norm", nn_op::normalization::RmsNorm)
        .register_op("layer-norm", nn_op::normalization::LayerNorm)
        .register_op("attention", nn_op::attention::Attention)
        .register_op("split", nn_op::split::Split)
        .register_op("swiglu", nn_op::activation::SwiGLU)
        .register_op("gelu", nn_op::activation::GeLU)
        .register_op("linear", nn_op::linear::Linear)
        .register_op("rope", nn_op::rope::Rope)
        .register_op("concat", nn_op::concat::Concat)
        .build(
            llama,
            [
                TensorMeta::new(types::U32, [Dim::var("n")]),
                TensorMeta::new(types::U32, [Dim::var("n")]),
            ],
        )
        .unwrap();
    times.push("build");

    // for cuda
    let mut ranges = RangeCollector::new(512);
    let edges = edges
        .into_iter()
        .map(|nn::Edge { meta, external }| nn::Edge {
            meta,
            external: external.map(|nn::External { name, item }| nn::External {
                name,
                item: {
                    let ans = gguf.tensors[&*item].as_ref();
                    ranges.insert(ans.get());
                    ans
                },
            }),
        })
        .collect::<Box<_>>();

    assert!(cuda::init().is_ok());
    let dev = Device::new(0);
    let minimum = dev.mem_prop().granularity_minimum();
    let (edges, weight_vir, weight_phy) = dev.context().apply(|ctx| {
        let mut loader = WeightLoader::new(
            ranges
                .sizes()
                .filter(|&(_, times)| times < 4)
                .map(|(size, _)| size),
        );

        let stream = ctx.stream();
        let mut mapped = VirMem::new(ranges.size().div_ceil(minimum) * minimum, 0).map_on(&dev);
        let edges = edges
            .into_iter()
            .map(|nn::Edge { meta, external }| nn::Edge {
                meta,
                external: external.map(|nn::External { name, item }| nn::External {
                    name,
                    item: item.map(|data| {
                        let range = ranges.get(data.as_ptr()).unwrap().clone();
                        let dst = &mut mapped[range];
                        loader.load(dst, data, &stream);
                        dst.as_ptr().cast::<VirByte>()
                    }),
                }),
            })
            .collect::<Box<_>>();
        let (weights, physical) = mapped.unmap();
        (edges, weights, physical)
    });
    times.push("cuda");

    let graph = nn::Graph(graph::Graph { topo, nodes, edges }).lower(&[("n", 5)].into(), |t| t);
    times.push("fix shape");

    let mem_range_map = graph.mem_range_map(8 << 30, 512);

    let n_pages = mem_range_map.range.len().div_ceil(minimum);
    let workspace_vir = (0..n_pages)
        .map(|_| VirMem::new(minimum, 0))
        .collect::<Box<_>>();
    assert!(
        (1..n_pages)
            .all(|i| workspace_vir[i].as_ptr_range().start
                == workspace_vir[i - 1].as_ptr_range().end)
    );
    let ptr = workspace_vir[0].as_ptr();
    let exec = graph
        .lower(
            |key| unsafe { ptr.byte_add(mem_range_map.map[&key].start) },
            |&data| data,
        )
        .into_exec();
    times.push("into exec");

    println!("{times}");

    // let mut graph = cuda::Graph::new();
    // let mut computation = Vec::new();
    dev.context().apply(|ctx| {
        let mut modules = op::Modules::new(ctx);
        let mut graph = cuda::Graph::new();
        let mut nodes = vec![];
        for Exec {
            node,
            inputs,
            outputs,
        } in exec
        {
            let Node { op, arg, .. } = node;
            let deps = std::mem::take(&mut nodes);
            macro_rules! add_to_graph {
                ($op:ident) => {
                    op::$op::add_to_graph(&graph, &deps, &mut modules, arg, inputs, outputs)
                };
            }
            nodes = match &*op {
                "embedding" => add_to_graph!(Embedding),
                "rms-norm" => add_to_graph!(RmsNorm),
                "linear" => add_to_graph!(Linear),
                _ => {
                    println!("next is {op}");
                    break;
                }
            };
        }
        graph.save_dot(std::env::current_dir().unwrap().join("graph.dot"));
    })
}

#[derive(Default)]
#[repr(transparent)]
struct TimeCollector(Vec<(String, Instant)>);

impl TimeCollector {
    pub fn push(&mut self, name: impl fmt::Display) {
        self.0.push((name.to_string(), Instant::now()))
    }
}

impl fmt::Display for TimeCollector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name_width = self.0.iter().map(|(name, _)| name.len()).max().unwrap_or(0) + 2;
        for i in 1..self.0.len() {
            writeln!(
                f,
                "{:·<name_width$}{:?}",
                self.0[i].0,
                self.0[i].1 - self.0[i - 1].1
            )?
        }
        Ok(())
    }
}
