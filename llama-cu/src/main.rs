mod blob;
mod exec;
mod fmt;
mod gguf;
mod llama;
mod loader;
mod op;
mod range_collector;

use cuda::{Device, VirByte, VirMem, memcpy_h2d};
use exec::{Exec, merge_cuda_graph};
use gguf::{GGufModel, map_files};
use ggus::ggml_quants::digit_layout::types;
use loader::WeightLoader;
use nn::{Dim, GraphBuilder, TensorMeta, op as nn_op};
use range_collector::RangeCollector;
use std::time::Instant;

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
    let mut mapped = VirMem::new(ranges.size().div_ceil(minimum) * minimum, 0).map_on(&dev);
    let edges = dev.context().apply(|ctx| {
        let mut loader = WeightLoader::new(
            ranges
                .sizes()
                .filter(|&(_, times)| times < 4)
                .map(|(size, _)| size),
        );

        let stream = ctx.stream();
        edges
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
            .collect::<Box<_>>()
    });
    let (weight_vir, weight_phy) = mapped.unmap();
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
    let graph = graph.lower(
        |key| unsafe { ptr.byte_add(mem_range_map.map[&key].start) },
        |&data| data,
    );
    let global_inputs = graph
        .0
        .topo
        .global_inputs()
        .map(|i| graph.0.edges[i].clone())
        .collect::<Box<_>>();
    let global_outputs = graph
        .0
        .topo
        .global_outputs()
        .iter()
        .map(|&i| graph.0.edges[i].clone())
        .collect::<Box<_>>();
    let exec = graph.into_exec();
    times.push("into exec");
    println!("{times}");

    // memcpy node 要求当时虚地址有对应的物理页
    let _weight = weight_vir.map(weight_phy);
    let _workspace = workspace_vir
        .into_iter()
        .map(|vir| vir.map_on(&dev))
        .collect::<Box<_>>();

    dev.context().apply(|ctx| {
        let tokens = [9038u32, 2501, 263, 931, 29892];
        memcpy_h2d(
            unsafe {
                std::slice::from_raw_parts_mut(
                    global_inputs[0].get().cast_mut().cast(),
                    size_of_val(&tokens),
                )
            },
            &tokens,
        );
        let tokens = [0u32, 1, 2, 3, 4];
        memcpy_h2d(
            unsafe {
                std::slice::from_raw_parts_mut(
                    global_inputs[1].get().cast_mut().cast(),
                    size_of_val(&tokens),
                )
            },
            &tokens,
        );
        //
        // let mut handle = op::Handle::new(ctx);
        // let stream = ctx.stream();
        // for nn::Exec {
        //     node,
        //     inputs,
        //     outputs,
        // } in exec
        // {
        //     let nn::Node { op, arg, .. } = node;
        //     use op::Operator;
        //     match &*op {
        //         "embedding" => op::Embedding::launch(&mut handle, arg, inputs, outputs, &stream),
        //         "rms-norm" => op::RmsNorm::launch(&mut handle, arg, inputs, outputs, &stream),
        //         "linear" => op::Linear::launch(&mut handle, arg, inputs, outputs, &stream),
        //         "rope" => op::Rope::launch(&mut handle, arg, inputs, outputs, &stream),
        //         "empty" => {}
        //         "attention" => {
        //             let mut inputs = inputs.into_iter();
        //             let q = inputs.next().unwrap();
        //             let k = inputs.next().unwrap();
        //             let v = inputs.next().unwrap();
        //             fmt::fmt(&q, ctx);
        //             fmt::fmt(&k, ctx);
        //             fmt::fmt(&v, ctx);
        //             break;
        //         }
        //         _ => {
        //             print!("todo! {op} ({arg:?})");
        //             for t in inputs {
        //                 print!(" {}{:?}", t.dt(), t.shape())
        //             }
        //             print!(" ->");
        //             for t in outputs {
        //                 print!(" {}{:?}", t.dt(), t.shape())
        //             }
        //             println!();
        //             break;
        //         }
        //     }
        // }
        //
        let (_handle, exec) = merge_cuda_graph(ctx, exec);

        times.push("build cuda graph");
        println!("{times}");

        let stream = ctx.stream();
        for (i, exec) in exec.iter().enumerate() {
            match exec {
                Exec::Graph(graph) => graph.launch(&stream),
                Exec::Attention { q, k, v, .. } => {
                    println!("{i} attention");
                    fmt::fmt(&q, ctx);
                    fmt::fmt(&k, ctx);
                    fmt::fmt(&v, ctx);
                    break;
                }
            }
        }
    });
}

#[derive(Default)]
#[repr(transparent)]
struct TimeCollector(Vec<(String, Instant)>);

impl TimeCollector {
    pub fn push(&mut self, name: impl std::fmt::Display) {
        self.0.push((name.to_string(), Instant::now()))
    }
}

impl std::fmt::Display for TimeCollector {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
