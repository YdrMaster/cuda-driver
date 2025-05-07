mod blob;
mod exec;
mod fmt;
mod gguf;
mod llama;
mod loader;
mod macros;
mod op;
mod range_collector;

use blob::Blob;
use cuda::{Device, VirByte, VirMem, memcpy_h2d};
use exec::{Exec, merge_cuda_graph};
use gguf::{GGufModel, map_files};
use ggus::ggml_quants::{digit_layout::types, f16};
use loader::WeightLoader;
use nn::{
    Dim, GraphBuilder, Tensor, TensorMeta,
    op::{self as nn_op},
};
use range_collector::RangeCollector;
use std::time::Instant;
use tensor::digit_layout::DigitLayout;

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
    let page_size = dev.mem_prop().granularity_minimum();
    let mut mapped = VirMem::new(ranges.size().div_ceil(page_size) * page_size, 0).map_on(&dev);
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

    let workspace_vir = reserve_pages(mem_range_map.range.len(), page_size);
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
    // 创建 kv cache
    let kv_cache = llama::kv_cache::<2>(&gguf); // kv cache 的最大容量
    let each = kv_cache.get() / kv_cache.shape()[0]; // kv cache 每个 token 的尺寸
    let kv_cache_vir = reserve_pages(*kv_cache.get(), page_size); // 为 kv cache 分配虚页
    let kv_cache = kv_cache
        .map(|_| kv_cache_vir[0].as_ptr()) // 存入张量
        .transform(|layout| layout.transpose(&[3, 1, 2, 0])); // 转置 [nkvh, nblk, 2, nctx, dh]

    // memcpy node 要求当时虚地址有对应的物理页
    let _weight = weight_vir.map(weight_phy);
    let _workspace = workspace_vir
        .into_iter()
        .map(|vir| vir.map_on(&dev))
        .collect::<Box<_>>();
    let _kv_cache = kv_cache_vir
        .into_iter()
        .map(|vir| vir.map_on(&dev))
        .collect::<Box<_>>();

    let tokens = [9038u32, 2501, 263, 931, 29892];
    let pos = [0u32, 1, 2, 3, 4];
    let attn_pos = 0;
    let attn_seq = tokens.len();
    let mask =
        Tensor::<usize, 2>::from_dim_slice(types::F16, &[1, 1, attn_seq, attn_pos + attn_seq])
            .map(|_| build_mask(types::F16, attn_pos, attn_seq));
    dev.context().apply(|ctx| {
        let (_handle, exec) = merge_cuda_graph(ctx, exec);
        times.push("build cuda graph");
        println!("{times}");

        memcpy_h2d(
            unsafe {
                std::slice::from_raw_parts_mut(
                    global_inputs[0].get().cast_mut().cast(),
                    size_of_val(&tokens),
                )
            },
            &tokens,
        );
        memcpy_h2d(
            unsafe {
                std::slice::from_raw_parts_mut(
                    global_inputs[1].get().cast_mut().cast(),
                    size_of_val(&pos),
                )
            },
            &pos,
        );
        let stream = ctx.stream();
        let mask = mask.as_ref().map(|blob| stream.from_host(blob));
        let mask = mask.as_ref().map(|blob| blob.as_ptr().cast::<VirByte>());

        for exec in &exec {
            match exec {
                Exec::Graph(graph) => graph.launch(&stream),
                Exec::Attention(exec::Attention { iblk, q, k, v, o }) => {
                    let q = q
                        .clone()
                        .transform(|layout| layout.tile_be(0, &[1, layout.shape()[0]]));
                    let k = k
                        .clone()
                        .transform(|layout| layout.tile_be(0, &[1, layout.shape()[0]]));
                    let v = v
                        .clone()
                        .transform(|layout| layout.tile_be(0, &[1, layout.shape()[0]]));
                    let o = o
                        .clone()
                        .transform(|layout| layout.tile_be(0, &[1, layout.shape()[0]]));

                    // [1, nkvh, 2, nctx, dh]
                    let blk_cache = kv_cache.clone().transform(|layout| {
                        layout.index(1, *iblk).tile_be(0, &[1, layout.shape()[0]])
                    });
                    let kv_cache = blk_cache
                        .clone()
                        .transform(|layout| layout.slice(3, 0, 1, attn_pos));
                    let kv_cahce_end = blk_cache
                        .clone()
                        .transform(|layout| layout.slice(3, attn_pos, 1, attn_seq));
                    let k_cache = kv_cache.clone().transform(|layout| layout.index(2, 0));
                    let v_cache = kv_cache.clone().transform(|layout| layout.index(2, 1));
                    let k_cache_end = kv_cahce_end.clone().transform(|layout| layout.index(2, 0));
                    let v_cache_end = kv_cahce_end.clone().transform(|layout| layout.index(2, 1));

                    fmt::fmt(&q, ctx);
                    fmt::fmt(&k, ctx);
                    fmt::fmt(&v, ctx);
                    op::launch_attention_kv(
                        q,
                        k,
                        v,
                        k_cache,
                        k_cache_end.clone(),
                        v_cache,
                        v_cache_end.clone(),
                        mask,
                        o.clone(),
                        &stream,
                    );
                    println!("-------------------------");
                    fmt::fmt(&k_cache_end, ctx);
                    fmt::fmt(&v_cache_end, ctx);
                    fmt::fmt(&o, ctx);
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

fn reserve_pages(size: usize, page_size: usize) -> Box<[VirMem]> {
    let n_pages = size.div_ceil(page_size);
    if n_pages == 0 {
        return Box::new([]);
    }

    let mut ans = Vec::with_capacity(n_pages);
    let first = VirMem::new(page_size, 0);
    let mut end = first.as_ptr_range().end;
    ans.push(first);
    while ans.len() < n_pages {
        let next = VirMem::new(page_size, end as _);
        let ptr = next.as_ptr();
        if ptr != end {
            ans.clear()
        }
        end = next.as_ptr_range().end;
        ans.push(next)
    }
    ans.into()
}

fn build_mask(dt: DigitLayout, pos: usize, seq: usize) -> Blob {
    let mut ans = Blob::new(dt.nbytes() * seq * (pos + seq));
    match dt {
        types::F16 => {
            let ([], slice, []) = (unsafe { ans.align_to_mut::<f16>() }) else {
                unreachable!()
            };
            fill_mask(slice, pos, seq, f16::ZERO, -f16::INFINITY)
        }
        _ => todo!(),
    }
    ans
}

fn fill_mask<T: Copy>(slice: &mut [T], pos: usize, seq: usize, zero: T, neg_inf: T) {
    slice.fill(zero);
    for r in 0..seq - 1 {
        slice[r * (pos + seq)..][..pos + seq][pos + r + 1..].fill(neg_inf)
    }
}
