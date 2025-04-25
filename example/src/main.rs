mod blob;
mod gguf;
mod loader;
mod nn;

use blob::Blob;
use cuda::{Device, VirMem};
use gguf::{GGufModel, GGufTensor, map_files};
use ggus::{GGufMetaMapExt, ggml_quants::digit_layout::types};
use loader::{MemCalculator, WeightLoader};
use nn::llama::Meta;
use std::{
    collections::{BTreeMap, HashMap},
    ops::Range,
};
use tensor::Tensor;

pub use nn::*;
const ALIGN: usize = 512;

fn main() {
    if !cuda::init().is_ok() {
        return;
    }

    let path = std::env::args_os().nth(1).unwrap();
    let maps = map_files(path);
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    insert_sin_cos(&mut gguf);

    let nblk = meta![gguf => llm_block_count];
    let llama = llama::Weight::<String> {
        embedding: embedding::Weight {
            token_embd: "token_embd.weight".into(),
        },
        blks: (0..nblk)
            .map(|iblk| transformer::Weight {
                attn_norm: normalization::Weight::rms_norm(format!("blk.{iblk}.attn_norm.weight")),
                attn: self_attn::Weight {
                    qkv: linear::Weight::new(format!("blk.{iblk}.attn_qkv.weight"), None),
                    rope: rope::Weight::new("sin_table".into(), "cos_table".into()),
                    output: linear::Weight::new(format!("blk.{iblk}.attn_output.weight"), None),
                },
                ffn_norm: normalization::Weight::rms_norm(format!("blk.{iblk}.ffn_norm.weight")),
                ffn: ffn::Weight {
                    up: linear::Weight::new(format!("blk.{iblk}.ffn_gate_up.weight"), None),
                    down: linear::Weight::new(format!("blk.{iblk}.ffn_down.weight"), None),
                },
            })
            .collect(),
        output_norm: normalization::Weight::rms_norm(format!("output_norm.weight")),
        lm_head: linear::Weight::new("output.weight".into(), None),
    }
    .map(|name| &gguf.tensors[&*name]);

    let mut weights = HashMap::<*const u8, Range<usize>>::new();
    let mut calculator = MemCalculator::new(ALIGN);
    let mut sizes = BTreeMap::<usize, usize>::new();
    let _ = llama.as_ref().map(|t| {
        let ptr = t.data.as_ptr();
        let len = t.data.len();

        use std::collections::hash_map::Entry::{Occupied, Vacant};
        match weights.entry(ptr) {
            Occupied(entry) => {
                assert_eq!(entry.get().len(), len)
            }
            Vacant(entry) => {
                entry.insert(calculator.push(len));
                *sizes.entry(len).or_insert(0) += 1
            }
        }
    });

    let dev = Device::new(0);
    let prop = dev.mem_prop();
    let page_size = prop.granularity_minimum();

    let meta = build_llama_meta(&gguf);
    let workspace = meta.workspace(1, ALIGN);
    println!("size = {}", workspace.size);

    let workspace = VirMem::new(workspace.size.div_ceil(page_size) * page_size);
    println!("workspace.len = {}", workspace.len());

    let kv_cache = meta.kv_cache(4096).take();
    println!("kv cache = {kv_cache}");

    let kv_cache = VirMem::new(kv_cache.div_ceil(page_size) * page_size);
    println!("kv cache = {}", kv_cache.len());

    Device::new(0).context().apply(|ctx| {
        let mut loader = WeightLoader::new(
            sizes
                .iter()
                .filter(|(_, times)| **times < nblk)
                .map(|(&size, _)| size),
        );
        let mut weight_memory = ctx.malloc::<u8>(calculator.size());

        let stream = ctx.stream();
        let weights = llama
            .map(|t| {
                let range = weights[&t.data.as_ptr()].clone();
                let dev = &mut weight_memory[range.clone()];
                loader.load(dev, &t.data, &stream);
                Tensor::<usize, 4>::from_dim_slice(t.ty, &t.shape).map(|_| range)
            })
            .map(|t| t.map(|range| &weight_memory[range]));
    })
}

/// 构造 sin cos 表张量，存储到 GGufModel 中
fn insert_sin_cos(gguf: &mut GGufModel) {
    let nctx = meta![gguf => llm_context_length];
    let d = meta![gguf => llm_embedding_length];
    let nh = meta![gguf => llm_attention_head_count];
    let dh = meta![gguf => llm_rope_dimension_count; d / nh];
    let theta = meta![gguf => llm_rope_freq_base; 1e4];

    let ty = types::F32;
    let mut sin = Blob::new(nctx * dh / 2 * ty.nbytes());
    let mut cos = Blob::new(nctx * dh / 2 * ty.nbytes());

    {
        let ([], sin, []) = (unsafe { sin.align_to_mut::<f32>() }) else {
            unreachable!()
        };
        let ([], cos, []) = (unsafe { cos.align_to_mut::<f32>() }) else {
            unreachable!()
        };
        for pos in 0..nctx {
            for i in 0..dh / 2 {
                let theta = theta.powf(-((2 * i) as f32 / dh as f32));
                let freq = pos as f32 * theta;
                let (sin_, cos_) = freq.sin_cos();
                sin[pos * dh / 2 + i] = sin_;
                cos[pos * dh / 2 + i] = cos_;
            }
        }
    }

    let mut insert = |name, data: Blob| {
        assert!(
            gguf.tensors
                .insert(
                    name,
                    GGufTensor {
                        ty,
                        shape: vec![nctx, dh / 2].into(),
                        data: data.into(),
                    },
                )
                .is_none()
        )
    };
    insert("sin_table", sin);
    insert("cos_table", cos);
}

fn build_llama_meta(gguf: &GGufModel) -> llama::Meta {
    let nblk = meta![gguf => llm_block_count];
    let d = meta![gguf => llm_embedding_length];
    let nh = meta![gguf => llm_attention_head_count];
    let nkvh = meta![gguf => llm_attention_head_count_kv; nh];
    let dh = meta![gguf => llm_rope_dimension_count; d / nh];
    let di = meta![gguf => llm_feed_forward_length] * 2;
    Meta {
        t_tok: types::U32,
        t_pos: types::U32,
        t_embd: types::F32,
        nblk,
        d,
        nh,
        nkvh,
        dh,
        di,
    }
}

#[test]
fn test_behavior() {
    use cublas::{Cublas, cublas};
    use cuda::{AsRaw, Device, memcpy_d2h};

    if !cuda::init().is_ok() {
        return;
    }

    Device::new(0).context().apply(|ctx| {
        // [3, 2] <- [3, 4] x [4, 2]
        let mut c = ctx.from_host(&[0.0f32; 6]);
        let a: [f32; 12] = std::array::from_fn(|i| (i % 3 + 1) as _);
        let a = ctx.from_host(&a);
        let b: [f32; 8] = std::array::from_fn(|i| (i % 4 + 1) as _);
        let b = ctx.from_host(&b);

        let alpha = 1.0f32;
        let beta = 1.0f32;

        let stream = ctx.stream();
        let mut blas = Cublas::new(ctx);

        let stream = stream.capture();

        blas.set_stream(&stream);
        cublas!(cublasGemmEx(
            blas.as_raw(),
            cublasOperation_t::CUBLAS_OP_N,
            cublasOperation_t::CUBLAS_OP_N,
            3,
            2,
            4,
            (&raw const alpha).cast(),
            a.as_ptr().cast(),
            cudaDataType::CUDA_R_32F,
            3,
            b.as_ptr().cast(),
            cudaDataType::CUDA_R_32F,
            4,
            (&raw const beta).cast(),
            c.as_mut_ptr().cast(),
            cudaDataType::CUDA_R_32F,
            3,
            cublasComputeType_t::CUBLAS_COMPUTE_32F,
            cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
        ));

        let graph = stream.end();
        let exec = graph.instantiate();

        let stream = ctx.stream();
        exec.launch(&stream);

        let mut host = vec![0.0f32; 3 * 2];
        stream.synchronize();
        memcpy_d2h(&mut host, &c);

        println!("{host:?}");
    })
}
