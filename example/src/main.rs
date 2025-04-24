mod blob;
mod gguf;
mod loader;
mod nn;

use blob::Blob;
use cuda::Device;
use gguf::{GGufModel, GGufTensor, map_files};
use ggus::{GGufMetaMapExt, ggml_quants::digit_layout::types};
use loader::{WeightLoader, WeightMemCalculator};
use nn::{
    Attention, Embedding, Ffn, Linear, LinearResidual, Llama, RmsNorm, RoPE, SelfAttn, SwiGLU,
    TransformerBlk,
};
use std::{
    collections::{BTreeMap, HashMap},
    marker::PhantomData,
    ops::Range,
};
use tensor::Tensor;

fn main() {
    if !cuda::init().is_ok() {
        return;
    }

    let path = std::env::args_os().nth(1).unwrap();
    let maps = map_files(path);
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    insert_sin_cos(&mut gguf);

    let nblk = meta![gguf => llm_block_count];
    let llama = Llama::<String> {
        embedding: Embedding {
            token_embd: "token_embd.weight".into(),
        },
        blks: (0..nblk)
            .map(|iblk| TransformerBlk {
                attn_norm: RmsNorm {
                    weight: format!("blk.{iblk}.attn_norm.weight"),
                },
                attn: SelfAttn {
                    qkv: Linear {
                        weight: format!("blk.{iblk}.attn_qkv.weight"),
                    },
                    q_rope: RoPE {
                        sin: "sin_table".into(),
                        cos: "cos_table".into(),
                    },
                    k_rope: RoPE {
                        sin: "sin_table".into(),
                        cos: "cos_table".into(),
                    },
                    attn: Attention(PhantomData),
                    output: LinearResidual {
                        weight: format!("blk.{iblk}.attn_output.weight"),
                    },
                },
                ffn_norm: RmsNorm {
                    weight: format!("blk.{iblk}.ffn_norm.weight"),
                },
                ffn: Ffn {
                    up: Linear {
                        weight: format!("blk.{iblk}.ffn_gate_up.weight"),
                    },
                    act: SwiGLU(PhantomData),
                    down: LinearResidual {
                        weight: format!("blk.{iblk}.ffn_down.weight"),
                    },
                },
            })
            .collect(),
        output_norm: RmsNorm {
            weight: "output_norm.weight".into(),
        },
        lm_head: Linear {
            weight: "output.weight".into(),
        },
    }
    .map(|name| &gguf.tensors[&*name]);

    let mut weights = HashMap::<*const u8, Range<usize>>::new();
    let mut calculator = WeightMemCalculator::new(512);
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

    Device::new(0).context().apply(|ctx| {
        let mut loader = WeightLoader::new(
            sizes
                .iter()
                .filter(|(_, times)| **times < nblk)
                .map(|(&size, _)| size),
        );
        let mut weight_memory = ctx.malloc::<u8>(calculator.size());

        let stream = ctx.stream();
        let _llama = llama
            .map(|t| {
                let range = weights[&t.data.as_ptr()].clone();
                let dev = &mut weight_memory[range.clone()];
                loader.load(dev, &t.data, &stream);
                Tensor::<usize, 4>::from_dim_slice(t.ty, &t.shape).map(|_| range)
            })
            .map(|t| t.map(|range| &weight_memory[range]));
        stream.synchronize();
    })
}

// 构造 sin cos 表张量，存储到 GGufModel 中
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
