mod blob;
mod gguf;
mod loader;
mod nn;

use ::nn::{Context, Dim};
use blob::Blob;
use gguf::{GGufModel, GGufTensor, map_files};
use ggus::{GGufMetaMapExt, ggml_quants::digit_layout::types};
use std::fmt;
use tensor::Tensor;

// cargo run --release -- ../TinyStory-5M-v0.0-F32.gguf
fn main() {
    if cuda::init().is_err() {
        return;
    }

    let path = std::env::args_os().nth(1).unwrap();
    let maps = map_files(path);
    let mut gguf = GGufModel::read(maps.iter().map(|x| &**x));
    insert_sin_cos(&mut gguf);

    let nvoc = 32000usize;
    let nctx = meta![gguf => llm_context_length];
    let nblk = meta![gguf => llm_block_count];
    let d = meta![gguf => llm_embedding_length];
    let nh = meta![gguf => llm_attention_head_count];
    let nkvh = meta![gguf => llm_attention_head_count_kv; nh];
    let dh = meta![gguf => llm_rope_dimension_count; d / nh];
    let di = meta![gguf => llm_feed_forward_length] * 2;
    let epsilon = meta![gguf => llm_attention_layer_norm_rms_epsilon; 1e-5];

    let context = Context::default();
    let x = context.global_input(types::U32, [Dim::var("n")]);
    let pos = context.global_input(types::U32, [Dim::var("n")]);
    context
        .launch(
            ::nn::LLaMA {
                embedding: ::nn::Embedding {
                    dt: types::F32,
                    d: d.into(),
                    wte: ::nn::Table {
                        row: nvoc.into(),
                        weight: "token_embd.weight".to_string(),
                    },
                    wpe: None,
                },
                blks: (0..nblk)
                    .map(|iblk| ::nn::TransformerBlk {
                        attn_norm: ::nn::Normalization {
                            d: d.into(),
                            epsilon: epsilon as _,
                            items: ::nn::NormType::RmsNorm {
                                dt: types::F32,
                                scale: format!("blk.{iblk}.attn_norm.weight"),
                            },
                        },
                        attn: ::nn::Attention {
                            nh: nh.into(),
                            nkvh: nkvh.into(),
                            qkv: ::nn::Linear {
                                dt: types::F32,
                                shape: [((nh + nkvh + nkvh) * dh).into(), d.into()],
                                weight: format!("blk.{iblk}.attn_qkv.weight"),
                                bias: None,
                            },
                            rope: Some(::nn::RoPE {
                                nctx: nctx.into(),
                                sin: "sin_table".into(),
                                cos: "cos_table".into(),
                            }),
                            output: ::nn::Linear {
                                dt: types::F32,
                                shape: [d.into(), (nh * dh).into()],
                                weight: format!("blk.{iblk}.attn_output.weight"),
                                bias: None,
                            },
                        },
                        ffn_norm: ::nn::Normalization {
                            d: d.into(),
                            epsilon: epsilon as _,
                            items: ::nn::NormType::RmsNorm {
                                dt: types::F32,
                                scale: format!("blk.{iblk}.ffn_norm.weight"),
                            },
                        },
                        ffn: ::nn::Mlp {
                            up: ::nn::Linear {
                                dt: types::F32,
                                shape: [(di * 2).into(), d.into()],
                                weight: format!("blk.{iblk}.ffn_gate_up.weight"),
                                bias: None,
                            },
                            act: ::nn::Activation::SwiGLU,
                            down: ::nn::Linear {
                                dt: types::F32,
                                shape: [di.into(), d.into()],
                                weight: format!("blk.{iblk}.ffn_down.weight"),
                                bias: None,
                            },
                        },
                    })
                    .collect(),
            },
            [x, pos],
        )
        .unwrap();
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

struct Fmt<'a, const N: usize>(Tensor<&'a [u8], N>);

impl<const N: usize> fmt::Display for Fmt<'_, N> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let layout = self.0.layout();
        let ptr = self.0.get().as_ptr();
        match self.0.dt() {
            types::F32 => unsafe { layout.write_array(f, ptr.cast::<DataFmt<f32>>()) },
            types::U32 => unsafe { layout.write_array(f, ptr.cast::<DataFmt<u32>>()) },
            _ => todo!(),
        }
    }
}

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct DataFmt<T>(T);

impl fmt::Display for DataFmt<f32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0. {
            write!(f, " ________")
        } else {
            write!(f, "{:>9.3e}", self.0)
        }
    }
}

impl fmt::Display for DataFmt<u32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.0 == 0 {
            write!(f, " ________")
        } else {
            write!(f, "{:>6}", self.0)
        }
    }
}
