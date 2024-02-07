use cuda::{AsRaw, ContextGuard, CudaDataType, DevSlice, KernelFn, Stream};
use std::ffi::{c_uint, c_void};

pub struct Gather {
    fn_16bytes: KernelFn,
    fn_32bytes: KernelFn,
}

pub struct GatherMeta {
    pub data_type: CudaDataType,
    pub tokens_len: usize,
    pub hidden_size: usize,
}

impl Gather {
    pub fn new(ctx: &ContextGuard) -> Self {
        ctx.compile(include_str!("../templates/gather.cuh"));
        Self {
            fn_16bytes: KernelFn::get("gather_float4").unwrap(),
            fn_32bytes: KernelFn::get("gather_double4").unwrap(),
        }
    }

    pub fn launch(
        &self,
        hidden_state: &DevSlice,
        vocab: &DevSlice,
        tokens: &DevSlice,

        meta: &GatherMeta,
        max_block_size: usize,
        stream: &Stream,
    ) {
        let grid_size = meta.tokens_len as c_uint;
        let hidden_state = unsafe { hidden_state.as_raw() };
        let vocab = unsafe { vocab.as_raw() };
        let tokens = unsafe { tokens.as_raw() };
        let params: [*const c_void; 3] = [
            (&hidden_state) as *const _ as _,
            (&vocab) as *const _ as _,
            (&tokens) as *const _ as _,
        ];

        let hidden_bytes_size = meta.hidden_size * meta.data_type.size();
        if hidden_bytes_size <= 16 * max_block_size {
            debug_assert_eq!(hidden_bytes_size % 16, 0);
            self.fn_16bytes.launch(
                grid_size,
                (hidden_bytes_size / 16) as c_uint,
                params.as_ptr(),
                0,
                Some(stream),
            );
        } else if hidden_bytes_size <= 32 * max_block_size {
            debug_assert_eq!(hidden_bytes_size % 32, 0);
            self.fn_32bytes.launch(
                grid_size,
                (hidden_bytes_size / 32) as c_uint,
                params.as_ptr(),
                0,
                Some(stream),
            );
        } else {
            unimplemented!()
        }
    }
}
