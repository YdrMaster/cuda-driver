use cuda::{AsRaw, ContextGuard, CudaDataType, DevSlice, KernelFn, Stream};
use std::ffi::{c_uint, c_void};

pub struct FusedSoftmax {
    padding: KernelFn,
    folding: KernelFn,
    data_type: CudaDataType,
    block_size: c_uint,
    items_per_thread: c_uint,
}

impl FusedSoftmax {
    pub fn new(
        data_type: CudaDataType,
        max_item_size: usize,
        block_size: usize,
        ctx: &ContextGuard,
    ) -> Self {
        let ty_arg = data_type.name();
        let mask = "AttentionCausualMask";
        let items_per_thread = (max_item_size + block_size - 1) / block_size;
        let padding = format!("fused_softmax_padding_{block_size}");
        let folding = format!("fused_softmax_folding_{items_per_thread}x{block_size}");

        const FUSED_SOFTMAX: &str = include_str!("../templates/fused_softmax.cuh");
        let code = format!(
            r#"{FUSED_SOFTMAX}

extern "C" __global__ void {padding}(
    {ty_arg} *__restrict__ att,
    unsigned int const leading_dim
){{
    padding<{block_size}>
    (att, {mask}(), leading_dim);
}}

extern "C" __global__ void {folding}(
    {ty_arg} *__restrict__ att,
    unsigned int const leading_dim,
    unsigned int const att_len
){{
    folding<{block_size}, {items_per_thread}>
    (att, {mask}(), leading_dim, att_len);
}}
"#
        );

        ctx.compile(code);
        Self {
            padding: KernelFn::get(padding).unwrap(),
            folding: KernelFn::get(folding).unwrap(),
            data_type,
            block_size: block_size as _,
            items_per_thread: items_per_thread as _,
        }
    }

    pub fn launch(&self, att: &DevSlice, leading_dim: usize, att_len: usize, stream: &Stream) {
        let row = (att.len() / self.data_type.size() / leading_dim) as c_uint;
        let att_ptr = unsafe { att.as_raw() };
        let leading_dim = leading_dim as c_uint;
        let att_len = att_len as c_uint;
        let params: [*const c_void; 3] = [
            (&att_ptr) as *const _ as _,
            (&leading_dim) as *const _ as _,
            (&att_len) as *const _ as _,
        ];
        if att_len <= self.block_size {
            self.padding
                .launch(row, att_len, params.as_ptr(), 0, Some(stream));
        } else {
            let block_size = (att_len + self.items_per_thread - 1) / self.items_per_thread;
            self.folding
                .launch(row, block_size, params.as_ptr(), 0, Some(stream));
        }
    }
}

#[cfg(test)]
mod bench;
