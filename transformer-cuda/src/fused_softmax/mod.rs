use cuda::{AsRaw, ContextGuard, CudaDataType, DevSlice, KernelFn, Stream};
use std::ffi::{c_uint, c_void};

pub struct FusedSoftmax {
    padding: KernelFn,
    folding: KernelFn,
    block_size: c_uint,
    items_per_thread: c_uint,
}

impl FusedSoftmax {
    pub fn new(
        data_type: CudaDataType,
        max_seq_len: usize,
        block_size: usize,
        ctx: &ContextGuard,
    ) -> Self {
        let ty_arg = data_type.name();
        let mask = "AttentionCausualMask";
        let items_per_thread = (max_seq_len + block_size - 1) / block_size;
        let padding = format!("fused_softmax_padding_{block_size}");
        let folding = format!("fused_softmax_folding_{block_size}x{items_per_thread}");

        const FUSED_SOFTMAX: &str = include_str!("../templates/fused_softmax.cuh");
        let code = format!(
            r#"{FUSED_SOFTMAX}

extern "C" __global__ void {padding}(
    {ty_arg} *__restrict__ att,
    unsigned int const max_seq_len,
    unsigned int const buf_len
){{
    padding<{block_size}>
    (att, {mask}(), max_seq_len, buf_len);
}}

extern "C" __global__ void {folding}(
    {ty_arg} *__restrict__ att,
    unsigned int const max_seq_len,
    unsigned int const buf_len,
    unsigned int const att_len
){{
    folding<{block_size}, {items_per_thread}>
    (att, {mask}(), max_seq_len, buf_len, att_len);
}}
"#
        );

        ctx.compile(code);
        Self {
            padding: KernelFn::get(padding).unwrap(),
            folding: KernelFn::get(folding).unwrap(),
            block_size: block_size as _,
            items_per_thread: items_per_thread as _,
        }
    }

    pub fn launch(
        &self,
        att: &DevSlice,
        layout: (usize, usize, usize),
        valid: (usize, usize, usize),
        stream: &Stream,
    ) {
        let att_ptr = unsafe { att.as_raw() };

        let (total_batch, total_row, total_col) = layout;
        let max_seq_len = total_row as c_uint;
        let buf_len = total_col as c_uint;

        let (batch, row, col) = valid;
        debug_assert!(batch <= total_batch);
        debug_assert!(row <= total_row);
        debug_assert!(col <= total_col);
        let batch = batch as c_uint;
        let seq_len = row as c_uint;
        let att_len = col as c_uint;
        debug_assert!(seq_len <= att_len); // att_len = past_seq_len + seq_len

        let grid_dims = (batch, seq_len);
        let (kernel, block_dims) = if att_len <= self.block_size {
            (&self.padding, att_len)
        } else {
            (
                &self.folding,
                (att_len + self.items_per_thread - 1) / self.items_per_thread,
            )
        };

        let params: [*const c_void; 4] = [
            (&att_ptr) as *const _ as _,
            (&max_seq_len) as *const _ as _,
            (&buf_len) as *const _ as _,
            (&att_len) as *const _ as _,
        ];

        kernel.launch(grid_dims, block_dims, params.as_ptr(), 0, Some(stream));
    }
}

#[cfg(test)]
mod bench;
#[cfg(test)]
mod verify;
