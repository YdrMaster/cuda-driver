use cuda::{AsRaw, ContextGuard, CudaDataType, DevSlice, KernelFn, Stream};
use std::ffi::{c_uint, c_void};

pub struct RmsNormalization {
    padding: KernelFn,
    folding: KernelFn,
    data_type: CudaDataType,
    block_size: c_uint,
    items_per_thread: c_uint,
}

impl RmsNormalization {
    pub fn new(
        data_type: CudaDataType,
        max_item_size: usize,
        block_size: usize,
        ctx: &ContextGuard,
    ) -> Self {
        let ty_arg = data_type.name();
        let items_per_thread = (max_item_size + block_size - 1) / block_size;
        let padding = format!("rms_normalization_padding_{block_size}");
        let folding = format!("rms_normalization_folding_{items_per_thread}x{block_size}");

        const RMS_NORMALIZATION: &str = include_str!("../templates/rms_normalization.cuh");
        let code = format!(
            r#"{RMS_NORMALIZATION}

extern "C" __global__ void {padding}(
    {ty_arg}       *__restrict__ y,
    {ty_arg} const *__restrict__ x,
    {ty_arg} const *__restrict__ w,
    float epsilon,
    unsigned int const leading_dim
){{
    padding<{block_size}>
    (y, x, w, epsilon, leading_dim);
}}

extern "C" __global__ void {folding}(
    {ty_arg}       *__restrict__ y,
    {ty_arg} const *__restrict__ x,
    {ty_arg} const *__restrict__ w,
    float epsilon,
    unsigned int const leading_dim,
    unsigned int const items_size
){{
    folding<{block_size}, {items_per_thread}>
    (y, x, w, epsilon, leading_dim, items_size);
}}
"#
        );

        ctx.compile(code);
        Self {
            padding: KernelFn::get(&padding).unwrap(),
            folding: KernelFn::get(&folding).unwrap(),
            data_type,
            block_size: block_size as _,
            items_per_thread: items_per_thread as _,
        }
    }

    pub fn launch(&self, x: &DevSlice, y: &DevSlice, items_len: usize, stream: &Stream) {
        let row = (y.len() / self.data_type.size()) as c_uint;
        let leading_dim = (x.len() / y.len()) as c_uint;
        let x_ptr = unsafe { x.as_raw() };
        let y_ptr = unsafe { y.as_raw() };
        let items_len = items_len as c_uint;
        if items_len <= self.block_size {
            let params: [*const c_void; 3] = [
                (&x_ptr) as *const _ as _,
                (&y_ptr) as *const _ as _,
                (&leading_dim) as *const _ as _,
            ];
            self.padding
                .launch(row, items_len, params.as_ptr(), 0, Some(stream));
        } else {
            let params: [*const c_void; 4] = [
                (&x_ptr) as *const _ as _,
                (&y_ptr) as *const _ as _,
                (&leading_dim) as *const _ as _,
                (&items_len) as *const _ as _,
            ];
            let block_size = (items_len + self.items_per_thread - 1) / self.items_per_thread;
            self.folding
                .launch(row, block_size, params.as_ptr(), 0, Some(stream));
        }
    }
}
