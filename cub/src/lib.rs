use cuda::{AsRaw, ContextGuard, DevSlice, KernelFn, Stream};
use std::{
    ffi::{c_uint, c_void},
    mem::size_of,
};

pub struct ReduceMean {
    padding: KernelFn,
    folding: KernelFn,
    block_size: c_uint,
}

impl ReduceMean {
    pub fn new(max_item_size: usize, block_size: usize, ctx: &ContextGuard) -> Self {
        let ty_arg = "float";
        let ty_cal = "float";
        let items_per_thread = (max_item_size + block_size - 1) / block_size;
        let padding = format!("reduce_mean_padding_{block_size}");
        let folding = format!("reduce_mean_folding_{items_per_thread}_{block_size}");

        const REDUCE_MEAN: &str = include_str!("templates/reduce_mean.cuh");
        let code = format!(
            r#"{REDUCE_MEAN}

extern "C" __global__ void {padding}(
    {ty_arg} const *__restrict__ x_,
    {ty_arg}       *__restrict__ y_,
    unsigned int const leading_dim
){{
    padding<{ty_cal}, {block_size}>
    (x_, y_, leading_dim);
}}

extern "C" __global__ void {folding}(
    {ty_arg} const *__restrict__ x_,
    {ty_arg}       *__restrict__ y_,
    {ty_arg} const init,
    unsigned int const leading_dim,
    unsigned int const item_size
){{
    folding<{ty_cal}, {block_size}, {items_per_thread}>
    (x_, y_, init, leading_dim, item_size);
}}
"#
        );

        ctx.compile(code);
        Self {
            padding: KernelFn::get(&padding).unwrap(),
            folding: KernelFn::get(&folding).unwrap(),
            block_size: block_size as _,
        }
    }

    pub fn launch(&self, x: &DevSlice, y: &DevSlice, items_len: usize, stream: &Stream) {
        let row = (y.len() / size_of::<f32>()) as c_uint;
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
            let params: [*const c_void; 5] = [
                (&x_ptr) as *const _ as _,
                (&y_ptr) as *const _ as _,
                (&0.0f32) as *const _ as _,
                (&leading_dim) as *const _ as _,
                (&items_len) as *const _ as _,
            ];
            self.folding
                .launch(row, self.block_size, params.as_ptr(), 0, Some(stream));
        }
    }
}

#[cfg(test)]
mod bench;
#[cfg(test)]
mod verify;
