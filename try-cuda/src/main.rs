#![cfg(detected_cuda)]

use cuda::{params, ComputeCapability, DevMem, Device, Ptx};
use std::{
    ffi::{c_int, c_uint, CString},
    mem::size_of,
};

fn main() {
    cuda::init();

    let Some(dev) = Device::fetch() else {
        return;
    };

    basic(&dev);
    transpose(&dev);
}

fn compile(code: &str, cc: ComputeCapability) -> Ptx {
    let (ptx, msg) = Ptx::compile(code, cc);
    if !msg.is_empty() {
        println!("{msg}");
    }
    ptx.unwrap()
}

fn basic(dev: &Device) {
    let ptx = compile(include_str!("basic.cu"), dev.compute_capability());
    dev.retain_primary().apply(|ctx| {
        let module = ctx.load(&ptx);
        let kernel = module.get_kernel(CString::new("print").unwrap());

        let host = (0..64).collect::<Vec<_>>();
        let dev = ctx.from_host(&host);

        kernel.launch(32, 2, params![dev.as_ptr()].as_ptr(), 0, None);
        ctx.synchronize();
    });
}

fn transpose(dev: &Device) {
    let ptx = compile(include_str!("transpose.cu"), dev.compute_capability());
    dev.retain_primary().apply(|ctx| {
        let module = ctx.load(&ptx);
        let print = module.get_kernel(CString::new("print_matrix").unwrap());
        let one_thread = module.get_kernel(CString::new("one_thread").unwrap());
        let one_block = module.get_kernel(CString::new("one_block").unwrap());
        let multi_blocks = module.get_kernel(CString::new("multi_blocks").unwrap());

        const D: usize = 32;
        let d = D as c_uint;

        let host = (0..D as c_int).cycle().take(D * D).collect::<Vec<_>>();
        // let host = (0..(D * D) as c_int).collect::<Vec<_>>();
        let src = ctx.from_host(&host);
        let dst = ctx.malloc::<u8>(src.len());

        let print_dev =
            |mat: &DevMem| print.launch(1, 1, params![mat.as_ptr(), d].as_ptr(), 0, None);

        print_dev(&src);
        {
            one_thread.launch(
                1,
                1,
                params![dst.as_ptr(), src.as_ptr(), d].as_ptr(),
                0,
                None,
            );
            print_dev(&dst);
        }
        {
            one_block.launch(
                1,
                (d, d),
                params![dst.as_ptr(), src.as_ptr()].as_ptr(),
                0,
                None,
            );
            print_dev(&dst);
        }
        {
            let d_inner = 4;
            multi_blocks.launch(
                (d / d_inner, d / d_inner),
                (d_inner, d_inner),
                params![dst.as_ptr(), src.as_ptr()].as_ptr(),
                (d_inner * d_inner) as usize * size_of::<c_int>(),
                None,
            );
            print_dev(&dst);
        }

        ctx.synchronize();
    });
}
