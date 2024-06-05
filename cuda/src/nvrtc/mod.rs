mod kernel_fn;
mod module;
mod ptx;

pub use kernel_fn::{AsParam, KernelFn};
pub use module::{Module, ModuleSpore};
pub use ptx::Ptx;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Symbol<'a> {
    Global(&'a str),
    Device(&'a str),
}

impl<'a> Symbol<'a> {
    pub fn search(code: &'a str) -> impl Iterator<Item = Self> {
        code.split("extern")
            .skip(1)
            .filter_map(|s| s.trim().strip_prefix(r#""C""#))
            .filter_map(|f| f.split_once('(').map(|(head, _)| head.trim()))
            .filter_map(|head| {
                #[inline(always)]
                fn split(head: &str) -> &str {
                    head.rsplit_once(char::is_whitespace).unwrap().1
                }
                if head.contains("__global__") && head.contains("void") {
                    Some(Self::Global(split(head)))
                } else if head.contains("__device__") {
                    Some(Self::Device(split(head)))
                } else {
                    None
                }
            })
    }
}

#[test]
fn test_env() {
    assert!(!std::env!("CUDA_ROOT").is_empty());
}

#[test]
fn test_search_symbols() {
    let code = r#"
        extern  "C" __global__ void kernel0() { printf("Hello World from GPU!\n"); }
        extern "C"  __device__ long kernel1() { printf("Hello World from GPU!\n"); }
        extern "C"  __global__ void kernel2() { printf("Hello World from GPU!\n"); }
    "#;
    assert_eq!(
        Symbol::search(code).collect::<Vec<_>>(),
        &[
            Symbol::Global("kernel0"),
            Symbol::Device("kernel1"),
            Symbol::Global("kernel2"),
        ]
    );
}

#[test]
fn test_behavior() {
    use std::{
        ffi::{CStr, CString},
        ptr::{null, null_mut},
    };

    let src = r#"extern "C" __global__ void kernel() { printf("Hello World from GPU!\n"); }"#;
    let code = CString::new(src).unwrap();
    let ptx = {
        let mut program = null_mut();
        nvrtc!(nvrtcCreateProgram(
            &mut program,
            code.as_ptr().cast(),
            null(),
            0,
            null(),
            null(),
        ));
        nvrtc!(nvrtcCompileProgram(program, 0, null()));

        let mut ptx_len = 0;
        nvrtc!(nvrtcGetPTXSize(program, &mut ptx_len));

        let mut ptx = vec![0u8; ptx_len];
        nvrtc!(nvrtcGetPTX(program, ptx.as_mut_ptr().cast()));
        nvrtc!(nvrtcDestroyProgram(&mut program));
        ptx
    };
    let ptx = CStr::from_bytes_with_nul(ptx.as_slice()).unwrap();
    let name = CString::new("kernel").unwrap();

    let mut m = null_mut();
    let mut f = null_mut();
    crate::init();
    if let Some(dev) = crate::Device::fetch() {
        #[rustfmt::skip]
        dev.context().apply(|_| {
            driver!(cuModuleLoadData(&mut m, ptx.as_ptr().cast()));
            driver!(cuModuleGetFunction(&mut f, m, name.as_ptr()));
            driver!(cuLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, null_mut(), null_mut(), null_mut()));
        });
    };
}
