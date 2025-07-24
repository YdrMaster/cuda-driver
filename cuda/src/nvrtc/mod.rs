mod kernel_fn;
mod module;
mod ptx;

use std::{ffi::CString, str::FromStr};

pub use kernel_fn::{KernelFn, KernelParamPtrs, KernelParams};
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

    pub fn to_c_string(&self) -> CString {
        match self {
            Self::Global(s) | Self::Device(s) => CString::from_str(s).unwrap(),
        }
    }
}

#[test]
fn test_search_symbols() {
    let code = r#"
extern "C" __global__ void kernel0() { printf("Hello World from GPU!\n"); }
extern "C" __device__ long kernel1() { printf("Hello World from GPU!\n"); }
extern "C" __global__ void kernel2() { printf("Hello World from GPU!\n"); }
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
        ffi::CString,
        ptr::{null, null_mut},
    };

    let src = r#"extern "C" __global__ void kernel() { printf("Hello World from GPU!\n"); }"#;
    let code = CString::new(src).unwrap();
    let ptx = {
        let mut program = null_mut();
        nvrtc!(hcrtcCreateProgram(
            &mut program,
            code.as_ptr().cast(),
            null(),
            0,
            null(),
            null(),
        ));
        nvrtc!(hcrtcCompileProgram(program, 0, null()));

        let mut ptx_len = 0;
        nvrtc!(hcrtcGetBitcodeSize(program, &mut ptx_len));
        println!("ptx_len = {ptx_len}");

        let mut ptx = vec![0u8; ptx_len];
        nvrtc!(hcrtcGetBitcode(program, ptx.as_mut_ptr().cast()));
        nvrtc!(hcrtcDestroyProgram(&mut program));
        ptx
    };
    let ptx = ptx.as_slice();
    let name = CString::new("kernel").unwrap();

    let mut m = null_mut();
    let mut f = null_mut();

    if let Err(crate::NoDevice) = crate::init() {
        return;
    }
    crate::Device::new(0).context().apply(|_| {
        driver!(hcModuleLoadData(&mut m, ptx.as_ptr().cast()));
        driver!(hcModuleGetFunction(&mut f, m, name.as_ptr()));
        #[rustfmt::skip]
        driver!(hcModuleLaunchKernel(f, 1, 1, 1, 1, 1, 1, 0, null_mut(), null_mut(), null_mut()));
    });
}
