use crate::bindings as cuda;
use core::fmt;
use std::{
    ffi::{c_char, CString},
    path::{Path, PathBuf},
    ptr::{null, null_mut},
};

pub struct Ptx(CString);

impl Ptx {
    pub fn compile(code: impl AsRef<str>) -> (Result<Self, cuda::nvrtcResult>, String) {
        let code = code.as_ref();
        let need_cub = code.contains("cub");
        let need_thrust = code.contains("thrust");
        let code = {
            let mut headers = String::new();

            if code.contains("half") {
                headers.push_str("#include <cuda_fp16.h>\n");
            }
            if code.contains("nv_bfloat16") {
                headers.push_str("#include <cuda_bf16.h>\n");
            }

            if !headers.is_empty() {
                headers.push_str(code);
                CString::new(headers.as_str())
            } else {
                CString::new(code)
            }
            .unwrap()
        };
        let mut program = null_mut();
        nvrtc!(nvrtcCreateProgram(
            &mut program,
            code.as_ptr().cast(),
            null(),
            0,
            null(),
            null(),
        ));

        let mut options = vec![
            CString::new("--std=c++17").unwrap(),
            CString::new("--gpu-architecture=compute_80").unwrap(),
        ];
        fn include_dir(options: &mut Vec<CString>, dir: impl AsRef<Path>) {
            options.push(CString::new(format!("-I{}\n", dir.as_ref().display())).unwrap());
        }
        if need_cub || need_thrust {
            let cccl = std::option_env!("CCCL_ROOT").map_or_else(
                || PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("cccl"),
                PathBuf::from,
            );
            assert!(cccl.is_dir(), "cccl not exist");
            include_dir(&mut options, cccl.join("libcudacxx/include"));
            include_dir(&mut options, cccl.join("libcudacxx/include/cuda/std"));
            if need_cub {
                include_dir(&mut options, cccl.join("cub"));
            }
            if need_thrust {
                include_dir(&mut options, cccl.join("thrust"));
            }
        }
        options.push(CString::new(format!("-I{}/include", std::env!("CUDA_ROOT"))).unwrap());
        let options = options
            .iter()
            .map(|s| s.as_ptr().cast::<c_char>())
            .collect::<Vec<_>>();

        let result =
            unsafe { cuda::nvrtcCompileProgram(program, options.len() as _, options.as_ptr()) };
        let log = {
            let mut log_len = 0;
            nvrtc!(nvrtcGetProgramLogSize(program, &mut log_len));
            if log_len > 1 {
                let mut log = vec![0u8; log_len];
                nvrtc!(nvrtcGetProgramLog(program, log.as_mut_ptr().cast()));
                log.pop();
                std::str::from_utf8(&log).unwrap().trim().to_string()
            } else {
                String::new()
            }
        };
        let ans = if result == cuda::nvrtcResult::NVRTC_SUCCESS {
            let mut ptx_len = 0;
            nvrtc!(nvrtcGetPTXSize(program, &mut ptx_len));
            let mut ptx = vec![0u8; ptx_len];
            nvrtc!(nvrtcGetPTX(program, ptx.as_mut_ptr().cast()));
            nvrtc!(nvrtcDestroyProgram(&mut program));
            Ok(Self(CString::from_vec_with_nul(ptx).unwrap()))
        } else {
            Err(result)
        };
        (ans, log)
    }
}

impl fmt::Display for Ptx {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.0.to_string_lossy())
    }
}

impl Ptx {
    #[inline]
    pub fn as_ptr(&self) -> *const c_char {
        self.0.as_ptr()
    }
}
