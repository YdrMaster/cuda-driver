use crate::{
    Version,
    bindings::{nvrtcCompileProgram, nvrtcResult},
};
use std::{
    env::temp_dir,
    ffi::{CString, c_char},
    fmt,
    path::PathBuf,
    process::Command,
    ptr::{null, null_mut},
    sync::OnceLock,
};

#[repr(transparent)]
pub struct Ptx(Vec<u8>);

impl Ptx {
    pub fn compile(code: impl AsRef<str>, cc: Version) -> (Result<Self, nvrtcResult>, String) {
        let code = code.as_ref();

        let options = collect_options(code, cc);
        let options = options
            .iter()
            .map(|s| s.as_ptr().cast::<c_char>())
            .collect::<Vec<_>>();

        let code = {
            let mut headers = String::new();

            if code.contains("half") {
                headers.push_str("#include <cuda_fp16.h>\n")
            }
            if code.contains("nv_bfloat16") {
                headers.push_str("#include <cuda_bf16.h>\n")
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

        let result = unsafe { nvrtcCompileProgram(program, options.len() as _, options.as_ptr()) };
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
        let ans = if result == nvrtcResult::NVRTC_SUCCESS {
            let mut ptx_len = 0;
            nvrtc!(nvrtcGetPTXSize(program, &mut ptx_len));
            let mut ptx = vec![0u8; ptx_len];
            nvrtc!(nvrtcGetPTX(program, ptx.as_mut_ptr().cast()));
            nvrtc!(nvrtcDestroyProgram(&mut program));
            Ok(Self(ptx))
        } else {
            Err(result)
        };
        (ans, log)
    }
}

impl fmt::Display for Ptx {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", String::from_utf8_lossy(&self.0))
    }
}

impl Ptx {
    #[inline]
    pub fn as_ptr(&self) -> *const u8 {
        self.0.as_ptr()
    }
}

fn collect_options(code: &str, _cc: Version) -> Vec<CString> {
    let mut options = vec![
        CString::new("--std=c++17").unwrap(),
        #[cfg(nvidia)]
        CString::new(format!(
            "--gpu-architecture=compute_{}",
            _cc.to_arch_string()
        ))
        .unwrap(),
    ];
    fn include_dir(dir: impl fmt::Display) -> CString {
        CString::new(format!("-I{dir}\n")).unwrap()
    }
    #[cfg(nvidia)]
    {
        use std::sync::LazyLock;
        static VERSION: LazyLock<Version> = LazyLock::new(crate::version);
        const TARGET: Version = Version {
            major: 12,
            minor: 6,
        };

        if *VERSION < TARGET {
            let cccl = std::option_env!("CCCL_ROOT").map_or_else(clone_cccl, PathBuf::from);
            if cccl.is_dir() {
                const DIRS: &[&str] = &[
                    "libcudacxx/include",
                    "libcudacxx/include/cuda/std",
                    "cub",
                    "thrust",
                ];
                options.extend(
                    DIRS.iter()
                        .map(|path| include_dir(cccl.join(path).display())),
                )
            } else if code.contains("cub") || code.contains("thrust") {
                log::warn!("cccl not found, but cub or thrust is used in code")
            }
        }
    }
    #[cfg(iluvatar)]
    let _ = code;

    // let cutlass = std::option_env!("CUTLASS_ROOT").map_or_else(
    //     || PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("cutlass"),
    //     PathBuf::from,
    // );
    // if cutlass.is_dir() {
    //     options.push(include_dir(cutlass.join("include")));
    //     options.push(CString::new("-default-device").unwrap());
    // } else if code.contains("cutlass") || code.contains("cute") {
    //     warn!("cutlass not found, but cutlass or cute is used in code");
    // }

    let toolkit = if cfg!(nvidia) {
        find_cuda_helper::find_cuda_root().unwrap()
    } else if cfg!(iluvatar) {
        search_corex_tools::find_corex().unwrap()
    } else {
        unimplemented!()
    };

    options.push(include_dir(toolkit.join("include").display()));
    options
}
#[allow(dead_code)]
fn clone_cccl() -> PathBuf {
    static ONCE: OnceLock<PathBuf> = OnceLock::new();
    ONCE.get_or_init(|| {
        let temp = temp_dir();
        let cccl = temp.join("cccl");
        if !cccl.is_dir() {
            println!("cccl not found, cloning from github");
            Command::new("git")
                .args([
                    "clone",
                    "https://github.com/NVIDIA/cccl",
                    "--branch",
                    "v2.8.3",
                    "--depth=1",
                ])
                .current_dir(temp)
                .status()
                .unwrap();
            println!("cccl cloned in {}", cccl.display())
        }
        cccl
    })
    .clone()
}
