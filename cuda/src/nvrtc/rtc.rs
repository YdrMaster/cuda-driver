use crate::{
    Version,
    bindings::{nvrtcCompileProgram, nvrtcResult},
};
use std::{
    collections::BTreeSet,
    ffi::CString,
    fmt,
    path::PathBuf,
    ptr::{null, null_mut},
    str::FromStr,
};

#[derive(Clone, Debug)]
pub struct Rtc {
    cc: Version,
    std: usize,
    line_info: bool,
    fast_math: bool,
    extra_device_vectorization: bool,
    include: BTreeSet<PathBuf>,
}

pub struct Program {
    pub bin: Box<[u8]>,
    pub code: String,
    pub log: String,
}

pub struct CompilationError {
    pub result: nvrtcResult,
    pub code: String,
    pub log: String,
}

impl Default for Rtc {
    fn default() -> Self {
        Self::new()
    }
}

impl Rtc {
    pub fn new() -> Self {
        Self {
            cc: Version { major: 8, minor: 0 },
            std: 17,
            line_info: false,
            fast_math: false,
            extra_device_vectorization: false,
            include: Default::default(),
        }
    }

    pub fn arch(mut self, cc: Version) -> Self {
        self.cc = cc;
        self
    }

    pub fn std(mut self, version: usize) -> Self {
        assert!(matches!(version, 3 | 11 | 14 | 17 | 20));
        self.std = version;
        self
    }

    pub fn line_info(mut self, enable: bool) -> Self {
        self.line_info = enable;
        self
    }

    pub fn fast_math(mut self, enable: bool) -> Self {
        self.fast_math = enable;
        self
    }

    pub fn extra_device_vectorization(mut self, enable: bool) -> Self {
        self.extra_device_vectorization = enable;
        self
    }

    pub fn include(mut self, path: impl Into<PathBuf>) -> Self {
        self.include.insert(path.into());
        self
    }

    pub fn compile(&self, code: &str) -> Result<Program, CompilationError> {
        let options = self.generate();
        let extra_includes = extra_includes(code);

        let options = options
            .iter()
            .map(|s| s.as_ref().as_ptr())
            .chain(extra_includes.iter().map(|s| s.as_ptr()))
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
        let code = code.to_string_lossy().to_string();

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
        if result == nvrtcResult::NVRTC_SUCCESS {
            let mut ptx_len = 0;
            nvrtc!(nvrtcGetPTXSize(program, &mut ptx_len));
            let mut bin = vec![0u8; ptx_len].into_boxed_slice();
            nvrtc!(nvrtcGetPTX(program, bin.as_mut_ptr().cast()));
            nvrtc!(nvrtcDestroyProgram(&mut program));
            Ok(Program { bin, code, log })
        } else {
            Err(CompilationError { result, code, log })
        }
    }

    fn generate(&self) -> Vec<CString> {
        let &Self {
            cc,
            std,
            line_info,
            fast_math,
            extra_device_vectorization,
            ref include,
        } = self;
        let mut vec = [
            format!("-arch=compute_{}", cc.to_arch_string()),
            format!("-std=c++{std}"),
        ]
        .map(|s| CString::from_str(&s).unwrap())
        .to_vec();
        if line_info {
            vec.push(c"-lineinfo".into())
        }
        if fast_math {
            vec.push(c"-use_fast_math".into())
        }
        if extra_device_vectorization {
            vec.push(c"-extra-device-vectorization".into())
        }
        vec.extend(
            include
                .iter()
                .map(|dir| CString::new(format!("-I{}", dir.display())).unwrap()),
        );
        vec
    }
}

impl fmt::Debug for CompilationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "rtc failed with {:?}", self.result)?;
        if !self.log.is_empty() {
            writeln!(f, "{}", self.log)
        } else {
            Ok(())
        }
    }
}

fn extra_includes(code: &str) -> Vec<CString> {
    let mut ans = Vec::new();
    fn include_dir(dir: impl fmt::Display) -> CString {
        CString::new(format!("-I{dir}")).unwrap()
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
                ans.extend(
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

    ans.push(include_dir(toolkit.join("include").display()));
    ans
}

#[cfg(nvidia)]
fn clone_cccl() -> PathBuf {
    use std::{env::temp_dir, process::Command, sync::OnceLock};

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
