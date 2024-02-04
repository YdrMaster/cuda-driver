use crate::{bindings as cuda, launch::KernelFn, Context, ContextGuard};
use std::{
    collections::HashMap,
    ffi::{c_char, CStr, CString},
    path::PathBuf,
    ptr::{null, null_mut},
    sync::{Arc, Mutex, OnceLock},
};

static MODULES: OnceLock<Mutex<HashMap<String, Arc<Module>>>> = OnceLock::new();

impl ContextGuard<'_> {
    pub fn compile(&self, code: impl AsRef<str>) {
        let symbols = {
            let mut symbols = Vec::new();
            let mut code = code.as_ref();
            while let Some(pos) = code.find("extern \"C\"") {
                let seg = &code[pos..];
                let (head, tail) = seg.split_at(seg.find('(').unwrap());
                let name = head.split_whitespace().last().unwrap();
                code = tail;
                symbols.push(name);
            }
            symbols
        };
        assert!(!symbols.is_empty(), "no symbol found");
        // 先检查一遍并确保静态对象创建
        let modules = if let Some(modules) = MODULES.get() {
            if check_hold(&modules.lock().unwrap(), &symbols) {
                return;
            }
            modules
        } else {
            MODULES.get_or_init(Default::default)
        };
        // 编译
        let (module, log) = Module::from_src(code.as_ref(), self);
        println!("{log}");
        // 再上锁检查一遍
        let module = Arc::new(module.unwrap());
        let mut map = modules.lock().unwrap();
        if !check_hold(&map, &symbols) {
            for k in symbols {
                // 确认指定的符号都存在
                module.get_function(k);
                map.insert(k.to_string(), module.clone());
            }
        }
    }
}

impl KernelFn {
    pub fn get(name: &str) -> Option<Self> {
        MODULES.get().and_then(|modules| {
            modules
                .lock()
                .unwrap()
                .get(name)
                .map(|module| Self(module.get_function(name)))
        })
    }
}

fn check_hold(map: &HashMap<String, Arc<Module>>, symbols: &[&str]) -> bool {
    let len = symbols.len();
    let had = symbols.iter().filter(|&&k| map.contains_key(k)).count();
    if had == len {
        true
    } else if had == 0 {
        false
    } else {
        panic!()
    }
}

struct Module {
    ctx: Arc<Context>,
    module: cuda::CUmodule,
}

unsafe impl Send for Module {}
unsafe impl Sync for Module {}

impl Drop for Module {
    #[inline]
    fn drop(&mut self) {
        driver!(cuModuleUnload(self.module));
    }
}

impl Module {
    fn from_src(code: &str, ctx: &ContextGuard) -> (Result<Self, cuda::nvrtcResult>, String) {
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
        {
            let cccl = std::option_env!("CCCL_ROOT").map_or_else(
                || PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("cccl"),
                PathBuf::from,
            );
            let cudacxx = cccl.join("libcudacxx/include");
            let cub = cccl.join("cub");
            assert!(cudacxx.is_dir(), "cudacxx not exist");
            assert!(cub.is_dir(), "cub not exist");
            options.push(CString::new(format!("-I{}\n", cudacxx.display())).unwrap());
            options.push(CString::new(format!("-I{}\n", cub.display())).unwrap());
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
                String::from_utf8(log).unwrap()
            } else {
                String::new()
            }
        };
        if result != cuda::nvrtcResult::NVRTC_SUCCESS {
            return (Err(result), log);
        }

        let ptx = {
            let mut ptx_len = 0;
            nvrtc!(nvrtcGetPTXSize(program, &mut ptx_len));
            let mut ptx = vec![0u8; ptx_len];
            nvrtc!(nvrtcGetPTX(program, ptx.as_mut_ptr().cast()));
            nvrtc!(nvrtcDestroyProgram(&mut program));
            ptx
        };
        let ptx = CStr::from_bytes_with_nul(ptx.as_slice()).unwrap();

        let mut module = null_mut();
        driver!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));
        (
            Ok(Self {
                ctx: ctx.clone_ctx(),
                module,
            }),
            log,
        )
    }

    #[inline]
    fn get_function(&self, name: &str) -> cuda::CUfunction {
        let name = CString::new(name).unwrap();
        let mut func = null_mut();
        self.ctx
            .apply(|_| driver!(cuModuleGetFunction(&mut func, self.module, name.as_ptr())));
        func
    }
}

#[test]
fn test_env() {
    let cuda_root = std::env!("CUDA_ROOT");
    assert!(!cuda_root.is_empty());
    // println!("cuda root = \"{}\"", cuda_root);
}

#[test]
fn test_behavior() {
    const SRC: &str = r#"
extern "C" __global__ void kernel() {
    printf("Hello World from GPU!\n");
}
"#;

    crate::init();
    let Some(dev) = crate::Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        let code = CString::new(SRC).unwrap();
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

        let ptx = {
            let mut ptx_len = 0;
            nvrtc!(nvrtcGetPTXSize(program, &mut ptx_len));
            let mut ptx = vec![0u8; ptx_len];
            nvrtc!(nvrtcGetPTX(program, ptx.as_mut_ptr().cast()));
            nvrtc!(nvrtcDestroyProgram(&mut program));
            ptx
        };

        let ptx = CStr::from_bytes_with_nul(ptx.as_slice()).unwrap();
        let mut module = null_mut();
        driver!(cuModuleLoadData(&mut module, ptx.as_ptr().cast()));

        let name = CString::new("kernel").unwrap();
        let mut function = null_mut();
        driver!(cuModuleGetFunction(&mut function, module, name.as_ptr()));
        KernelFn(function).launch((), (), null_mut(), 0, None);
        ctx.synchronize();
    });
}

#[test]
fn test_module() {
    const SRC: &str = r#"
extern "C" __global__ void kernel() {
    printf("Hello World from GPU!\n");
}
"#;

    crate::init();
    let Some(dev) = crate::Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        let (module, _log) = Module::from_src(SRC, ctx);
        let module = module.unwrap();
        KernelFn(module.get_function("kernel")).launch((), (), null_mut(), 0, None);
        ctx.synchronize();
    });
}
