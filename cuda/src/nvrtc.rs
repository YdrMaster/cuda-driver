use crate::{bindings as cuda, Context, ContextGuard};
use std::{
    collections::{hash_map::Keys, HashMap},
    ffi::{c_char, CStr, CString},
    path::PathBuf,
    ptr::{null, null_mut},
    sync::{Arc, Mutex, OnceLock},
};

static MODULES: OnceLock<Mutex<HashMap<String, Arc<Module>>>> = OnceLock::new();

pub fn compile<'a, I, U, V>(code: &str, symbols: I, ctx: &ContextGuard)
where
    I: IntoIterator<Item = (U, V)>,
    U: AsRef<str>,
    V: AsRef<str>,
{
    let symbols = symbols
        .into_iter()
        .map(|(k, v)| (k.as_ref().to_owned(), v.as_ref().to_owned()))
        .collect::<HashMap<_, _>>();
    // 先检查一遍并确保静态对象创建
    let modules = if let Some(modules) = MODULES.get() {
        if check_hold(&modules.lock().unwrap(), symbols.keys()) {
            return;
        }
        modules
    } else {
        MODULES.get_or_init(Default::default)
    };
    // 编译
    let (module, log) = Module::from_src(code, ctx);
    println!("{log}");
    // 再上锁检查一遍
    let module = Arc::new(module.unwrap());
    let mut map = modules.lock().unwrap();
    if !check_hold(&map, symbols.keys()) {
        for k in symbols.keys() {
            // 确认指定的符号都存在
            module.get_function(k);
            map.insert(k.clone(), module.clone());
        }
    }
}

pub fn get_function(name: &str) -> Option<cuda::CUfunction> {
    MODULES.get().and_then(|modules| {
        modules
            .lock()
            .unwrap()
            .get(name)
            .map(|module| module.get_function(name))
    })
}

fn check_hold(map: &HashMap<String, Arc<Module>>, symbols: Keys<'_, String, String>) -> bool {
    let len = symbols.len();
    let had = symbols.filter(|&k| map.contains_key(k)).count();
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
            CString::new(format!("-I{}/include", std::env!("CUDA_ROOT"))).unwrap(),
        ];
        {
            let cccl = std::option_env!("CCCL_ROOT").map_or_else(
                || PathBuf::from(std::env!("CARGO_MANIFEST_DIR")).join("cuda/cccl"),
                |dir| PathBuf::from(dir),
            );
            let cudacxx = cccl.join("libcudacxx/include");
            let cub = cccl.join("cub");
            assert!(cudacxx.is_dir(), "cudacxx not exist");
            assert!(cub.is_dir(), "cub not exist");
            options.push(CString::new(format!("-I{}\n", cudacxx.display())).unwrap());
            options.push(CString::new(format!("-I{}\n", cub.display())).unwrap());
        }
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
    let proj = std::env!("CARGO_MANIFEST_DIR");
    println!("proj = \"{}\"", proj);
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

        driver!(cuLaunchKernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            null_mut(),
            null_mut(),
            null_mut()
        ));
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
        let function = module.get_function("kernel");

        driver!(cuLaunchKernel(
            function,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            null_mut(),
            null_mut(),
            null_mut()
        ));
        ctx.synchronize();
    });
}
