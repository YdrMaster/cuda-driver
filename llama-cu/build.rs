use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    use find_cuda_helper::{find_cuda_root, include_cuda};
    // 获取输出目录
    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);

    // 获取cuda根目录
    let cuda_root = find_cuda_root().unwrap();

    include_cuda();
    // 获取当前目录
    let current_dir = env::current_dir().unwrap();

    // 定义CUDA源文件目录路径，避免重复
    let cuda_src_dir = "src/op/attention_kv-c";
    let cuda_src_dir_path = current_dir.join(cuda_src_dir);

    // CUDA源文件路径
    let cuda_src_path = cuda_src_dir_path.join("attention_kv.c");
    let header_path = cuda_src_dir_path.join("attention_kv.h");

    // 输出库文件路径 - 更改为明确的库名称
    let lib_name = "attention_kv";
    let lib_output_path = out_path.join(format!("lib{}.so", lib_name));

    // 运行nvcc编译命令
    let status = Command::new(cuda_root.join("bin/nvcc"))
        .arg(cuda_src_path)
        .arg("-Xcompiler")
        .arg("-fPIC")
        .arg("-shared")
        .arg("-o")
        .arg(&lib_output_path)
        .status()
        .expect("无法执行nvcc命令");

    if !status.success() {
        panic!("nvcc编译失败");
    }

    // 重新运行构建脚本的条件
    println!("cargo:rerun-if-changed={}/attention_kv.c", cuda_src_dir);
    // 打印cargo相关信息，告诉Rust编译器动态库的位置
    println!("cargo:rustc-link-search=native={}", out_dir);
    // 使用正确的库名称
    println!("cargo:rustc-link-lib=dylib={}", lib_name);

    // 使用bindgen生成Rust绑定
    let bindings = bindgen::Builder::default()
        // 输入头文件
        .header(header_path.to_str().unwrap())
        .clang_arg(format!("-I{}", cuda_src_dir_path.to_str().unwrap()))
        // 告诉bindgen在哪里可以找到CUDA头文件
        .clang_arg(format!("-I{}", cuda_root.join("include").to_str().unwrap()))
        // 只生成attention_kv.h中的特定函数
        .allowlist_function("launch_attention_kv")
        // 生成Rust风格的枚举
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        // 使用derive特性
        .derive_default(true)
        .derive_debug(true)
        // 处理cargo回调
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
        // 生成绑定
        .generate()
        // 处理错误
        .expect("无法生成绑定");

    // 将绑定写入指定的文件
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("无法写入绑定");
}
