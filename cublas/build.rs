use build_script_cfg::Cfg;
use find_cuda_helper::{find_cuda_root, include_cuda};
use search_corex_tools::{find_corex, include_corex};
use search_maca_tools::{find_maca_root, include_maca};
use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let nvidia = Cfg::new("nvidia");
    let iluvatar = Cfg::new("iluvatar");
    let metax = Cfg::new("metax");

    if let Some(pair) = find_maca_root() {
        metax.define();
        include_maca(&pair);
        bind_maca(pair.1)
    } else if let Some(corex) = find_corex() {
        iluvatar.define();
        include_corex(&corex);
        bind_cuda(corex, true)
    } else if let Some(cuda_root) = find_cuda_root() {
        nvidia.define();
        include_cuda();
        bind_cuda(cuda_root, false)
    }
}

fn bind_cuda(toolkit: impl AsRef<Path>, _x_cxx: bool) {
    let toolkit = toolkit.as_ref();
    println!("cargo:rustc-link-lib=dylib=cublas");
    println!("cargo:rustc-link-lib=dylib=cublasLt");

    // Tell cargo to invalidate the built crate whenever the wrapper changes.
    println!("cargo:rerun-if-changed=wrapper.h");
    let include = toolkit.join("include");
    // The bindgen::Builder is the main entry point to bindgen,
    // and lets you build up options for the resulting bindings.
    let mut builder = bindgen::Builder::default();
    builder = builder
        // The input header we would like to generate bindings for.
        .header("wrapper.h")
        .clang_arg(format!("-I{}", include.display()))
        // Only generate bindings for the functions in these namespaces.
        .allowlist_function("cublas.*")
        .allowlist_item("cublas.*")
        // Annotate the given type with the #[must_use] attribute.
        .must_use_type("cublasStatus_t")
        // Generate rust style enums.
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        // Use core instead of std in the generated bindings.
        .use_core()
        // Tell cargo to invalidate the built crate whenever any of the included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    if _x_cxx {
        builder = builder.clang_args(["-x", "c++"])
    }
    let bindings = builder
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}

fn bind_maca(toolkit: impl AsRef<Path>) {
    let toolkit = toolkit.as_ref();
    println!("cargo:rustc-link-lib=dylib=hcblas");
    println!("cargo:rustc-link-lib=dylib=hcblasLt");

    println!("cargo:rustc-link-lib=dylib=hcruntime");
    println!("cargo:rustc-link-lib=dylib=htc-runtime64");
    // Tell cargo to invalidate the built crate whenever the wrapper changes.
    println!("cargo:rerun-if-changed=wrapper_maca.h");

    // The bindgen::Builder is the main entry point to bindgen,
    // and lets you build up options for the resulting bindings.
    let mut builder = bindgen::Builder::default();
    builder = builder
        // The input header we would like to generate bindings for.
        .header("wrapper_maca.h")
        .clang_arg(format!("-I{}", toolkit.join("include").display()))
        // .clang_arg("-x hpcc")
        .clang_args(["-x", "c++"])
        // Only generate bindings for the functions in these namespaces.
        .allowlist_function("hcblas.*")
        .allowlist_item("hcblas.*")
        // Annotate the given type with the #[must_use] attribute.
        .must_use_type("hcblasStatus_t")
        // Generate rust style enums.
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        // Use core instead of std in the generated bindings.
        .use_core()
        // Tell cargo to invalidate the built crate whenever any of the included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()));
    let bindings = builder
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
