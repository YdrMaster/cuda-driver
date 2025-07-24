use build_script_cfg::Cfg;
use search_corex_tools::find_corex;
use search_cuda_tools::{find_cuda_root, find_nccl_root};
use search_maca_tools::{find_maca_root, include_maca};
use std::{
    env,
    path::{Path, PathBuf},
};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let nccl = Cfg::new("detected_nccl");
    if let Some(pair) = find_maca_root() {
        nccl.define();
        include_maca(&pair);
        bind_hccl(pair.1)
    } else if let Some(_toolkit) = find_corex() {
        // TODO
        // nccl.define();
        // bind(toolkit, None)
    } else if let Some(toolkit) = find_cuda_root() {
        if let Some(nccl_) = find_nccl_root() {
            nccl.define();
            bind(toolkit, nccl_)
        }
    }
}

fn bind(toolkit: impl AsRef<Path>, nccl: Option<PathBuf>) {
    let mut includes = vec![format!("-I{}/include", toolkit.as_ref().display())];
    if let Some(nccl_root) = nccl {
        let nccl_root = nccl_root.display();
        includes.push(format!("-I{nccl_root}/include"));
        println!("cargo:rustc-link-search={nccl_root}/lib");
    }

    println!("cargo:rustc-link-lib=dylib=nccl");

    // Tell cargo to invalidate the built crate whenever the wrapper changes.
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point to bindgen,
    // and lets you build up options for the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for.
        .header("wrapper.h")
        .clang_args(&includes)
        // Only generate bindings for the functions in these namespaces.
        .allowlist_function("nccl.*")
        .allowlist_item("nccl.*")
        // Annotate the given type with the #[must_use] attribute.
        .must_use_type("ncclResult_t")
        // Generate rust style enums.
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        // Use core instead of std in the generated bindings.
        .use_core()
        // Tell cargo to invalidate the built crate whenever any of the included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
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

fn bind_hccl(maca: impl AsRef<Path>) {
    println!("cargo:rustc-link-lib=dylib=hccl");
    println!("cargo:rustc-link-lib=dylib=hcruntime");
    println!("cargo:rustc-link-lib=dylib=htc-runtime64");
    // Tell cargo to invalidate the built crate whenever the wrapper changes.
    println!("cargo:rerun-if-changed=wrapper_maca.h");

    // The bindgen::Builder is the main entry point to bindgen,
    // and lets you build up options for the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for.
        .header("wrapper_maca.h")
        // .clang_args(&includes)
        .clang_arg(format!("-I{}", maca.as_ref().join("include").display()))
        .clang_arg("-x")
        .clang_arg("c++")
        // Only generate bindings for the functions in these namespaces.
        // .clang_arg("-x hpcc")
        .allowlist_function("hccl.*")
        .allowlist_item("hccl.*")
        // Annotate the given type with the #[must_use] attribute.
        .must_use_type("hcclResult_t")
        // Generate rust style enums.
        .default_enum_style(bindgen::EnumVariation::Rust {
            non_exhaustive: true,
        })
        // Use core instead of std in the generated bindings.
        .use_core()
        // Tell cargo to invalidate the built crate whenever any of the included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
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
