﻿fn main() {
    use std::{env, path::PathBuf};

    let Some(cuda_root) = search_cuda_tools::find_cuda_root() else {
        return;
    };
    let Some(nccl_root) = search_cuda_tools::find_nccl_root() else {
        return;
    };
    search_cuda_tools::detect_nccl();
    search_cuda_tools::include_cuda();

    let mut includes = vec![format!("-I{}/include", cuda_root.display())];
    if let Some(nccl_root) = nccl_root {
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
