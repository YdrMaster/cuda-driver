fn main() {
    use search_cuda_tools::{find_cuda_root, include_cuda, Cfg};
    use std::{env, path::PathBuf};

    let cuda = Cfg::new("cuda");
    let Some(cuda_root) = find_cuda_root() else {
        return;
    };
    cuda.detect();
    include_cuda();

    println!("cargo:rustc-link-lib=dylib=nvrtc");
    println!("cargo:rustc-env=CUDA_ROOT={}", cuda_root.display());

    // Tell cargo to invalidate the built crate whenever the wrapper changes.
    println!("cargo:rerun-if-changed=wrapper.h");

    // The bindgen::Builder is the main entry point to bindgen,
    // and lets you build up options for the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate bindings for.
        .header("wrapper.h")
        .clang_arg(format!("-I{}/include", cuda_root.display()))
        // Only generate bindings for the functions in these namespaces.
        .allowlist_function("cu.*")
        .allowlist_function("nvrtc.*")
        .allowlist_item("CU.*")
        // Annotate the given type with the #[must_use] attribute.
        .must_use_type("CUresult")
        .must_use_type("nvrtcResult")
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
