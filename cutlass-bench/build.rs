fn main() {
    use build_script_cfg::Cfg;
    use search_cuda_tools::find_cuda_root;

    println!("cargo:rerun-if-changed=build.rs");

    let cuda = Cfg::new("detected_cuda");
    if find_cuda_root().is_some() {
        cuda.define();
    };
}
