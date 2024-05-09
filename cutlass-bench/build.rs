fn main() {
    if search_cuda_tools::find_cuda_root().is_some() {
        search_cuda_tools::detect_cuda();
    };
}
