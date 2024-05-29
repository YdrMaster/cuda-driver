fn main() {
    use search_cuda_tools::{allow_cfg, detect, find_cuda_root};

    allow_cfg("cuda");
    if find_cuda_root().is_some() {
        detect("cuda");
    };
}
