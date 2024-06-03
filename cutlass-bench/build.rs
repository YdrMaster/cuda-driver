fn main() {
    use search_cuda_tools::{find_cuda_root, Cfg};

    let cuda = Cfg::new("cuda");
    if find_cuda_root().is_some() {
        cuda.detect();
    };
}
