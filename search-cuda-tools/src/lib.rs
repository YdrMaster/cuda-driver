use std::{
    env::{split_paths, var_os},
    fs,
    path::PathBuf,
    process::Command,
};

pub use find_cuda_helper::{find_cuda_root, include_cuda};

pub fn find_nccl_root() -> Option<Option<PathBuf>> {
    if !cfg!(target_os = "linux") {
        return None;
    }
    let output = Command::new("ldconfig").arg("-p").output().ok()?;
    if unsafe { String::from_utf8_unchecked(output.stdout) }.contains("libnccl.so") {
        Some(None)
    } else {
        split_paths(&var_os("LIBRARY_PATH")?)
            .chain(split_paths(&var_os("LD_LIBRARY_PATH")?))
            .filter_map(|path| fs::read_dir(path).ok())
            .flatten()
            .filter_map(|result| result.ok())
            .find(|entry| entry.file_name() == "libnccl.so")?
            .path()
            .parent()?
            .parent()
            .filter(|root| root.join("include/nccl.h").is_file())
            .map(|path| Some(path.into()))
    }
}

#[test]
fn test_find() {
    let Some(root) = find_cuda_root() else {
        println!("cuda not exist");
        return;
    };
    println!("cuda root = {}", root.display());
    let Some(nccl) = find_nccl_root() else {
        println!("nccl not exist");
        return;
    };
    let Some(nccl) = nccl else {
        println!("find nccl in ldconfig path");
        return;
    };
    println!("nccl root = {nccl:?}")
}
