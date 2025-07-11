use std::{
    env,
    path::{Path, PathBuf},
};

/// Returns the path to the mx home directory, if it is set.
#[inline]
pub fn find_maca_root() -> Option<PathBuf> {
    env::var_os("MACA_PATH").map(PathBuf::from)
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum MacaType {
    MX,
    HT,
}

pub fn include_maca(path: impl AsRef<Path>) -> Option<MacaType> {
    if env::var_os("DOCS_RS").is_some() || cfg!(doc) {
        return None;
    }
    let path = path.as_ref();
    println!("cargo:rustc-link-search={}", path.join("lib").display());

    let bin = path.join("mxgpu_llvm/bin");
    if bin.join("mxcc").is_file() {
        println!("cargo:rustc-link-lib=dylib=mcruntime");
        println!("cargo:rustc-link-lib=dylib=mxc-runtime64");
        Some(MacaType::MX)
    } else if bin.join("hpcc").is_file() {
        println!("cargo:rustc-link-lib=dylib=hcruntime");
        println!("cargo:rustc-link-lib=dylib=htc-runtime64");
        Some(MacaType::HT)
    } else {
        panic!()
    }
}

#[test]
fn test() {
    println!("{:?}", find_maca_root())
}
