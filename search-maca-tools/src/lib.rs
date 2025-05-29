use std::{
    env,
    path::{Path, PathBuf},
};

/// Returns the path to the mx home directory, if it is set.
#[inline]
pub fn find_maca_root() -> Option<PathBuf> {
    env::var_os("MACA_PATH").map(PathBuf::from)
}

pub fn include_maca(path: impl AsRef<Path>) {
    if env::var_os("DOCS_RS").is_some() || cfg!(doc) {
        return;
    }
    let lib = path.as_ref().join("lib");
    println!("cargo:rustc-link-search={}", lib.display());
    println!("cargo:rustc-link-lib=dylib=hcruntime");
    println!("cargo:rustc-link-lib=dylib=htc-runtime64")
}

#[test]
fn test() {
    println!("{:?}", find_maca_root())
}
