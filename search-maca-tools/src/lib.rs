use std::{
    env,
    path::{Path, PathBuf},
};

/// Returns the path to the mx home directory, if it is set.
#[inline]
pub fn find_maca_root() -> Option<(MacaType, PathBuf)> {
    if let Some(path) = ENVS.iter().filter_map(std::env::var_os).next() {
        let path = PathBuf::from(path);
        if path.join("htgpu_llvm").is_dir() {
            Some((MacaType::HT, path))
        } else if path.join("mxgpu_llvm").is_dir() {
            Some((MacaType::MX, path))
        } else {
            None
        }
    } else {
        let hpcc = Path::new("/opt/hpcc");
        let mxcc = Path::new("/opt/mxcc");
        if hpcc.is_dir() {
            Some((MacaType::HT, hpcc.into()))
        } else if mxcc.is_dir() {
            Some((MacaType::MX, mxcc.into()))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug)]
#[repr(u8)]
pub enum MacaType {
    MX,
    HT,
}

pub fn include_maca((ty, path): &(MacaType, PathBuf)) {
    if env::var_os("DOCS_RS").is_some() || cfg!(doc) {
        return;
    }

    println!("cargo:rustc-link-search={}", path.join("lib").display());
    let libs = match ty {
        MacaType::MX => ["mcruntime", "mxc-runtime64"],
        MacaType::HT => ["hcruntime", "htc-runtime64"],
    };
    for lib in libs {
        println!("cargo:rustc-link-lib=dylib={lib}")
    }
}

#[test]
fn test() {
    println!("{:?}", find_maca_root())
}

const ENVS: &[&str] = &["MACA_PATH", "HPCC_PATH"];
