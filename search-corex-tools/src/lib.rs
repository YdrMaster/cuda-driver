use std::{
    env,
    path::{Path, PathBuf},
};

pub fn find_corex() -> Option<PathBuf> {
    let path = ENVS
        .iter()
        .filter_map(std::env::var_os)
        .next()
        .unwrap_or("/usr/local/corex".into());
    let path = PathBuf::from(path);
    if path.is_dir() {
        Some(path)
    } else {
        None
    }
}

pub fn include_corex(path: impl AsRef<Path>) {
    if env::var_os("DOCS_RS").is_some() || cfg!(doc) {
        return;
    }
    for env in ENVS {
        println!("cargo:rerun-if-env-changed={env}")
    }
    let lib = path.as_ref().join("lib");
    println!("cargo:rustc-link-search={}", lib.display());
    println!("cargo:rustc-link-lib=cuda")
}

const ENVS: &[&str] = &[
    "ILUVATAR_ROOT",
    "ILUVATAR_HOME",
    "ILUVATAR_PATH",
    "COREX_ROOT",
    "COREX_HOME",
    "COREX_PATH",
];
