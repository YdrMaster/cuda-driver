use std::{env, path::PathBuf, process::Command};

pub struct Builder {
    name: String,
    toolkit: PathBuf,

    cc: String,
    cc_flags: Vec<String>,
    sources: Vec<PathBuf>,

    headers: Vec<PathBuf>,
    include_dirs: Vec<PathBuf>,
    symbols: Vec<String>,
    allow_list: Vec<(AllowListItem, String)>,
}

enum AllowListItem {
    Type,
    Fn,
    Var,
    Item,
}

impl Builder {
    pub fn new(name: impl ToString, toolkit: impl Into<PathBuf>, cc: impl ToString) -> Self {
        Self {
            name: name.to_string(),
            toolkit: toolkit.into(),

            cc: cc.to_string(),
            cc_flags: Default::default(),
            sources: Default::default(),

            headers: Default::default(),
            include_dirs: Default::default(),
            symbols: Default::default(),
            allow_list: Default::default(),
        }
    }

    pub fn cc_flag(mut self, flag: impl ToString) -> Self {
        self.cc_flags.push(flag.to_string());
        self
    }

    pub fn cc_flags<T: ToString>(mut self, flags: impl IntoIterator<Item = T>) -> Self {
        self.cc_flags
            .extend(flags.into_iter().map(|f| f.to_string()));
        self
    }

    pub fn source(mut self, source: impl Into<PathBuf>) -> Self {
        self.sources.push(source.into());
        self
    }

    pub fn sources<T: Into<PathBuf>>(mut self, sources: impl IntoIterator<Item = T>) -> Self {
        self.sources.extend(sources.into_iter().map(T::into));
        self
    }

    pub fn header(mut self, header: impl Into<PathBuf>) -> Self {
        self.headers.push(header.into());
        self
    }

    pub fn headers<T: Into<PathBuf>>(mut self, headers: impl IntoIterator<Item = T>) -> Self {
        self.headers.extend(headers.into_iter().map(T::into));
        self
    }

    pub fn include_dir(mut self, include_dir: impl Into<PathBuf>) -> Self {
        self.include_dirs.push(include_dir.into());
        self
    }

    pub fn include_dirs<T: Into<PathBuf>>(
        mut self,
        include_dirs: impl IntoIterator<Item = T>,
    ) -> Self {
        self.include_dirs
            .extend(include_dirs.into_iter().map(T::into));
        self
    }

    pub fn symbol(mut self, symbol: impl ToString) -> Self {
        self.symbols.push(symbol.to_string());
        self
    }

    pub fn symbols<T: ToString>(mut self, symbols: impl IntoIterator<Item = T>) -> Self {
        self.symbols
            .extend(symbols.into_iter().map(|f| f.to_string()));
        self
    }

    pub fn bind_type(mut self, pattern: impl ToString) -> Self {
        self.allow_list
            .push((AllowListItem::Type, pattern.to_string()));
        self
    }

    pub fn bind_fn(mut self, pattern: impl ToString) -> Self {
        self.allow_list
            .push((AllowListItem::Fn, pattern.to_string()));
        self
    }

    pub fn bind_var(mut self, pattern: impl ToString) -> Self {
        self.allow_list
            .push((AllowListItem::Var, pattern.to_string()));
        self
    }

    pub fn bind_item(mut self, pattern: impl ToString) -> Self {
        self.allow_list
            .push((AllowListItem::Item, pattern.to_string()));
        self
    }

    pub fn compile(self) {
        let Self {
            name,
            toolkit,
            cc,
            cc_flags,
            headers,
            sources,
            include_dirs,
            symbols,
            allow_list,
        } = self;

        let out_dir = PathBuf::from(&env::var_os("OUT_DIR").unwrap());
        let ext = if cfg!(windows) { "dll" } else { "so" };

        if !Command::new(toolkit.join(cc))
            .args(cc_flags)
            .arg("-shared")
            .args(&sources)
            .arg("-o")
            .arg(out_dir.join(format!("lib{name}.{ext}")))
            .status()
            .expect("compiler executable error")
            .success()
        {
            panic!("compile error")
        }

        println!("cargo:rustc-link-search={}", out_dir.display());
        println!("cargo:rustc-link-lib={name}");

        let mut builder = bindgen::Builder::default();
        for header in headers {
            builder = builder.header(header.display().to_string())
        }
        for sym in symbols {
            builder = builder.clang_arg(format!("-D{sym}"))
        }
        for include_dir in include_dirs {
            builder = builder.clang_arg(format!("-I{}", include_dir.display()))
        }
        for src in sources {
            if let Some(dir) = src.parent() {
                builder = builder.clang_arg(format!("-I{}", dir.display()))
            }
        }
        for (ty, pattern) in allow_list {
            builder = match ty {
                AllowListItem::Type => builder.allowlist_type(pattern),
                AllowListItem::Fn => builder.allowlist_function(pattern),
                AllowListItem::Var => builder.allowlist_var(pattern),
                AllowListItem::Item => builder.allowlist_item(pattern),
            }
        }
        builder
            .clang_arg(format!("-I{}", toolkit.join("include").display()))
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: true,
            })
            .use_core()
            .derive_default(true)
            .derive_debug(true)
            .parse_callbacks(Box::new(bindgen::CargoCallbacks::new()))
            .generate()
            .expect("Unable to generate bindings")
            .write_to_file(out_dir.join(format!("{name}_bindings.rs")))
            .expect("Couldn't write bindings!")
    }
}
