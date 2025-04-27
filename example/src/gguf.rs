﻿use crate::blob::Data;
use ggus::{
    GENERAL_ALIGNMENT, GGuf, GGufError, GGufFileName, GGufMetaDataValueType, GGufMetaKV,
    GGufMetaMap, ggml_quants::digit_layout::DigitLayout,
};
use memmap2::Mmap;
use std::{collections::HashMap, fmt::Debug, fs::File, path::Path, thread};

/// 从指定文件的路径出发，映射所有分片文件。
pub fn map_files(path: impl AsRef<Path>) -> Box<[Mmap]> {
    fn throw(path: &Path, e: impl Debug) -> ! {
        let path = path.display();
        panic!(
            "\
Error occurred at path: {path}
  error: {e:?}"
        )
    }

    #[inline]
    fn map_file(path: &Path) -> Mmap {
        let file = File::open(path).unwrap_or_else(|e| throw(path, e));
        unsafe { Mmap::map(&file) }.unwrap()
    }

    let path = path.as_ref();
    let name = GGufFileName::try_from(path).unwrap_or_else(|e| throw(path, e));

    if name.shard_count() == 1 {
        Box::new([map_file(path)])
    } else {
        let dir = path.parent().unwrap();
        name.iter_all()
            .map(|name| map_file(&dir.join(name.to_string())))
            .collect()
    }
}

/// GGuf 模型，可能来自多个分片文件。
pub struct GGufModel<'a> {
    /// 元数据键值对。
    pub meta_kvs: HashMap<&'a str, GGufMetaKV<'a>>,
    /// 张量。
    pub tensors: HashMap<&'a str, GGufTensor<'a>>,
}

/// GGuf 张量。
pub struct GGufTensor<'a> {
    pub ty: DigitLayout,
    pub shape: Box<[usize]>,
    pub data: Data<'a>,
}

impl<'a> GGufModel<'a> {
    /// 从多个分片文件中读取 GGuf 模型。
    pub fn read(files: impl IntoIterator<Item = &'a [u8]> + 'a) -> Self {
        let mut ans = Self {
            meta_kvs: Default::default(),
            tensors: Default::default(),
        };
        thread::scope(|s| {
            for (i, thread) in files
                .into_iter()
                .map(|data| s.spawn(|| GGuf::new(data)))
                .collect::<Vec<_>>()
                .into_iter()
                .enumerate()
            {
                thread
                    .join()
                    .unwrap()
                    .and_then(|gguf| ans.merge(gguf))
                    .unwrap_or_else(|e| panic!("Error at file {i}: {e}"));
            }
        });
        ans
    }

    fn merge(&mut self, gguf: GGuf<'a>) -> Result<(), GGufError> {
        for (k, kv) in gguf.meta_kvs {
            if k == GENERAL_ALIGNMENT || k.starts_with("split.") {
                continue;
            }
            if self.meta_kvs.insert(k, kv).is_some() {
                return Err(GGufError::DuplicateMetaKey(k.into()));
            }
        }

        for (name, t) in gguf.tensors {
            use std::collections::hash_map::Entry::{Occupied, Vacant};
            match self.tensors.entry(name) {
                Occupied(_) => return Err(GGufError::DuplicateTensorName(name.into())),
                Vacant(vacant) => {
                    let t = t.to_info();
                    vacant.insert(GGufTensor {
                        ty: t.ty().to_digit_layout(),
                        shape: t.shape().iter().rev().map(|&x| x as usize).collect(),
                        data: gguf.data[t.offset()..][..t.nbytes()].into(),
                    });
                }
            }
        }

        Ok(())
    }
}

impl GGufMetaMap for GGufModel<'_> {
    fn get(&self, key: &str) -> Option<(GGufMetaDataValueType, &[u8])> {
        self.meta_kvs.get(key).map(|kv| (kv.ty(), kv.value_bytes()))
    }
}

mod macros {
    #[macro_export]
    macro_rules! meta {
        ($gguf:expr => $key:ident) => {
            $gguf.$key().unwrap()
        };
        ($gguf:expr => $key:ident; $default:expr) => {
            match $gguf.$key() {
                Ok(val) => val,
                Err(ggus::GGufMetaError::NotExist) => $default,
                Err(e) => panic!("failed to read meta: {e:?}"),
            }
        };

        ($gguf:expr => (usize) $key:expr) => {
            $gguf.get_usize($key).unwrap()
        };
        ($gguf:expr => (usize) $key:expr; $default:expr) => {
            match $gguf.get_usize($key) {
                Ok(val) => val,
                Err(ggus::GGufMetaError::NotExist) => $default,
                Err(e) => panic!("failed to read meta: {e:?}"),
            }
        };
    }
    #[macro_export]
    macro_rules! tensor {
        ($gguf:expr => $name:expr) => {
            &$gguf.tensors[&*$name]
        };
    }
}
