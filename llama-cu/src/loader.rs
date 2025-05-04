use cuda::{DevByte, Event, HostMem, Stream};
use std::{
    borrow::Borrow,
    collections::{BTreeSet, HashMap, VecDeque},
    hash::Hash,
};

pub struct WeightLoader<'ctx> {
    /// 锁页内存的 slab 分配器
    slab: Slab<usize, HostMem<'ctx>>,
    /// 异步加载任务队列
    queue: VecDeque<(Event<'ctx>, HostMem<'ctx>)>,
    /// 使用 slab 分配器的黑名单
    /// slab 分配器的原理导致分配出来的空间会持续存在，不会自动释放
    /// 不常用规模的空间没必要使用分配器
    no_reuse: BTreeSet<usize>,
}

impl<'ctx> WeightLoader<'ctx> {
    pub fn new(no_reuse: impl IntoIterator<Item = usize>) -> Self {
        Self {
            slab: Slab(HashMap::new()),
            queue: VecDeque::new(),
            no_reuse: no_reuse.into_iter().collect(),
        }
    }

    pub fn load(&mut self, dst: &mut [DevByte], src: &[u8], stream: &Stream<'ctx>) {
        // 此次加载的任务规模
        let size = size_of_val(dst);
        // 从 slab 分配器调用
        let mut host = self
            .slab
            .take(&size)
            .unwrap_or_else(|| stream.ctx().malloc_host::<u8>(size));
        // host -> locked -> device
        host.copy_from_slice(src);
        stream.memcpy_h2d(dst, &host);

        if self.no_reuse.contains(&size) {
            // 不使用分配器，先出队后同步等待
            self.free_complete();
            stream.synchronize();
        } else {
            // 使用分配器，先入队再出队
            self.queue.push_back((stream.record(), host));
            self.free_complete()
        }
    }

    fn free_complete(&mut self) {
        while let Some((event, host)) = self.queue.pop_front() {
            if event.is_complete() {
                self.slab.put(host.len(), host)
            } else {
                self.queue.push_front((event, host));
                break;
            }
        }
    }
}

#[repr(transparent)]
struct Slab<K, V>(HashMap<K, Vec<V>>);

impl<K: Eq + Hash, V> Slab<K, V> {
    #[inline]
    pub fn take<Q>(&mut self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
        Q: ?Sized + Hash + Eq,
    {
        self.0.get_mut(key).and_then(|pool| pool.pop())
    }

    #[inline]
    pub fn put(&mut self, key: K, value: V) {
        self.0.entry(key).or_default().push(value);
    }
}
