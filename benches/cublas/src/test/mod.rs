mod bench;
mod verify;

use cuda::{DevicePtr, Stream};
use rand::Rng;
use std::alloc::Layout;

const M: usize = 5376;
const K: usize = 2048;
const N: usize = 256;
const ALPHA: f32 = 1.;
const BETA: f32 = 0.;
const TIMES: usize = 1000;

fn rand_blob(len: usize, stream: &Stream) -> DevicePtr {
    let mut rng = rand::thread_rng();
    let mut mem: Vec<f32> = vec![0.; len];
    rng.fill(&mut mem[..]);
    let size = Layout::array::<f32>(mem.len()).unwrap().size();
    let mut ans = stream.malloc(size);
    unsafe { ans.copy_in_async(&mem, stream) };
    ans
}

fn uninit_blob(len: usize, stream: &Stream) -> DevicePtr {
    let size = Layout::array::<f32>(len).unwrap().size();
    stream.malloc(size)
}
