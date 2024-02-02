mod bench;
mod verify;

use cuda::{DevBlob, Stream};
use rand::Rng;

const M: usize = 5376;
const K: usize = 2048;
const N: usize = 256;
const ALPHA: f32 = 1.;
const BETA: f32 = 0.;
const TIMES: usize = 1000;

fn rand_blob(len: usize, stream: &Stream) -> DevBlob {
    let mut rng = rand::thread_rng();
    let mut mem = vec![0.0f32; len];
    rng.fill(&mut mem[..]);
    stream.from_slice(&mem)
}
