use super::ReduceMean;
use rand::Rng;

/// 计算 reduceMean 并与 kernel 函数的结果进行比较。
fn check(data: &[f32], result: &[f32], item_len: usize) -> (f64, f64) {
    // &self, x: &DevSlice, y: &DevSlice, item_len: usize
    let row = result.len();
    let col = data.len() / row;
    debug_assert_eq!(data.len(), row * col);

    let mut answer = vec![0.0f32; row];
    for (i, ans) in answer.iter_mut().enumerate() {
        *ans = data[i * col..][..item_len].iter().sum::<f32>() / item_len as f32;
    }
    test_utils::diff(&result, &answer)
}

#[test]
fn padding() {
    const ROW: usize = 256;
    const COL: usize = 711;
    const BLOCK_SIZE: usize = 1024;

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let mut rng = rand::thread_rng();
        let mut x_data = vec![0.0f32; ROW * COL];
        rng.fill(&mut x_data[..]);
        let x = stream.from_slice(&x_data);
        let y = stream.malloc_for::<f32>(ROW);
        let x = x.as_slice(ctx);
        let y = y.as_slice(ctx);

        ReduceMean::new(BLOCK_SIZE, BLOCK_SIZE, ctx).launch(&x, &y, COL, &stream);

        let mut result = vec![0.0f32; ROW];
        y.copy_out(&mut result);
        let (abs_diff, rel_diff) = check(&x_data, &result, COL);
        assert!(abs_diff < 1e-6, "abs_diff: {abs_diff}");
        assert!(rel_diff < 1e-6, "rel_diff: {rel_diff}");
    });
}

#[test]
fn folding() {
    const ROW: usize = 256;
    const COL: usize = 1024;
    const VALID: usize = 711;
    const BLOCK_SIZE: usize = 256;

    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let mut rng = rand::thread_rng();
        let mut x_data = vec![0.0f32; ROW * COL];
        rng.fill(&mut x_data[..]);
        let x = stream.from_slice(&x_data);
        let y = stream.malloc_for::<f32>(ROW);
        let x = x.as_slice(ctx);
        let y = y.as_slice(ctx);

        ReduceMean::new(COL, BLOCK_SIZE, ctx).launch(&x, &y, VALID, &stream);

        let mut result = vec![0.0f32; ROW];
        y.copy_out(&mut result);
        let (abs_diff, rel_diff) = check(&x_data, &result, VALID);
        assert!(abs_diff < 1e-6, "abs_diff: {abs_diff}");
        assert!(rel_diff < 1e-6, "rel_diff: {rel_diff}");
    });
}
