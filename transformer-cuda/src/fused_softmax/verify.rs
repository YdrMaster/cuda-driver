use super::FusedSoftmax;
use cuda::CudaDataType;
use rand::Rng;
use transformer::fused_softmax::slice_softmax;

fn test(seq_len: usize, buf_len: usize, att_len: usize, block_size: usize) {
    cuda::init();
    let Some(dev) = cuda::Device::fetch() else {
        return;
    };
    dev.context().apply(|ctx| {
        let stream = ctx.stream();
        let mut rng = rand::thread_rng();
        let mut x_data = vec![0.0f32; seq_len * buf_len];
        rng.fill(&mut x_data[..]);
        let x = stream.from_slice(&x_data);
        let x = x.as_slice(ctx);

        FusedSoftmax::new(CudaDataType::float, buf_len, block_size, ctx).launch(
            &x,
            (1, seq_len, buf_len),
            (1, seq_len, att_len),
            &stream,
        );
        slice_softmax(&mut x_data, (1, seq_len, buf_len), (1, seq_len, att_len));

        let mut result = vec![0.0f32; seq_len * buf_len];
        x.copy_out(&mut result);
        let (abs_diff, rel_diff) = test_utils::diff(&result, &x_data);
        assert!(abs_diff < 1e-6, "abs_diff: {abs_diff}");
        assert!(rel_diff < 1e-6, "rel_diff: {rel_diff}");
    });
}

#[test]
fn padding() {
    test(711, 1024, 711, 1024);
}

#[test]
fn folding() {
    test(3333, 4096, 3333, 1024);
}
