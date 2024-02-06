use super::FusedSoftmax;
use cuda::CudaDataType;
use rand::Rng;

fn softmax(x: &mut [f32], buf_len: usize, att_len: usize) {
    fn attention_causual_mask(
        tok_id: usize,
        seq_len: usize,
        pos_id: usize,
        att_len: usize,
    ) -> bool {
        att_len + tok_id >= pos_id + seq_len
    }

    debug_assert_eq!(x.len() % buf_len, 0);
    let seq_len = x.len() / buf_len;
    for tok_id in 0..seq_len {
        let x = &mut x[tok_id * buf_len..][..att_len];
        for pos_id in 0..att_len {
            if !attention_causual_mask(tok_id, seq_len, pos_id, att_len) {
                x[pos_id] = f32::NEG_INFINITY;
            }
        }
        let max = *x.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
        x.iter_mut().for_each(|v| *v = (*v - max).exp());
        let sum = x.iter().sum::<f32>();
        x.iter_mut().for_each(|v| *v /= sum);
    }
}

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

        let mut result = vec![0.0f32; seq_len * buf_len];
        x.copy_out(&mut result);
        softmax(&mut x_data, buf_len, att_len);
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
