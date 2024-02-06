pub fn block_softmax(buf: &mut [f32], tok_id: usize, seq_len: usize) {
    #[inline]
    fn attention_causual_mask(
        tok_id: usize,
        seq_len: usize,
        pos_id: usize,
        att_len: usize,
    ) -> bool {
        att_len + tok_id >= pos_id + seq_len
    }

    // mask
    let att_len = buf.len();
    buf.iter_mut().enumerate().for_each(|(pos_id, x)| {
        if !attention_causual_mask(tok_id, seq_len, pos_id, att_len) {
            *x = f32::NEG_INFINITY;
        }
    });
    // max
    let max = *buf.iter().max_by(|x, y| x.total_cmp(y)).unwrap();
    // exp + sum
    let sum = buf
        .iter_mut()
        .map(|x| {
            *x = (*x - max).exp();
            *x
        })
        .sum::<f32>();
    // mean
    buf.iter_mut().for_each(|x| *x /= sum);
}

pub fn slice_softmax(buf: &mut [f32], layout: (usize, usize, usize), valid: (usize, usize, usize)) {
    let (total_batch, total_seq_len, buf_len) = layout;
    let (batch, seq_len, att_len) = valid;
    assert!(batch <= total_batch);
    assert!(seq_len <= total_seq_len);
    assert!(att_len <= buf_len);
    assert!(seq_len <= att_len);
    for x in 0..batch {
        for y in 0..seq_len {
            block_softmax(
                &mut buf[(x * total_seq_len + y) * buf_len..][..att_len],
                y,
                seq_len,
            );
        }
    }
}
