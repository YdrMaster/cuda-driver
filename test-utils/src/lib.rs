use num_traits::Float;

pub fn diff<T: Float>(result: &[T], ans: &[T]) -> (f64, f64) {
    assert_eq!(result.len(), ans.len());
    let mut max_abs_diff = 0.;
    let mut up = 0.;
    let mut down = 0.;
    for (r, a) in result.iter().zip(ans) {
        let r = r.to_f64().unwrap();
        let a = a.to_f64().unwrap();
        let diff = (r - a).abs();
        max_abs_diff = max_abs_diff.max(diff);
        up += diff;
        down += a.abs();
    }
    (max_abs_diff, up / down)
}
