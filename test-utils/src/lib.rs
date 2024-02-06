use num_traits::Float;
use std::{fmt, io::Write};

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

#[macro_export]
macro_rules! slice {
    ($blob:expr; $width:expr; [$line:expr]) => {
        $blob[$line * $width..][..$width]
    };
}

pub fn write_tensor<T: fmt::LowerExp>(
    to: &mut impl Write,
    buf: &[T],
    shape: &[usize],
) -> std::io::Result<()> {
    match shape {
        [] => {
            writeln!(to, "<>")?;
            write_matrix(to, buf, (1, 1))
        }
        [len] => {
            writeln!(to, "<{len}>")?;
            write_matrix(to, buf, (*len, 1))
        }
        [rows, cols] => {
            writeln!(to, "<{rows}x{cols}>")?;
            write_matrix(to, buf, (*rows, *cols))
        }
        [batch @ .., rows, cols] => {
            let mut strides = vec![1usize; batch.len()];
            for i in (1..batch.len()).rev() {
                strides[i - 1] = strides[i] * batch[i];
            }
            let strides = strides.as_slice();
            for i in 0..batch[0] * strides[0] {
                let mut which = vec![0usize; strides.len()];
                let mut rem = i;
                for (j, &stride) in strides.iter().enumerate() {
                    which[j] = rem / stride;
                    rem %= stride;
                }
                writeln!(
                    to,
                    "<{rows}x{cols}>[{}]",
                    which
                        .iter()
                        .map(usize::to_string)
                        .collect::<Vec<_>>()
                        .join(", "),
                )?;
                write_matrix(to, &slice!(buf; rows * cols; [i]), (*rows, *cols))?;
            }
            Ok(())
        }
    }
}

fn write_matrix<T: fmt::LowerExp>(
    to: &mut impl Write,
    buf: &[T],
    shape: (usize, usize),
) -> std::io::Result<()> {
    let (rows, cols) = shape;
    for r in 0..rows {
        let row = &slice!(buf; cols; [r]);
        for it in row {
            write!(to, "{it:.3e} ")?;
        }
        writeln!(to)?;
    }
    Ok(())
}

#[test]
fn test_log() {
    let array = [
        1., 2., 3., //
        4., 5., 6., //
        7., 8., 9., //
        10., 11., 12., //
    ];
    write_tensor(&mut std::io::stdout(), &array, &[2, 2, 3]).unwrap();
}
