//! Debug helpers for dumping layer outputs as .npy files.
//!
//! Controlled by the ``DUMP_LAYERS`` environment variable — when set to a
//! directory path, forward pass outputs are written as ``func_{n}_{m}.npy``.

use crate::tensor::Tensor;

/// Write intermediate layer outputs to .npy files when `DUMP_LAYERS` env is set.
///
/// Skips outputs containing NaN (uninitialized buffer), all-identical values
/// (possible read bug), or empty slices. Each output file is named
/// ``func_{fi}_{oi}.npy`` under the `DUMP_LAYERS` directory.
pub fn dump_layers(func_outputs: &[Vec<Tensor>]) {
    let Ok(dump_dir) = std::env::var("DUMP_LAYERS") else { return };
    let _ = std::fs::create_dir_all(&dump_dir);
    for (fi, outputs) in func_outputs.iter().enumerate() {
        for (oi, t) in outputs.iter().enumerate() {
            let path = format!("{}/func_{}_{}.npy", dump_dir, fi, oi);
            let slice = t.as_slice();
            if !slice.is_empty() {
                let has_nan = slice.iter().any(|&x| x.is_nan());
                if has_nan {
                    log::warn!(
                        "DUMP_LAYERS: func[{}] output[{}] contains NaN — \
                         possible uninitialized buffer or dynamic shape sret issue",
                        fi, oi,
                    );
                }
                let all_same = slice.iter().all(|&x| x == slice[0]);
                if all_same {
                    log::warn!(
                        "DUMP_LAYERS: func[{}] output[{}] has ALL IDENTICAL \
                         values ({}) — possible read bug",
                        fi, oi, slice[0],
                    );
                }
                if has_nan || all_same {
                    continue;
                }
            } else {
                continue;
            }
            let _ = write_npy(&path, slice, &t.shape);
        }
    }
}

fn write_npy(path: &str, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;
    let shape_str = shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ");
    let header = if shape.is_empty() {
        "{'descr': '<f4', 'fortran_order': False, 'shape': (), }".to_string()
    } else if shape.len() == 1 {
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}", shape_str)
    } else {
        format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}", shape_str)
    };
    let header_bytes = header.as_bytes();
    let header_len = header_bytes.len() as u16;
    let padding = (64 - ((10 + header_bytes.len()) % 64)) % 64;
    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1, 0])?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_bytes)?;
    for _ in 0..padding { file.write_all(b" ")?; }
    // SAFETY: `data` is a valid Vec<f32>; `size_of_val(data)` gives the
    // exact byte count, so the raw pointer cast produces a valid &[u8].
    let byte_slice = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data))
    };
    file.write_all(byte_slice)?;
    Ok(())
}
