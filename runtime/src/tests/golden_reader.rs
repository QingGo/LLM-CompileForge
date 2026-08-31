//! Golden .npz reader — reads float32 tensor dicts from .npz files.
//!
//! Contract: Shared between compiler and runtime sub-projects.
//!   .npz key = output name (e.g. "output_0", "func_main_1_output")
//!   .npz value = float32 data
//!
//! Parses .npy format v1.0/2.0 manually — no numpy dependency.
//! .npy layout (v1.0):
//!   [magic:6][version:2][header_len:u16][header:dict_literal+padded][data:f32...]
//!
//! Tests:
//!   cargo test golden_reader --lib
//!   (write_test_npz.py must be run first to generate test data)

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

/// Magic bytes that start every .npy file.
const NPY_MAGIC: &[u8] = b"\x93NUMPY";

/// Read an .npz file (zip of .npy entries) into a HashMap of {name -> Vec<f32>}.
///
/// Key names are extracted from the zip entry name (e.g. "output_0.npy" → "output_0").
/// All entries must be float32 arrays.
pub fn read_npz(path: impl AsRef<Path>) -> Result<HashMap<String, Vec<f32>>, String> {
    let path = path.as_ref();
    let file =
        std::fs::File::open(path).map_err(|e| format!("read_npz: cannot open {:?}: {}", path, e))?;
    let mut archive =
        zip::ZipArchive::new(file).map_err(|e| format!("read_npz: bad zip {:?}: {}", path, e))?;

    let mut result = HashMap::new();

    for i in 0..archive.len() {
        let mut entry = archive
            .by_index(i)
            .map_err(|e| format!("read_npz: entry {} error: {}", i, e))?;
        let entry_name = entry.name().to_string();
        // Strip .npy extension to get the key name
        let key = entry_name
            .strip_suffix(".npy")
            .ok_or_else(|| format!("read_npz: expected .npy extension, got '{}'", entry_name))?
            .to_string();

        let mut buf = Vec::new();
        entry
            .read_to_end(&mut buf)
            .map_err(|e| format!("read_npz: read '{}' error: {}", entry_name, e))?;

        let data = parse_npy(&buf, &entry_name)?;
        result.insert(key, data);
    }

    Ok(result)
}

/// Parse a single .npy byte buffer into a Vec<f32>.
///
/// Supports v1.0 (u16 header_len) and v2.0 (u32 header_len).
fn parse_npy(buf: &[u8], entry_name: &str) -> Result<Vec<f32>, String> {
    if buf.len() < 10 {
        return Err(format!("parse_npy: '{}' too short ({} bytes)", entry_name, buf.len()));
    }

    // Magic
    if &buf[..6] != NPY_MAGIC {
        return Err(format!(
            "parse_npy: '{}' bad magic: {:?}",
            entry_name,
            &buf[..6]
        ));
    }

    let major = buf[6];
    let minor = buf[7];

    // Parse header_len and data_start based on version
    let (header_len, data_start) = match major {
        1 => {
            let len = u16::from_le_bytes([buf[8], buf[9]]) as usize;
            (len, 10 + len)
        }
        2 => {
            if buf.len() < 12 {
                return Err(format!("parse_npy: '{}' too short for v2 ({} bytes)", entry_name, buf.len()));
            }
            let len = u32::from_le_bytes([buf[8], buf[9], buf[10], buf[11]]) as usize;
            (len, 12 + len)
        }
        v => {
            return Err(format!(
                "parse_npy: '{}' unsupported version {}.{}",
                entry_name, v, minor
            ));
        }
    };

    if buf.len() < data_start {
        return Err(format!(
            "parse_npy: '{}' header_len {} exceeds buffer size {}",
            entry_name,
            header_len,
            buf.len()
        ));
    }

    // Parse header dict literal
    let header_bytes = &buf[10..10 + header_len];
    let header_str =
        std::str::from_utf8(header_bytes).map_err(|e| format!("parse_npy: invalid utf8: {}", e))?;
    let (shape, descr, fortran_order) = parse_header_dict(header_str, entry_name)?;

    // Validate dtype is float32
    let is_f4 = descr == "<f4" || descr == "|f4"; // numpy outputs <f4
    if !is_f4 {
        return Err(format!(
            "parse_npy: '{}' unsupported dtype '{}', expected '<f4'",
            entry_name, descr
        ));
    }

    if fortran_order {
        return Err(format!(
            "parse_npy: '{}' has fortran_order=True, not supported (only C-contiguous)",
            entry_name
        ));
    }

    // Calculate expected number of elements
    let num_elements: usize = shape.iter().product();
    let data_bytes = &buf[data_start..];

    let expected_bytes = num_elements * std::mem::size_of::<f32>();
    if data_bytes.len() < expected_bytes {
        return Err(format!(
            "parse_npy: '{}' data too short: shape {:?} needs {} bytes, got {}",
            entry_name,
            shape,
            expected_bytes,
            data_bytes.len()
        ));
    }

    // Read f32 values (little-endian)
    let mut result = Vec::with_capacity(num_elements);
    let chunks = data_bytes[..expected_bytes].chunks_exact(4);
    for chunk in chunks {
        let val = f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        result.push(val);
    }

    Ok(result)
}

/// Parse the Python dict literal from the .npy header.
///
/// Extracts 'descr', 'fortran_order', and 'shape' fields.
/// Header example: "{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }"
fn parse_header_dict(
    header: &str,
    entry_name: &str,
) -> Result<(Vec<usize>, String, bool), String> {
    let descr = extract_string_value(header, "'descr':")
        .or_else(|| extract_string_value(header, "\"descr\":"))
        .ok_or_else(|| {
            format!("parse_npy: '{}' missing 'descr' in header", entry_name)
        })?;
    let descr = descr.trim_matches('\'').trim_matches('"').to_string();

    let fortran = header.contains("True") && header.contains("fortran_order");

    let shape_str = extract_paren_value(header, "'shape':")
        .or_else(|| extract_paren_value(header, "\"shape\":"))
        .ok_or_else(|| {
            format!("parse_npy: '{}' missing 'shape' in header", entry_name)
        })?;

    let shape: Vec<usize> = shape_str
        .trim_matches(|c| c == '(' || c == ')' || c == ',')
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| {
            s.trim()
                .parse::<usize>()
                .map_err(|e| format!("parse_npy: bad shape element '{}': {}", s, e))
        })
        .collect::<Result<_, _>>()?;

    Ok((shape, descr, fortran))
}

/// Extract a single-quoted string value after a key, e.g. `'descr': '<f4'` → `'<f4'`.
fn extract_string_value(header: &str, key: &str) -> Option<String> {
    let pos = header.find(key)?;
    let after_key = &header[pos + key.len()..];
    let after_key = after_key.trim_start();
    if after_key.starts_with('\'') || after_key.starts_with('"') {
        let quote = after_key.chars().next()?;
        let end = after_key[1..].find(quote)?;
        Some(after_key[..end + 2].to_string())
    } else {
        None
    }
}

/// Extract a parenthesized value after a key, e.g. `'shape': (2, 3)` → `(2, 3)`.
fn extract_paren_value(header: &str, key: &str) -> Option<String> {
    let pos = header.find(key)?;
    let after_key = &header[pos + key.len()..];
    let after_key = after_key.trim_start();
    let start = after_key.find('(')?;
    let end = after_key[start..].find(')')?;
    Some(after_key[start..start + end + 1].to_string())
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// Write a minimal valid .npy file (v1.0) for float32 data.
    fn write_npy_v1(buf: &mut Vec<u8>, shape: &[usize], data: &[f32]) {
        let shape_parts: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
        let shape_str = if shape.len() == 1 {
            format!("({},)", shape_parts[0])
        } else {
            format!("({})", shape_parts.join(", "))
        };
        let header = format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': {}, }}",
            shape_str
        );
        let header_len = header.len();
        // Pad to make total prefix (10 + header_len) align to 64 bytes — numpy style
        // Actually numpy pads with spaces to align data to 16-byte boundary
        let total_needed = 10 + header_len;
        let remainder = total_needed % 64;
        let padding = if remainder > 0 { 64 - remainder } else { 0 };
        let padded_header = format!("{}{}\n", header, " ".repeat(padding));
        let final_header_len = padded_header.len();

        buf.extend(NPY_MAGIC); // 6
        buf.push(1u8); // major
        buf.push(0u8); // minor
        buf.extend(&(final_header_len as u16).to_le_bytes()); // 2
        buf.extend(padded_header.as_bytes());
        // Data
        for &val in data {
            buf.extend(&val.to_le_bytes());
        }
    }

    /// Create a .npz zip file from a dict of {name: Vec<f32>} with given shapes.
    fn write_test_npz(path: &Path, entries: &[(&str, &[usize], &[f32])]) {
        let writer = std::fs::File::create(path).unwrap();
        let mut zip_writer = zip::ZipWriter::new(writer);
        let options =
            zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Stored);

        for &(name, shape, data) in entries {
            let entry_name = format!("{}.npy", name);
            zip_writer.start_file(entry_name, options).unwrap();
            let mut npy_buf = Vec::new();
            write_npy_v1(&mut npy_buf, shape, data);
            zip_writer.write_all(&npy_buf).unwrap();
        }
        zip_writer.finish().unwrap();
    }

    #[test]
    fn test_parse_npy_v1_simple() {
        let mut buf = Vec::new();
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        write_npy_v1(&mut buf, &[2, 3], &data);
        let result = parse_npy(&buf, "test.npy").unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_parse_npy_v1_1d() {
        let mut buf = Vec::new();
        let data: Vec<f32> = vec![0.0, -1.5, 3.25, 42.0];
        write_npy_v1(&mut buf, &[4], &data);
        let result = parse_npy(&buf, "test_1d.npy").unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_parse_npy_v1_3d() {
        let mut buf = Vec::new();
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        write_npy_v1(&mut buf, &[1, 6, 4], &data);
        let result = parse_npy(&buf, "test_3d.npy").unwrap();
        assert_eq!(result, data);
    }

    #[test]
    fn test_read_npz_roundtrip() {
        let tmp = std::env::temp_dir().join("test_golden_reader_roundtrip.npz");
        let data_0: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let data_1: Vec<f32> = vec![1.0f32; 12 * 64];
        let data_2: Vec<f32> = vec![0.0f32; 768];
        let entries = vec![
            ("output_0", &[1usize, 6, 4][..], &data_0[..]),
            ("output_1", &[12, 64][..], &data_1[..]),
            ("output_2", &[768][..], &data_2[..]),
        ];
        write_test_npz(&tmp, &entries);

        let result = read_npz(&tmp).unwrap();
        assert_eq!(result.len(), 3);
        assert!(result.contains_key("output_0"));
        assert!(result.contains_key("output_1"));
        assert!(result.contains_key("output_2"));

        let out0 = &result["output_0"];
        assert_eq!(out0.len(), 24);
        for (i, &val) in out0.iter().enumerate() {
            assert!((val - i as f32).abs() < 1e-7, "output_0[{}] = {}", i, val);
        }

        let out1 = &result["output_1"];
        assert_eq!(out1.len(), 12 * 64);
        assert!(out1.iter().all(|&v| (v - 1.0).abs() < 1e-7));

        let out2 = &result["output_2"];
        assert_eq!(out2.len(), 768);
        assert!(out2.iter().all(|&v| v.abs() < 1e-7));

        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn test_read_npz_from_python() {
        // This test reads a .npz file written by golden_io.py's save_npz.
        let py_npz = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../tests/data/golden/test_python_written.npz");

        assert!(
            py_npz.exists(),
            "SKIP test_read_npz_from_python: {} not found. \
             Generate with: python -c \"from compiler.tests.golden_io import save_npz; \
             import numpy as np; import os; \
             os.makedirs('tests/data/golden', exist_ok=True); \
             save_npz('tests/data/golden/test_python_written.npz', \
             {{'output_0': np.arange(24, dtype=np.float32).reshape(1,6,4), \
              'output_1': np.ones((12,64), dtype=np.float32), \
              'output_2': np.zeros((768,), dtype=np.float32)}})\"",
            py_npz.display()
        );

        let result = read_npz(&py_npz).unwrap();
        assert_eq!(result.len(), 3, "Expected 3 arrays in .npz");

        // output_0: arange(24) as f32, shape (1,6,4)
        let out0 = &result["output_0"];
        assert_eq!(out0.len(), 24);
        for (i, &val) in out0.iter().enumerate() {
            assert!(
                (val - i as f32).abs() < 1e-7,
                "output_0[{}] expected {}, got {}",
                i, i, val
            );
        }

        // output_1: ones(12, 64)
        let out1 = &result["output_1"];
        assert_eq!(out1.len(), 12 * 64);
        assert!(
            out1.iter().all(|&v| (v - 1.0).abs() < 1e-7),
            "output_1: expected all 1.0"
        );

        // output_2: zeros(768)
        let out2 = &result["output_2"];
        assert_eq!(out2.len(), 768);
        assert!(
            out2.iter().all(|&v| v.abs() < 1e-7),
            "output_2: expected all 0.0"
        );
    }
}
