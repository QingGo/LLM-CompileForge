//! Dump all runtime-loaded weights as .npy files for offline inspection.
//!
//! Loads the compiled .dylib, reads the name_mapping from the embedded
//! SFCF blob, resolves HF keys in the safetensors file, converts f16→f32,
//! and writes individual .npy files plus a weights_index.json.

use std::collections::HashMap;
use std::path::PathBuf;

// ── Safetensors index (lightweight parser, no crate needed) ────────

struct STensorInfo {
    dtype: String, // "F16", "F32", "BF16", etc.
    shape: Vec<usize>,
    data_start: usize,
    data_end: usize,
}

/// Parse the safetensors header JSON and build an offset index.
fn parse_safetensors_index(data: &[u8]) -> Result<HashMap<String, STensorInfo>, Box<dyn std::error::Error>> {
    if data.len() < 8 {
        return Err("safetensors file too short".into());
    }
    let header_len = u64::from_le_bytes(data[..8].try_into()?) as usize;
    let header_bytes = &data[8..8 + header_len];
    let header: serde_json::Value = serde_json::from_slice(header_bytes)?;

    let mut index = HashMap::new();
    if let Some(obj) = header.as_object() {
        for (key, info) in obj {
            let dtype = info["dtype"].as_str().unwrap_or("F32").to_string();
            let shape: Vec<usize> = info["shape"]
                .as_array()
                .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect())
                .unwrap_or_default();
            let offsets = match info["data_offsets"].as_array() {
                Some(arr) if arr.len() >= 2 => arr,
                _ => continue,
            };
            let start = offsets[0].as_u64().unwrap_or(0) as usize + 8 + header_len;
            let end = offsets[1].as_u64().unwrap_or(0) as usize + 8 + header_len;
            index.insert(
                key.clone(),
                STensorInfo {
                    dtype,
                    shape,
                    data_start: start,
                    data_end: end,
                },
            );
        }
    }
    Ok(index)
}

// ── Data conversion helpers ────────────────────────────────────────

/// Convert raw bytes to f32, dispatching on safetensors dtype string.
fn convert_to_f32(data: &[u8], dtype: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match dtype {
        "F16" => Ok(data
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        "BF16" => Ok(data
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        "F32" => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        "F64" => Ok(data
            .chunks_exact(8)
            .map(|c| {
                f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
            })
            .collect()),
        "I64" => Ok(data
            .chunks_exact(8)
            .map(|c| {
                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
            })
            .collect()),
        "I32" => Ok(data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect()),
        "I8" => Ok(data.iter().map(|&b| b as i8 as f32).collect()),
        "U8" => Ok(data.iter().map(|&b| b as f32).collect()),
        _ => Err(format!("unsupported safetensors dtype: {}", dtype).into()),
    }
}

/// Convert a ConstantTensor to f32 (all dtypes promote to f32 for npy).
fn constant_to_f32(
    ct: &crate::weight_loader::ConstantTensor,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    match ct.dtype {
        crate::tensor::Dtype::F32 => Ok(ct
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        crate::tensor::Dtype::F16 => Ok(ct
            .data
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        crate::tensor::Dtype::BF16 => Ok(ct
            .data
            .chunks_exact(2)
            .map(|c| half::bf16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect()),
        crate::tensor::Dtype::I64 => Ok(ct
            .data
            .chunks_exact(8)
            .map(|c| {
                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
            })
            .collect()),
        crate::tensor::Dtype::I32 => Ok(ct
            .data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect()),
        crate::tensor::Dtype::I8 => Ok(ct.data.iter().map(|&b| b as i8 as f32).collect()),
        crate::tensor::Dtype::U8 => Ok(ct.data.iter().map(|&b| b as f32).collect()),
    }
}

// ── NumPy .npy writer ──────────────────────────────────────────────

/// Write an f32 tensor to a .npy file (NumPy format v1.0).
fn write_npy(path: &str, data: &[f32], shape: &[usize]) -> std::io::Result<()> {
    use std::io::Write;
    let mut file = std::fs::File::create(path)?;

    let shape_str = shape
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    let header = if shape.is_empty() {
        "{'descr': '<f4', 'fortran_order': False, 'shape': (), }".to_string()
    } else if shape.len() == 1 {
        format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({},), }}",
            shape_str
        )
    } else {
        format!(
            "{{'descr': '<f4', 'fortran_order': False, 'shape': ({}), }}",
            shape_str
        )
    };

    let header_bytes = header.as_bytes();
    let total_before_pad = 10 + header_bytes.len();
    let padding = (64 - (total_before_pad % 64)) % 64;
    let header_len = header_bytes.len() as u16 + padding as u16;

    file.write_all(b"\x93NUMPY")?;
    file.write_all(&[1, 0])?;
    file.write_all(&header_len.to_le_bytes())?;
    file.write_all(header_bytes)?;
    for _ in 0..padding {
        file.write_all(b" ")?;
    }

    let byte_slice =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    file.write_all(byte_slice)?;
    Ok(())
}

// ── File discovery ─────────────────────────────────────────────────

/// Find a .dylib in the compiled directory.
fn find_dylib(compiled_dir: &PathBuf) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let entries = std::fs::read_dir(compiled_dir).map_err(|e| {
        format!("Cannot read compiled dir '{}': {}", compiled_dir.display(), e)
    })?;
    for entry in entries.flatten() {
        if entry
            .path()
            .extension()
            .map(|ext| ext == "dylib")
            .unwrap_or(false)
        {
            return Ok(entry.path());
        }
    }
    Err(format!("No .dylib found in '{}'.", compiled_dir.display()).into())
}

/// Find a safetensors weight file (checks compiled dir, then HF cache).
fn find_safetensors(compiled_dir: &PathBuf) -> Result<String, Box<dyn std::error::Error>> {
    // Check compiled dir directly
    for name in &["model.safetensors", "weights.safetensors"] {
        let p = compiled_dir.join(name);
        if p.exists() {
            return Ok(p.to_string_lossy().to_string());
        }
    }
    // Check HF cache via metadata.json
    let meta_path = compiled_dir.join("metadata.json");
    if meta_path.exists() {
        let meta_str = std::fs::read_to_string(&meta_path)?;
        let meta: serde_json::Value = serde_json::from_str(&meta_str)?;
        if let Some(source_path) = meta["weight_source"]["path"].as_str() {
            let p = PathBuf::from(source_path);
            // If the path itself is a safetensors file
            if p.extension()
                .map(|e| e == "safetensors")
                .unwrap_or(false)
                && p.exists()
            {
                return Ok(p.to_string_lossy().to_string());
            }
            // Try model.safetensors in the same directory
            if let Some(parent) = p.parent() {
                let safetensors = parent.join("model.safetensors");
                if safetensors.exists() {
                    return Ok(safetensors.to_string_lossy().to_string());
                }
            }
            // Try HF cache (model id encoded in path)
            if let Some(model_part) = source_path.split("models--").nth(1) {
                let model_id = model_part.split('/').next().unwrap_or("");
                let hub_dir = PathBuf::from(&std::env::var("HOME").unwrap_or_default())
                    .join(".cache/huggingface/hub")
                    .join(format!("models--{}", model_id));
                let snapshots_dir = hub_dir.join("snapshots");
                if let Ok(entries) = std::fs::read_dir(&snapshots_dir) {
                    for entry in entries.flatten() {
                        let safetensors = entry.path().join("model.safetensors");
                        if safetensors.exists() {
                            return Ok(safetensors.to_string_lossy().to_string());
                        }
                    }
                }
            }
        }
    }
    Err(format!(
        "Cannot find model.safetensors/weights.safetensors in '{}' or HF cache",
        compiled_dir.display()
    )
    .into())
}

// ── Main entry point ───────────────────────────────────────────────

/// Run the weight dump. `compiled_dir` is the directory containing the
/// compiled .dylib and metadata. `output_dir` defaults to
/// `compiled_dir/dumped_weights` if `None`.
pub fn run(
    compiled_dir: PathBuf,
    output_dir: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let output_dir = output_dir.unwrap_or_else(|| compiled_dir.join("dumped_weights"));

    // 1. Load .dylib and parse SFCF blob
    let dylib_path = find_dylib(&compiled_dir)?;
    println!("[dump_weights] dylib: {}", dylib_path.display());

    let lib = unsafe { libloading::Library::new(&dylib_path) }
        .map_err(|e| format!("Failed to load dylib '{}': {}", dylib_path.display(), e))?;
    let (registry, _graph_pos, _sfcf_version) =
        crate::weight_loader::load_registry_from_dylib(&lib)?;
    println!(
        "[dump_weights] name_mapping: {} entries, constants: {} entries",
        registry.name_mapping.len(),
        registry.constants.len()
    );

    // 2. Find and parse safetensors file
    let safetensors_path = find_safetensors(&compiled_dir)?;
    println!("[dump_weights] safetensors: {}", safetensors_path);

    let safetensors_data = std::fs::read(&safetensors_path)?;
    let st_index = parse_safetensors_index(&safetensors_data)?;

    // 3. Create output directory
    std::fs::create_dir_all(&output_dir)?;

    let mut index_map = serde_json::Map::new();

    // 4. Dump name_mapping weights
    for (compiled_name, hf_key) in &registry.name_mapping {
        let st_info = st_index
            .get(hf_key)
            .ok_or_else(|| format!("HF key '{}' not found in safetensors", hf_key))?;

        let raw_data = &safetensors_data[st_info.data_start..st_info.data_end];
        let f32_data = convert_to_f32(raw_data, &st_info.dtype)?;

        let filename = format!("{}.npy", compiled_name);
        let npy_path = output_dir.join(&filename);
        println!(
            "  [weight] {} -> shape={:?} dtype={}",
            compiled_name, st_info.shape, st_info.dtype
        );
        write_npy(&npy_path.to_string_lossy(), &f32_data, &st_info.shape)?;

        index_map.insert(
            compiled_name.clone(),
            serde_json::json!({
                "shape": st_info.shape,
                "dtype": "f32",
                "file": filename,
            }),
        );
    }

    // 5. Dump constant tensors
    for (name, ct) in &registry.constants {
        let f32_data = constant_to_f32(ct)?;
        let filename = format!("{}.npy", name);
        let npy_path = output_dir.join(&filename);
        println!("  [const] {} -> shape={:?}", name, ct.shape);
        write_npy(&npy_path.to_string_lossy(), &f32_data, &ct.shape)?;

        index_map.insert(
            name.clone(),
            serde_json::json!({
                "shape": ct.shape,
                "dtype": "f32",
                "file": filename,
            }),
        );
    }

    // 6. Write weights_index.json
    let index_path = output_dir.join("weights_index.json");
    let index_json = serde_json::Value::Object(index_map);
    std::fs::write(&index_path, serde_json::to_string_pretty(&index_json)?)?;

    println!(
        "[dump_weights] ✓ Done — wrote {} weights to {}",
        index_json.as_object().map(|o| o.len()).unwrap_or(0),
        output_dir.display()
    );
    Ok(())
}
