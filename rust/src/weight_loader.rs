//! Safetensors weight loader with embedded name mapping and constants.
//!
//! The compiled .dylib contains two exported symbols:
//!   ``serveforge_constants_data`` — binary SFCF blob (name mapping + constants)
//!   ``serveforge_constants_size`` — byte size of the blob
//!
//! The Rust runtime reads these symbols at load time and uses the
//! name mapping to resolve compiled weight names → original HF safetensors
//! keys, falling back to the embedded constants for compiler-synthesized
//! tensors.  Weights are accessed via zero-copy mmap.

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

use crate::error::ExecutorError;
use crate::hal::cpu::MemRefDesc2;
use crate::sfcf;
use crate::tensor::Dtype;

// ── Weight registry ────────────────────────────────────────────────

pub struct WeightRegistry {
    pub name_mapping: HashMap<String, String>,
    pub constants: HashMap<String, ConstantTensor>,
}

#[derive(Debug, Clone)]
pub struct ConstantTensor {
    #[allow(dead_code)]
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl ConstantTensor {
    #[allow(dead_code)]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

// ── Binary format parsing (SFCF v2) ────────────────────────────────

pub fn parse_embedded(data: &[u8]) -> Result<(WeightRegistry, usize), anyhow::Error> {
    if data.len() < 8 {
        return Err(ExecutorError::SfcfParse(format!(
            "embedded data too short: {} bytes", data.len(),
        )).into());
    }
    if &data[0..4] != b"SFCF" {
        return Err(ExecutorError::SfcfParse(format!(
            "bad magic: {:?}", &data[0..4],
        )).into());
    }
    let version = u32::from_le_bytes(data[4..8].try_into()?);
    if version != 2 {
        return Err(ExecutorError::SfcfParse(format!(
            "unsupported binary version: {} (expected 2)", version,
        )).into());
    }

    let mut pos = 8usize;

    let nm_count = sfcf::read_u32(data, &mut pos)? as usize;
    let mut name_mapping = HashMap::with_capacity(nm_count);
    for _ in 0..nm_count {
        let compiled = sfcf::read_string(data, &mut pos)?;
        let hf_key = sfcf::read_string(data, &mut pos)?;
        name_mapping.insert(compiled, hf_key);
    }

    let const_count = sfcf::read_u32(data, &mut pos)? as usize;
    let mut constants = HashMap::with_capacity(const_count);
    for _ in 0..const_count {
        let name = sfcf::read_string(data, &mut pos)?;
        if pos >= data.len() {
            return Err(ExecutorError::SfcfParse(
                format!("truncated constant: {}", name),
            ).into());
        }
        let dtype_code = data[pos];
        pos += 1;
        let dtype = Dtype::from_code(dtype_code).ok_or_else(|| {
            anyhow::anyhow!("unknown dtype code {} for constant {}", dtype_code, name)
        })?;
        let ndim = data[pos] as usize;
        pos += 1;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(sfcf::read_u64(data, &mut pos)? as usize);
        }
        let data_len = sfcf::read_u64(data, &mut pos)? as usize;
        if pos + data_len > data.len() {
            return Err(ExecutorError::SfcfParse(
                format!("truncated constant data: {} (need {} bytes)", name, data_len),
            ).into());
        }
        let tensor_data = data[pos..pos + data_len].to_vec();
        pos += data_len;
        constants.insert(
            name,
            ConstantTensor {
                dtype,
                shape,
                data: tensor_data,
            },
        );
    }

    Ok((
        WeightRegistry {
            name_mapping,
            constants,
        },
        pos,
    ))
}

// ── Dylib loading ─────────────────────────────────────────────────

#[allow(dead_code)]
pub fn load_registry_from_dylib(
    lib: &libloading::Library,
) -> Result<(WeightRegistry, usize), anyhow::Error> {
    // SAFETY: `serveforge_constants_data` is a `const uint8_t[]` symbol
    // embedded in the .dylib at compile time. It points to static data
    // in the dylib's read-only data section, valid for the Library lifetime.
    let data_ptr: *const u8 = {
        let sym: libloading::Symbol<*const c_void> = unsafe {
            lib.get(b"serveforge_constants_data")
                .map_err(|e| anyhow::anyhow!("{}", e))?
        };
        *sym as *const u8
    };

    let size_val: u64 = {
        let sym = unsafe {
            lib.get::<*const u64>(b"serveforge_constants_size")
                .map_err(|e| anyhow::anyhow!("{}", e))?
        };
        unsafe { *(*sym) }
    };

    if size_val == 0 || data_ptr.is_null() {
        return Err(ExecutorError::SfcfParse(
            "embedded data empty or missing".to_string(),
        ).into());
    }

    // SAFETY: `data_ptr` and `size_val` come from the same dylib.
    // The size bounds the data region and the pointer is to static memory.
    let data: &[u8] = unsafe { std::slice::from_raw_parts(data_ptr, size_val as usize) };
    parse_embedded(data)
}

// ── WeightProvider ─────────────────────────────────────────────────

/// Pre-parsed safetensors tensor metadata (cached for O(1) lookup).
struct CachedTensorInfo {
    data_start: usize,
    data_end: usize,
    shape: Vec<usize>,
}

pub struct WeightProvider {
    registry: WeightRegistry,
    safetensors_mmap: Option<memmap2::Mmap>,
    /// Cached header info: HF key → (start_offset, end_offset, shape).
    /// Parsed once in `new()` to avoid O(n×header_size) on every lookup.
    safetensors_index: HashMap<String, CachedTensorInfo>,
}

impl WeightProvider {
    pub fn new(
        registry: WeightRegistry,
        safetensors_path: Option<&Path>,
    ) -> Result<Self, anyhow::Error> {
        let (safetensors_mmap, safetensors_index) =
            if let Some(p) = safetensors_path {
                let file = std::fs::File::open(p)?;
                // SAFETY: The file is opened read-only. The mmap provides
                // immutable access to the file contents.
                let mmap = unsafe { memmap2::Mmap::map(&file)? };
                let index = build_safetensors_index(&mmap)?;
                (Some(mmap), index)
            } else {
                (None, HashMap::new())
            };
        Ok(Self {
            registry,
            safetensors_mmap,
            safetensors_index,
        })
    }

    #[allow(dead_code)]
    pub fn has_safetensors(&self) -> bool {
        self.safetensors_mmap.is_some()
    }

    pub fn get_weight_memref(&self, compiled_name: &str) -> Option<MemRefDesc2> {
        if let Some(ct) = self.registry.constants.get(compiled_name) {
            return Some(constant_as_memref(ct));
        }

        let hf_key = self.registry.name_mapping.get(compiled_name)?;
        let mmap = self.safetensors_mmap.as_ref()?;
        let info = self.safetensors_index.get(hf_key)?;
        let data_slice = &mmap[info.data_start..info.data_end];
        let rows = *info.shape.first().unwrap_or(&1);
        let cols = info.shape.get(1).copied().unwrap_or(1);
        // SAFETY: The mmap region (data_start..data_end) is read-only memory
        // backed by the safetensors file.  MemRefDesc2 stores pointers as
        // *mut c_void because the ciface kernel expects mutable descriptors,
        // but the safetensors data is never actually written to — the kernel
        // only reads weights.  The cast from *const u8 to *mut c_void is
        // safe because all accesses through this descriptor are reads.
        Some(MemRefDesc2 {
            allocated: data_slice.as_ptr() as *mut c_void,
            aligned: data_slice.as_ptr() as *mut c_void,
            offset: 0,
            sizes: [rows as i64, cols as i64],
            strides: [cols as i64, 1],
        })
    }

    pub fn name_mapping(&self) -> &HashMap<String, String> {
        &self.registry.name_mapping
    }

    pub fn constants(&self) -> &HashMap<String, ConstantTensor> {
        &self.registry.constants
    }
}

/// Parse the safetensors JSON header once and build an index of
/// (start_offset, end_offset, shape) for every tensor.
fn build_safetensors_index(
    mmap: &[u8],
) -> Result<HashMap<String, CachedTensorInfo>, anyhow::Error> {
    if mmap.len() < 8 {
        return Err(ExecutorError::SfcfParse(
            "safetensors file too short".to_string(),
        ).into());
    }
    let header_len = u64::from_le_bytes(mmap[..8].try_into()?);
    let header_len = header_len as usize;
    if mmap.len() < 8 + header_len {
        return Err(ExecutorError::SfcfParse(
            "safetensors header truncated".to_string(),
        ).into());
    }
    let header_bytes = &mmap[8..8 + header_len];
    let header: serde_json::Value = serde_json::from_slice(header_bytes)?;

    let mut index = HashMap::new();
    if let Some(obj) = header.as_object() {
        for (key, info) in obj {
            let shape: Vec<usize> = info
                .get("shape")
                .and_then(|s| s.as_array())
                .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(1) as usize).collect())
                .unwrap_or_default();
            let offsets = match info.get("data_offsets").and_then(|o| o.as_array()) {
                Some(arr) if arr.len() >= 2 => arr,
                _ => continue,
            };
            let start = offsets[0].as_u64().unwrap_or(0) as usize + 8 + header_len;
            let end = offsets[1].as_u64().unwrap_or(0) as usize + 8 + header_len;
            if end <= start || end > mmap.len() {
                continue;
            }
            index.insert(
                key.clone(),
                CachedTensorInfo {
                    data_start: start,
                    data_end: end,
                    shape,
                },
            );
        }
    }
    Ok(index)
}

fn constant_as_memref(ct: &ConstantTensor) -> MemRefDesc2 {
    let p = ct.data.as_ptr();
    let rows = *ct.shape.first().unwrap_or(&1);
    let cols = ct.shape.get(1).copied().unwrap_or(1);
    MemRefDesc2 {
        allocated: p as *mut c_void,
        aligned: p as *mut c_void,
        offset: 0,
        sizes: [rows as i64, cols as i64],
        strides: [cols as i64, 1],
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sfcf_v2_empty() -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"SFCF");
        buf.extend_from_slice(&2u32.to_le_bytes());
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 name mappings
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 constants
        buf
    }

    #[test]
    fn test_parse_empty() {
        let data = sfcf_v2_empty();
        let (reg, pos) = parse_embedded(&data).expect("parse");
        assert!(reg.name_mapping.is_empty());
        assert!(reg.constants.is_empty());
        assert!(pos > 0);
    }

    #[test]
    fn test_parse_bad_magic() {
        let data = b"XXXX".to_vec();
        assert!(parse_embedded(&data).is_err());
    }

    #[test]
    fn test_parse_bad_version() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"SFCF");
        buf.extend_from_slice(&1u32.to_le_bytes()); // old version
        assert!(parse_embedded(&buf).is_err());
    }

    #[test]
    fn test_parse_name_mapping_roundtrip() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"SFCF");
        buf.extend_from_slice(&2u32.to_le_bytes()); // version
        buf.extend_from_slice(&2u32.to_le_bytes()); // 2 entries
        for (short, long) in [("a", "model.a.weight"), ("b", "model.b.weight")] {
            let s = short.as_bytes();
            buf.extend_from_slice(&(s.len() as u32).to_le_bytes());
            buf.extend_from_slice(s);
            let l = long.as_bytes();
            buf.extend_from_slice(&(l.len() as u32).to_le_bytes());
            buf.extend_from_slice(l);
        }
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 constants

        let (reg, _pos) = parse_embedded(&buf).expect("parse");
        assert_eq!(reg.name_mapping.len(), 2);
        assert_eq!(
            reg.name_mapping.get("a").map(|s| s.as_str()),
            Some("model.a.weight")
        );
    }
}
