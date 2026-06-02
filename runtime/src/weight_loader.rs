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
        crate::hal::cpu::sret::checked_product_usize(&self.shape).unwrap_or(usize::MAX)
    }
}

// ── Binary format parsing (SFCF v2/v3) ─────────────────────────────

pub fn parse_embedded(data: &[u8]) -> Result<(WeightRegistry, usize, u32), anyhow::Error> {
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
    if !(2..=4).contains(&version) {
        return Err(ExecutorError::SfcfParse(format!(
            "unsupported binary version: {} (expected 2..=4)", version,
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
        version,
    ))
}

// ── Contract parsing (v4+) ──────────────────────────────────────────

/// Parse the contract section appended after the compute graph trailer.
///
/// Format:
///   contract_count: u32
///   for each entry:
///     key_len: u32, key: utf8 bytes, val_len: u32, val: utf8 bytes
///
/// Returns an empty HashMap when no contract section is present (v2/v3
/// backward compat).
pub fn parse_contract(data: &[u8], pos: &mut usize) -> Result<HashMap<String, String>, anyhow::Error> {
    if *pos >= data.len() {
        return Ok(HashMap::new());
    }
    let count = sfcf::read_u32(data, pos)? as usize;
    let mut contract = HashMap::with_capacity(count);
    for _ in 0..count {
        let key = sfcf::read_string(data, pos)?;
        let val = sfcf::read_string(data, pos)?;
        contract.insert(key, val);
    }
    Ok(contract)
}

// ── Dylib loading ─────────────────────────────────────────────────

#[allow(dead_code)]
pub fn load_registry_from_dylib(
    lib: &libloading::Library,
) -> Result<(WeightRegistry, usize, u32), anyhow::Error> {
    // SAFETY: `serveforge_constants_data` is a `const uint8_t[]` symbol
    // embedded in the .dylib at compile time. It points to static data
    // in the dylib's read-only data section, valid for the Library lifetime.
    let data_ptr: *const u8 = {
        // SAFETY: libloading::Symbol::get() is safe for the dylib's lifetime.
        let sym: libloading::Symbol<*const c_void> = unsafe {
            lib.get(b"serveforge_constants_data")
                .map_err(|e| anyhow::anyhow!("{}", e))?
        };
        *sym as *const u8
    };

    let size_val: u64 = {
        // SAFETY: libloading::get() looks up a typed symbol from the dylib,
        // valid for the Library lifetime.
        let sym = unsafe {
            lib.get::<*const u64>(b"serveforge_constants_size")
                .map_err(|e| anyhow::anyhow!("{}", e))?
        };
        // SAFETY: The symbol was just successfully looked up; dereferencing
        // it reads the compile-time constant embedded in the dylib.
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
#[derive(Debug, Clone)]
pub(crate) struct CachedTensorInfo {
    pub(crate) data_start: usize,
    pub(crate) data_end: usize,
    pub(crate) shape: Vec<usize>,
    pub(crate) dtype: Dtype,
}

pub struct WeightProvider {
    registry: WeightRegistry,
    #[allow(dead_code)]
    safetensors_mmap: Option<memmap2::Mmap>,
    /// Cached header info: HF key → (start_offset, end_offset, shape).
    /// Parsed once in `new()` to avoid O(n×header_size) on every lookup.
    safetensors_index: HashMap<String, CachedTensorInfo>,
    /// Reverse mapping: HF weight name → compiled SFCF name.
    /// Built once in `new()` from `registry.name_mapping`.
    #[allow(dead_code)]
    hf_to_compiled: HashMap<String, String>,
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
        // Build reverse mapping: HF name → compiled SFCF name.
        let hf_to_compiled: HashMap<String, String> = registry
            .name_mapping
            .iter()
            .map(|(k, v)| (v.clone(), k.clone()))
            .collect();

        Ok(Self {
            registry,
            safetensors_mmap,
            safetensors_index,
            hf_to_compiled,
        })
    }

    #[allow(dead_code)]
    pub fn has_safetensors(&self) -> bool {
        self.safetensors_mmap.is_some()
    }

    pub fn get_weight_memref(&self, compiled_name: &str) -> Option<(MemRefDesc2, Dtype)> {
        if let Some(ct) = self.registry.constants.get(compiled_name) {
            return Some((constant_as_memref(ct), ct.dtype));
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
        Some((MemRefDesc2 {
            allocated: data_slice.as_ptr() as *mut c_void,
            aligned: data_slice.as_ptr() as *mut c_void,
            offset: 0,
            sizes: [rows as i64, cols as i64],
            strides: [cols as i64, 1],
        }, info.dtype))
    }

    /// Look up a CachedTensorInfo by compiled weight name for dtype inspection.
    #[allow(dead_code)]
    pub(crate) fn get_weight_info(&self, compiled_name: &str) -> Option<&CachedTensorInfo> {
        let hf_key = self.registry.name_mapping.get(compiled_name)?;
        self.safetensors_index.get(hf_key)
    }

    /// Given an HF-style weight name (e.g. "model.decoder.layers.0.self_attn.q_proj.weight"),
    /// look up the compiled SFCF name from the reverse mapping.
    /// Returns `None` if no match is found.
    #[allow(dead_code)]
    pub fn resolve_hf_weight_name(&self, hf_name: &str) -> Option<&str> {
        self.hf_to_compiled.get(hf_name).map(|s| s.as_str())
    }

    #[allow(dead_code)]
    pub fn name_mapping(&self) -> &HashMap<String, String> {
        &self.registry.name_mapping
    }

    #[allow(dead_code)]
    pub fn constants(&self) -> &HashMap<String, ConstantTensor> {
        &self.registry.constants
    }

    /// Check if a compiled name corresponds to an embedded constant
    /// (i64 scalar) rather than a safetensors weight (f16).
    #[allow(dead_code)]
    pub fn is_constant(&self, compiled_name: &str) -> bool {
        self.registry.constants.contains_key(compiled_name)
    }
}

/// Parse the safetensors JSON header once and build an index of
/// (start_offset, end_offset, shape) for every tensor.
#[allow(dead_code)]
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
            let dtype_str = info
                .get("dtype")
                .and_then(|d| d.as_str())
                .unwrap_or("F32");
            let dtype = match dtype_str {
                "F32" => Dtype::F32,
                "F16" => Dtype::F16,
                "BF16" => Dtype::BF16,
                "I64" => Dtype::I64,
                "I32" => Dtype::I32,
                "I8" => Dtype::I8,
                "U8" => Dtype::U8,
                _ => Dtype::F32, // default fallback
            };
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
                    dtype,
                },
            );
        }
    }
    Ok(index)
}

#[allow(dead_code)]
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
        let (reg, pos, ver) = parse_embedded(&data).expect("parse");
        assert!(reg.name_mapping.is_empty());
        assert!(reg.constants.is_empty());
        assert!(pos > 0);
        assert_eq!(ver, 2);
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

        let (reg, _pos, _ver) = parse_embedded(&buf).expect("parse");
        assert_eq!(reg.name_mapping.len(), 2);
        assert_eq!(
            reg.name_mapping.get("a").map(|s| s.as_str()),
            Some("model.a.weight")
        );
    }

    #[test]
    fn test_parse_v3_supported() {
        let mut buf = Vec::new();
        buf.extend_from_slice(b"SFCF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // v3
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 name mappings
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 constants
        let (_, _, ver) = parse_embedded(&buf).expect("v3 should be supported");
        assert_eq!(ver, 3);
    }

    // ── Contract (v4) tests ──────────────────────────────────

    /// Build a minimal SFCF v4 binary followed by a contract section.
    fn sfcf_v4_with_contract(entries: &[(&str, &str)]) -> Vec<u8> {
        let mut buf = Vec::new();
        // SFCF header
        buf.extend_from_slice(b"SFCF");
        buf.extend_from_slice(&4u32.to_le_bytes()); // v4
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 name mappings
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 constants

        // Minimal compute graph: 0 functions, then global I/O trailer
        buf.extend_from_slice(&0u32.to_le_bytes()); // num_funcs
        buf.extend_from_slice(&0u32.to_le_bytes()); // global_input_func
        buf.extend_from_slice(&0u32.to_le_bytes()); // global_input_arg
        buf.extend_from_slice(&0u32.to_le_bytes()); // global_output_func
        buf.extend_from_slice(&0u32.to_le_bytes()); // global_output_idx

        // Contract section
        buf.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (k, v) in entries {
            let kb = k.as_bytes();
            buf.extend_from_slice(&(kb.len() as u32).to_le_bytes());
            buf.extend_from_slice(kb);
            let vb = v.as_bytes();
            buf.extend_from_slice(&(vb.len() as u32).to_le_bytes());
            buf.extend_from_slice(vb);
        }
        buf
    }

    #[test]
    fn test_parse_v4_supported() {
        let buf = sfcf_v4_with_contract(&[]);
        let (_, _, ver) = parse_embedded(&buf).expect("v4 should be supported");
        assert_eq!(ver, 4);
    }

    #[test]
    fn test_parse_contract_v4() {
        let buf = sfcf_v4_with_contract(&[
            ("sfcf_version", "4"),
            ("num_global_inputs", "2"),
            ("global_input_names", "input_ids,position_ids"),
        ]);
        let (_, graph_pos, ver) = parse_embedded(&buf).expect("parse v4 with contract");
        assert_eq!(ver, 4);

        // Parse contract from position after compute graph (which has 0 functions)
        // Header is 16 bytes + compute graph trailer is 20 bytes = 36
        let mut pos = graph_pos + 20; // skip compute graph (0 funcs)
        let contract = parse_contract(&buf, &mut pos).expect("parse contract");
        assert_eq!(contract.len(), 3);
        assert_eq!(contract.get("sfcf_version").map(|s| s.as_str()), Some("4"));
        assert_eq!(contract.get("num_global_inputs").map(|s| s.as_str()), Some("2"));
        assert_eq!(
            contract.get("global_input_names").map(|s| s.as_str()),
            Some("input_ids,position_ids"),
        );
    }

    #[test]
    fn test_parse_contract_v3_backward_compat() {
        // v3 binary has no contract section — parse_contract must return empty HashMap
        let mut buf = Vec::new();
        buf.extend_from_slice(b"SFCF");
        buf.extend_from_slice(&3u32.to_le_bytes()); // v3
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 name mappings
        buf.extend_from_slice(&0u32.to_le_bytes()); // 0 constants

        let mut pos = 16; // past the header
        let contract = parse_contract(&buf, &mut pos).expect("v3 backward compat");
        assert!(contract.is_empty(), "v3 binary should have no contract");
        // pos should not advance past the header for v3
        assert_eq!(pos, 16);
    }

    #[test]
    fn test_parse_contract_empty_data() {
        // Parsing from end of data returns empty HashMap (no crash)
        let mut pos = 0usize;
        let contract = parse_contract(&[], &mut pos).expect("empty data");
        assert!(contract.is_empty());
        assert_eq!(pos, 0);
    }
}
