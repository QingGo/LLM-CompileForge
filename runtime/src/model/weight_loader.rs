//! Safetensors weight loader with protobuf-based name mapping and constants.
//!
//! Weights are loaded via the proto ABI (``sfa_weights`` symbol in the dylib).
//! The legacy ``serveforge_constants_data`` / ``load_registry_from_dylib``
//! path is preserved for ``dump_weights_runner`` backward compatibility.

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

use crate::model::error::ExecutorError;
use crate::hal::cpu::MemRefDesc2;
use crate::model::tensor::Dtype;

// ── Weight registry ────────────────────────────────────────────────

pub struct WeightRegistry {
    pub name_mapping: HashMap<String, String>,
    pub constants: HashMap<String, ConstantTensor>,
}

#[derive(Debug, Clone)]
pub struct ConstantTensor {
    pub dtype: Dtype,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

// ── SFCF binary read helpers (private, used only by load_registry_from_dylib) ──

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, anyhow::Error> {
    if *pos + 4 > data.len() {
        anyhow::bail!("truncated at pos {} (need u32)", pos);
    }
    let val = u32::from_le_bytes(data[*pos..*pos + 4].try_into()?);
    *pos += 4;
    Ok(val)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, anyhow::Error> {
    if *pos + 8 > data.len() {
        anyhow::bail!("truncated at pos {} (need u64)", pos);
    }
    let val = u64::from_le_bytes(data[*pos..*pos + 8].try_into()?);
    *pos += 8;
    Ok(val)
}

fn read_string(data: &[u8], pos: &mut usize) -> Result<String, anyhow::Error> {
    let len = read_u32(data, pos)? as usize;
    if *pos + len > data.len() {
        anyhow::bail!("truncated string at pos {} (need {} bytes)", pos, len);
    }
    let s = String::from_utf8(data[*pos..*pos + len].to_vec())?;
    *pos += len;
    Ok(s)
}

// ── Dylib loading ─────────────────────────────────────────────────

/// Load the weight registry from a compiled dylib's legacy SFCF symbols.
///
/// Reads ``serveforge_constants_data`` and ``serveforge_constants_size``
/// from the dylib and parses the embedded SFCF binary blob.
///
/// Used by ``dump_weights_runner`` for model introspection.  The primary
/// executor path uses the proto ABI (``sfa_weights``) instead.
pub fn load_registry_from_dylib(
    lib: &libloading::Library,
) -> Result<(WeightRegistry, usize, u32), anyhow::Error> {
    let data_ptr: *const u8 = {
        // SAFETY: libloading::Symbol::get() is safe for the dylib's lifetime.
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
    let data: &[u8] = unsafe { std::slice::from_raw_parts(data_ptr, size_val as usize) };

    // ── Inline SFCF binary parsing ──────────────────────────────
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

    let nm_count = read_u32(data, &mut pos)? as usize;
    let mut name_mapping = HashMap::with_capacity(nm_count);
    for _ in 0..nm_count {
        let compiled = read_string(data, &mut pos)?;
        let hf_key = read_string(data, &mut pos)?;
        name_mapping.insert(compiled, hf_key);
    }

    let const_count = read_u32(data, &mut pos)? as usize;
    let mut constants = HashMap::with_capacity(const_count);
    for _ in 0..const_count {
        let name = read_string(data, &mut pos)?;
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
            shape.push(read_u64(data, &mut pos)? as usize);
        }
        let data_len = read_u64(data, &mut pos)? as usize;
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

    pub fn name_mapping(&self) -> &HashMap<String, String> {
        &self.registry.name_mapping
    }

    pub fn constants(&self) -> &HashMap<String, ConstantTensor> {
        &self.registry.constants
    }

    /// Look up a CachedTensorInfo by compiled weight name for dtype inspection.
    #[allow(dead_code)]
    pub(crate) fn get_weight_info(&self, compiled_name: &str) -> Option<&CachedTensorInfo> {
        let hf_key = self.registry.name_mapping.get(compiled_name)?;
        self.safetensors_index.get(hf_key)
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
