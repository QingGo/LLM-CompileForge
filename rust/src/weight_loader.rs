//! Safetensors weight loader with embedded name mapping and constants.
//!
//! The compiled .dylib contains two exported symbols:
//!   ``serveforge_constants_data`` — binary blob (name mapping + constants)
//!   ``serveforge_constants_size`` — byte size of the blob
//!
//! The Rust runtime reads these symbols at load time and uses the
//! name mapping to resolve compiled weight names → original HF safetensors
//! keys, falling back to the embedded constants for compiler-synthesized
//! tensors.

use std::collections::HashMap;
use std::ffi::c_void;
use std::path::Path;

use crate::hal_cpu::MemRefDescriptor;

/// Parsed weight lookup table from the embedded binary blob.
pub struct WeightRegistry {
    /// Compiled short name → original HF safetensors key.
    pub name_mapping: HashMap<String, String>,
    /// Embedded constant tensors (compiler-synthesized).
    pub constants: HashMap<String, ConstantTensor>,
}

/// A constant tensor embedded in the .dylib.
#[derive(Debug, Clone)]
pub struct ConstantTensor {
    pub dtype: u8,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl ConstantTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

// ── Binary format parsing (SFCF) ──────────────────────────────────

/// Parse the embedded SFCF binary format.
pub fn parse_embedded(data: &[u8]) -> Result<WeightRegistry, anyhow::Error> {
    if data.len() < 8 {
        anyhow::bail!("embedded data too short: {} bytes", data.len());
    }
    if &data[0..4] != b"SFCF" {
        anyhow::bail!("bad magic: {:?}", &data[0..4]);
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != 1 {
        anyhow::bail!("unsupported binary version: {}", version);
    }

    let mut pos = 8usize;

    // Name mapping
    let nm_count = read_u32(data, &mut pos)? as usize;
    let mut name_mapping = HashMap::with_capacity(nm_count);
    for _ in 0..nm_count {
        let compiled = read_string(data, &mut pos)?;
        let hf_key = read_string(data, &mut pos)?;
        name_mapping.insert(compiled, hf_key);
    }

    // Constants
    let const_count = read_u32(data, &mut pos)? as usize;
    let mut constants = HashMap::with_capacity(const_count);
    for _ in 0..const_count {
        let name = read_string(data, &mut pos)?;
        if pos >= data.len() {
            anyhow::bail!("truncated constant: {}", name);
        }
        let dtype = data[pos];
        pos += 1;
        let ndim = data[pos] as usize;
        pos += 1;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(read_u64(data, &mut pos)? as usize);
        }
        let data_len = read_u64(data, &mut pos)? as usize;
        if pos + data_len > data.len() {
            anyhow::bail!("truncated constant data: {} (need {} bytes)", name, data_len);
        }
        let tensor_data = data[pos..pos + data_len].to_vec();
        pos += data_len;
        constants.insert(name, ConstantTensor {
            dtype,
            shape,
            data: tensor_data,
        });
    }

    Ok(WeightRegistry { name_mapping, constants })
}

fn read_u32(data: &[u8], pos: &mut usize) -> Result<u32, anyhow::Error> {
    if *pos + 4 > data.len() {
        anyhow::bail!("truncated at pos {} (need u32)", pos);
    }
    let val = u32::from_le_bytes(data[*pos..*pos + 4].try_into().unwrap());
    *pos += 4;
    Ok(val)
}

fn read_u64(data: &[u8], pos: &mut usize) -> Result<u64, anyhow::Error> {
    if *pos + 8 > data.len() {
        anyhow::bail!("truncated at pos {} (need u64)", pos);
    }
    let val = u64::from_le_bytes(data[*pos..*pos + 8].try_into().unwrap());
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

/// Load the embedded weight registry from a compiled .dylib.
///
/// Looks up the ``serveforge_constants_data`` and ``serveforge_constants_size``
/// symbols via libloading.
pub fn load_registry_from_dylib(lib: &libloading::Library) -> Result<WeightRegistry, anyhow::Error> {
    // Safety: libloading provides raw symbol addresses; we trust the dylib.
    let data_ptr: *const u8 = {
        let sym: libloading::Symbol<*const c_void> = unsafe {
            lib.get(b"serveforge_constants_data")
                .map_err(|e| anyhow::anyhow!("{}", e))?
        };
        unsafe { *sym as *const u8 }
    };

    let size_val: u64 = {
        let sym = unsafe {
            lib.get::<*const u64>(b"serveforge_constants_size")
                .map_err(|e| anyhow::anyhow!("{}", e))?
        };
        let ptr: *const u64 = unsafe { *sym };
        unsafe { *ptr }
    };

    if size_val == 0 || data_ptr.is_null() {
        anyhow::bail!("embedded data empty or missing");
    }

    let data: &[u8] = unsafe { std::slice::from_raw_parts(data_ptr, size_val as usize) };
    parse_embedded(data)
}

// ── High-level weight lookup ──────────────────────────────────────

/// Combined weight source: mmap'd HF safetensors + embedded constants.
pub struct WeightProvider {
    registry: WeightRegistry,
    /// mmap'd HF safetensors data (if available).
    safetensors_mmap: Option<Vec<u8>>,
}

impl WeightProvider {
    pub fn new(
        registry: WeightRegistry,
        safetensors_path: Option<&Path>,
    ) -> Result<Self, anyhow::Error> {
        let safetensors_mmap = if let Some(p) = safetensors_path {
            let file = std::fs::File::open(p)?;
            // Safety: the file is opened read-only, we only read from the mmap.
            let mmap = unsafe { memmap2::Mmap::map(&file)? };
            Some(mmap.to_vec())
        } else {
            None
        };
        Ok(Self { registry, safetensors_mmap })
    }

    /// Look up a weight by compiled short name.
    ///
    /// Returns a MemRefDescriptor pointing to the weight data, or None
    /// if the name is unknown.
    pub fn get_memref(&self, compiled_name: &str) -> Option<MemRefDescriptor> {
        // --- Embedded constants first ---
        if let Some(ct) = self.registry.constants.get(compiled_name) {
            return Some(Self::constant_as_memref(ct));
        }

        // --- HF safetensors ---
        let hf_key = self.registry.name_mapping.get(compiled_name)?;
        if let Some(ref mmap_data) = self.safetensors_mmap {
            Self::lookup_safetensors(mmap_data, hf_key)
        } else {
            None
        }
    }

    fn constant_as_memref(ct: &ConstantTensor) -> MemRefDescriptor {
        let p = ct.data.as_ptr();
        let rows = *ct.shape.first().unwrap_or(&1);
        let cols = ct.shape.get(1).copied().unwrap_or(1);
        MemRefDescriptor {
            allocated: p as usize as i64,
            aligned: p as *mut c_void,
            offset: 0,
            sizes: [rows as i64, cols as i64],
            strides: [cols as i64, 1],
        }
    }

    fn lookup_safetensors(mmap_data: &[u8], hf_key: &str) -> Option<MemRefDescriptor> {
        // Parse safetensors header to find the tensor offset/size
        if mmap_data.len() < 8 {
            return None;
        }
        let header_len = u64::from_le_bytes(mmap_data[..8].try_into().ok()?) as usize;
        if mmap_data.len() < 8 + header_len {
            return None;
        }
        let header_bytes = &mmap_data[8..8 + header_len];
        let header: serde_json::Value = serde_json::from_slice(header_bytes).ok()?;
        let info = header.get(hf_key)?;
        let _dtype_str = info.get("dtype")?.as_str()?;
        let shape: Vec<usize> = info
            .get("shape")?
            .as_array()?
            .iter()
            .map(|v| v.as_u64().unwrap_or(1) as usize)
            .collect();
        let offsets = info.get("data_offsets")?.as_array()?;
        let start = offsets.first()?.as_u64()? as usize + 8 + header_len;
        let end = offsets.get(1)?.as_u64()? as usize + 8 + header_len;

        if end > mmap_data.len() {
            return None;
        }
        let p = &mmap_data[start..end];
        let rows = *shape.first().unwrap_or(&1);
        let cols = shape.get(1).copied().unwrap_or(1);
        // Note: memref descriptor is for float-like data; we treat all as raw.
        // Element size is determined by dtype — but for now we assume f32.
        Some(MemRefDescriptor {
            allocated: p.as_ptr() as usize as i64,
            aligned: p.as_ptr() as *mut c_void,
            offset: 0,
            sizes: [rows as i64, cols as i64],
            strides: [cols as i64, 1],
        })
    }
}
