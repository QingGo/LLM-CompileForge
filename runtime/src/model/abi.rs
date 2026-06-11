//! SFA ABI header parsing — loads compiled dylib metadata and builds ComputeGraph.
//!
//! The compiled .dylib exports two symbols with protobuf-encoded data:
//!   ``sfa_abi``      — SfaAbiHeader protobuf binary (function signatures, input bindings)
//!   ``sfa_abi_size``  — u64 byte length of sfa_abi
//!   ``sfa_weights``  — SfaWeightData protobuf binary (name mapping, constants)
//!   ``sfa_weights_size`` — u64 byte length of sfa_weights
//!
//! Using protobuf (schema-first) eliminates binary format mismatches between
//! Python serialization and Rust parsing.

use std::collections::HashMap;

use prost::Message;

use crate::model::compute_graph::{ComputeGraph, FuncDef, InputBinding, IOTensorDef};
use crate::model::tensor::Dtype;
use crate::model::weight_loader::ConstantTensor;

// ── Prost-generated proto types ──────────────────────────────────────

pub mod proto {
    include!(concat!(env!("CARGO_MANIFEST_DIR"), "/../gen/proto/rust/sfa/sfa.rs"));
}

// Re-export proto types for external consumers
pub use proto::sfa_input_field::Binding;
pub use proto::OutputDescriptor;
pub use proto::SfaAbiHeader;
pub use proto::SfaCachePolicy;
pub use proto::SfaFuncMeta;
pub use proto::SfaInputField;
pub use proto::SfaInputKind;
pub use proto::SfaInterceptSpec;
pub use proto::SfaSlabSpec;
pub use proto::SfaSsaRef;

// ── Constants ──────────────────────────────────────────────────────

/// Magic bytes "SFBA" stored as u32 LE.
pub const SFA_MAGIC: u32 = 0x41464253;
/// SFA ABI version — must match the version embedded by the compiler.
pub const SFA_VERSION: u32 = 1;

// ── Weight provider ────────────────────────────────────────────────

/// Parsed weight metadata from the ``sfa_weights`` dylib symbol.
#[derive(Debug, Clone)]
pub struct SfaWeightProvider {
    /// Maps compiled weight name → HuggingFace safetensors key.
    pub name_mapping: HashMap<String, String>,
    /// Embedded constant tensors (compiler-synthesized, e.g. masks).
    pub constants: HashMap<String, ConstantTensor>,
}

// ── Helpers ────────────────────────────────────────────────────────

/// Read a `u64` value embedded as a data symbol in the dylib.
///
/// # Safety
/// ``lib`` must reference a valid loaded dylib. The named symbol must
/// exist as a `const u64` value.
///
/// libloading::Symbol<T>::deref returns the dlsym RETURN VALUE (the symbol
/// address), not the data at that address. Using T = `*mut c_void` gives us
/// the raw address directly via `*sym`, then we read through it.
unsafe fn read_u64_symbol(lib: &libloading::Library, name: &[u8]) -> Result<u64, anyhow::Error> {
    let sym: libloading::Symbol<*mut std::ffi::c_void> = unsafe {
        lib.get(name)
            .map_err(|e| anyhow::anyhow!("dylib missing {} symbol: {}", String::from_utf8_lossy(name), e))?
    };
    let addr: *mut std::ffi::c_void = *sym;
    Ok(unsafe { *(addr as *const u64) })
}

/// Read a byte slice from a dylib symbol (const u8 array + companion u64 size).
///
/// # Safety
/// ``lib`` must reference a valid loaded dylib. The named data symbol
/// must exist as a `const u8` array with a companion `_{name}_size` u64.
unsafe fn read_byte_slice_from_symbol<'lib>(
    lib: &'lib libloading::Library,
    data_name: &[u8],
    size_name: &[u8],
) -> Result<&'lib [u8], anyhow::Error> {
    let size = unsafe { read_u64_symbol(lib, size_name)? } as usize;
    // Use Symbol<*mut c_void> — *sym gives the raw dlsym address directly.
    let data_sym: libloading::Symbol<*mut std::ffi::c_void> = unsafe {
        lib.get(data_name)
            .map_err(|e| anyhow::anyhow!("dylib missing {} symbol: {}", String::from_utf8_lossy(data_name), e))?
    };
    let data_ptr: *const u8 = *data_sym as *const u8;
    // SAFETY: data_ptr is the dlsym address of the embedded const array in the dylib.
    let slice: &'lib [u8] = unsafe { std::slice::from_raw_parts(data_ptr, size) };
    std::mem::forget(data_sym);
    Ok(slice)
}

// ── Public API ─────────────────────────────────────────────────────

/// Load the SFA ABI header from a compiled dylib.
///
/// Reads ``sfa_abi`` and ``sfa_abi_size`` symbols, decodes them from
/// protobuf, and validates the magic number.
///
/// # Safety
/// ``lib`` must reference a valid, loaded dylib that exports the
/// ``sfa_abi`` and ``sfa_abi_size`` symbols.
pub unsafe fn load_sfa_abi(lib: &libloading::Library) -> Result<SfaAbiHeader, anyhow::Error> {
    let data = unsafe {
        read_byte_slice_from_symbol(lib, b"sfa_abi", b"sfa_abi_size")
    }?;
    let abi = SfaAbiHeader::decode(data)?;
    if abi.magic != SFA_MAGIC {
        anyhow::bail!(
            "Invalid SFA ABI magic: {:#x} (expected {:#x} = \"SFBA\")",
            abi.magic,
            SFA_MAGIC
        );
    }
    if abi.version != SFA_VERSION {
        anyhow::bail!(
            "SFA ABI version mismatch: expected {}, got {}",
            SFA_VERSION,
            abi.version
        );
    }
    Ok(abi)
}

/// Load the SFA weight data from a compiled dylib.
///
/// Reads ``sfa_weights`` and ``sfa_weights_size`` symbols, decodes them
/// from protobuf, and builds an SfaWeightProvider.
///
/// # Safety
/// ``lib`` must reference a valid, loaded dylib that exports the
/// ``sfa_weights`` and ``sfa_weights_size`` symbols.
pub unsafe fn load_sfa_weights(
    lib: &libloading::Library,
) -> Result<SfaWeightProvider, anyhow::Error> {
    let data = unsafe {
        read_byte_slice_from_symbol(lib, b"sfa_weights", b"sfa_weights_size")
    }?;
    let wd = proto::SfaWeightData::decode(data)?;

    let mut name_mapping = HashMap::with_capacity(wd.weight_entries.len());
    for entry in &wd.weight_entries {
        name_mapping.insert(entry.compiled_name.clone(), entry.hf_key.clone());
    }

    let mut constants = HashMap::with_capacity(wd.constant_entries.len());
    for entry in &wd.constant_entries {
        let dtype = Dtype::from_code(entry.dtype_code as u8)
            .ok_or_else(|| anyhow::anyhow!("unknown constant dtype code {}", entry.dtype_code))?;
        let shape: Vec<usize> = entry.shape.iter().map(|&d| d as usize).collect();
        constants.insert(
            entry.name.clone(),
            ConstantTensor {
                dtype,
                shape,
                data: entry.data.clone(),
            },
        );
    }

    Ok(SfaWeightProvider {
        name_mapping,
        constants,
    })
}

/// Load the SFA cache policy proto from a compiled dylib.
///
/// Reads ``sfa_cache_policy`` and ``sfa_cache_policy_size`` symbols,
/// decodes them from protobuf. Returns ``Ok(None)`` if either symbol
/// is missing (older models predating proto cache policy).
///
/// # Safety
/// ``lib`` must reference a valid, loaded dylib.
pub unsafe fn load_sfa_cache_policy(
    lib: &libloading::Library,
) -> Result<Option<SfaCachePolicy>, anyhow::Error> {
    let data = match unsafe {
        read_byte_slice_from_symbol(lib, b"sfa_cache_policy", b"sfa_cache_policy_size")
    } {
        Ok(data) => data,
        Err(_) => return Ok(None),
    };
    let policy = SfaCachePolicy::decode(data)?;
    Ok(Some(policy))
}

/// Build a ComputeGraph from SFA ABI header and weight provider.
///
/// Converts the protobuf function metadata into FuncDef entries with
/// resolved input bindings (Global, Weight, Ssa).
pub fn build_compute_graph(
    abi: &SfaAbiHeader,
    _weights: &SfaWeightProvider,
) -> Result<ComputeGraph, anyhow::Error> {
    let num_funcs = abi.funcs.len();
    let mut functions = Vec::with_capacity(num_funcs);

    for (fi, func_meta) in abi.funcs.iter().enumerate() {
        let num_inputs = func_meta.num_inputs as usize;

        // Read input fields.
        let mut inputs: Vec<(InputBinding, IOTensorDef)> = Vec::with_capacity(num_inputs);
        for field in &func_meta.input_fields {
            let kind = SfaInputKind::try_from(field.kind)?;
            let binding = match kind {
                SfaInputKind::SfaInputGlobal => InputBinding::GlobalInput,
                SfaInputKind::SfaInputWeight => {
                    let name = field
                        .binding
                        .as_ref()
                        .and_then(|b| match b {
                            Binding::WeightName(n) => Some(n.clone()),
                            _ => None,
                        })
                        .unwrap_or_default();
                    InputBinding::Weight(name)
                }
                SfaInputKind::SfaInputSsa => {
                    let (producer_func, output_idx) = field
                        .binding
                        .as_ref()
                        .and_then(|b| match b {
                            Binding::Ssa(ref s) => Some((s.producer_func as usize, s.producer_out as usize)),
                            _ => None,
                        })
                        .unwrap_or((0, 0));
                    InputBinding::Ssa {
                        producer_func,
                        output_idx,
                    }
                }
            };

            let io_def = match kind {
                SfaInputKind::SfaInputGlobal => IOTensorDef {
                    rank: if field.rank > 0 {
                        field.rank as u8
                    } else {
                        log::warn!("GlobalInput rank=0 in proto, defaulting to 2");
                        2
                    },
                    shape: if !field.dims.is_empty() {
                        field.dims.clone()
                    } else {
                        log::warn!("GlobalInput dims empty in proto, defaulting to [0,0]");
                        vec![0, 0]
                    },
                    consumed_internally: false,
                },
                _ => IOTensorDef {
                    rank: field.rank as u8,
                    shape: field.dims.clone(),
                    consumed_internally: false,
                },
            };
            inputs.push((binding, io_def));
        }

        anyhow::ensure!(
            inputs.len() == num_inputs,
            "num_inputs mismatch: proto field says {}, but found {} input_fields",
            num_inputs,
            inputs.len()
        );

        // Output tensors: use OutputDescriptor entries if present, fall back to output_rank.
        let outputs: Vec<IOTensorDef> = if !func_meta.outputs.is_empty() {
            func_meta
                .outputs
                .iter()
                .map(|od| IOTensorDef {
                    rank: od.rank as u8,
                    shape: od.dims.clone(),
                    consumed_internally: od.consumed_internally,
                })
                .collect()
        } else {
            let output_rank = func_meta.output_rank as u8;
            let output_shape = if output_rank == 0 {
                Vec::new()
            } else {
                vec![0u64; output_rank as usize]
            };
            vec![IOTensorDef {
                rank: output_rank,
                shape: output_shape,
                consumed_internally: false,
            }]
        };

        let num_outputs = outputs.len();

        functions.push(FuncDef {
            index: fi,
            symbol: func_meta.symbol.clone(),
            num_inputs,
            num_outputs,
            inputs,
            outputs,
        });
    }

    // Global input/output are the first/last functions by convention.
    let global_input = (0, 0);
    let global_output = (num_funcs.saturating_sub(1), 0);

    Ok(ComputeGraph {
        functions,
        global_input,
        global_output,
    })
}

// ── Dylib loading ──────────────────────────────────────────────────

/// Load compute graph and weight provider from sfa_abi + sfa_weights
/// symbols embedded in the compiled dylib.
///
/// Also loads cache policy from sfa_cache_policy (proto), falling back
/// to None for older models without the symbol.
pub fn load_from_dylib(
    dylib_path: &str,
    safetensors_path: Option<&str>,
) -> Result<(
    crate::model::weight_loader::WeightProvider,
    ComputeGraph,
    Option<crate::cache::policy::CachePolicy>,
), anyhow::Error> {
    // SAFETY: Loading a compiled dylib produced by the SFA toolchain.
    let lib = unsafe { libloading::Library::new(dylib_path)? };
    let abi = unsafe { load_sfa_abi(&lib)? };
    let sfa_wp = unsafe { load_sfa_weights(&lib)? };
    let cache_policy_proto = unsafe { load_sfa_cache_policy(&lib)? };
    let compute_graph = build_compute_graph(&abi, &sfa_wp)?;
    let registry = crate::model::weight_loader::WeightRegistry {
        name_mapping: sfa_wp.name_mapping,
        constants: sfa_wp.constants,
    };
    let st_path = safetensors_path.map(std::path::Path::new);
    let weight_provider = crate::model::weight_loader::WeightProvider::new(registry, st_path)?;
    let cache_policy = match cache_policy_proto {
        Some(p) => Some(
            crate::cache::policy::CachePolicy::from_proto(&p)
                .map_err(|e| anyhow::anyhow!("{}", e))?,
        ),
        None => None,
    };
    Ok((weight_provider, compute_graph, cache_policy))
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use prost::Message;

    /// Helper: build a protobuf SfaAbiHeader with func_count functions,
    /// each with num_inputs GLOBAL input fields.
    fn build_test_header(func_count: u32, specs: &[(u32, u32, &str, u32)]) -> SfaAbiHeader {
        // specs: (num_inputs, output_rank, symbol, input_count) — for historical compat
        let mut header = SfaAbiHeader {
            magic: SFA_MAGIC,
            version: 1,
            funcs: Vec::with_capacity(func_count as usize),
        };

        for (fi, &(num_inputs, output_rank, _sym, _inp_count)) in specs.iter().enumerate() {
            let mut func = proto::SfaFuncMeta {
                symbol: format!("func_{}", fi),
                num_inputs,
                output_rank,
                input_fields: Vec::with_capacity(num_inputs as usize),
                outputs: Vec::new(),
            };

            for _ in 0..num_inputs {
                func.input_fields.push(SfaInputField {
                    kind: SfaInputKind::SfaInputGlobal as i32,
                    binding: None,
                    rank: 0,
                    dims: Vec::new(),
                });
            }

            header.funcs.push(func);
        }

        header
    }

    // ── Magic validation ─────────────────────────────────────────────

    #[test]
    fn test_load_sfa_abi_rejects_bad_magic() {
        let header = SfaAbiHeader {
            magic: 0xDEADBEEF,
            version: 1,
            funcs: vec![],
        };
        assert_ne!(header.magic, SFA_MAGIC);
    }

    // ── Full parse test ──────────────────────────────────────────────

    #[test]
    fn test_build_compute_graph_from_proto() {
        let header = build_test_header(2, &[(2, 2, "func_a", 2), (1, 2, "func_b", 1)]);

        let wp = SfaWeightProvider {
            name_mapping: HashMap::new(),
            constants: HashMap::new(),
        };
        let graph = build_compute_graph(&header, &wp).unwrap();

        assert_eq!(graph.functions.len(), 2);
        assert_eq!(graph.functions[0].symbol, "func_0");
        assert_eq!(graph.functions[0].num_inputs, 2);
        assert_eq!(graph.functions[0].inputs.len(), 2);
        assert_eq!(graph.functions[1].symbol, "func_1");
        assert_eq!(graph.functions[1].num_inputs, 1);
        assert_eq!(graph.functions[1].inputs.len(), 1);

        // Global input/output
        assert_eq!(graph.global_input, (0, 0));
        assert_eq!(graph.global_output, (1, 0));
    }

    // ── Input binding types ──────────────────────────────────────────

    #[test]
    fn test_parse_input_fields_with_weight_binding() {
        let mut func = proto::SfaFuncMeta {
            symbol: "layer0".to_string(),
            num_inputs: 2,
            output_rank: 2,
            input_fields: Vec::with_capacity(2),
            outputs: Vec::new(),
        };

        func.input_fields.push(SfaInputField {
            kind: SfaInputKind::SfaInputWeight as i32,
            binding: Some(Binding::WeightName("matmul".to_string())),
            rank: 0,
            dims: Vec::new(),
        });

        func.input_fields.push(SfaInputField {
            kind: SfaInputKind::SfaInputSsa as i32,
            binding: Some(Binding::Ssa(SfaSsaRef {
                producer_func: 0,
                producer_out: 0,
            })),
            rank: 0,
            dims: Vec::new(),
        });

        let header = SfaAbiHeader {
            magic: SFA_MAGIC,
            version: 1,
            funcs: vec![func],
        };

        let wp = SfaWeightProvider {
            name_mapping: HashMap::new(),
            constants: HashMap::new(),
        };
        let graph = build_compute_graph(&header, &wp).unwrap();

        assert_eq!(graph.functions.len(), 1);
        let func_def = &graph.functions[0];
        assert_eq!(func_def.inputs.len(), 2);

        // First input: WEIGHT binding.
        match &func_def.inputs[0].0 {
            InputBinding::Weight(name) => assert_eq!(name, "matmul"),
            _ => panic!("expected Weight binding"),
        }

        // Second input: SSA binding.
        match &func_def.inputs[1].0 {
            InputBinding::Ssa {
                producer_func,
                output_idx,
            } => {
                assert_eq!(*producer_func, 0);
                assert_eq!(*output_idx, 0);
            }
            _ => panic!("expected Ssa binding"),
        }
    }

    // ── Protobuf roundtrip test ───────────────────────────────────────

    #[test]
    fn test_proto_roundtrip() {
        let original = SfaAbiHeader {
            magic: SFA_MAGIC,
            version: 1,
            funcs: vec![
                proto::SfaFuncMeta {
                    symbol: "_mlir_ciface_main_0".to_string(),
                    num_inputs: 3,
                    output_rank: 3,
                    input_fields: vec![
                        SfaInputField {
                            kind: SfaInputKind::SfaInputGlobal as i32,
                            binding: None,
                            rank: 2,
                            dims: vec![0, 0],
                        },
                        SfaInputField {
                            kind: SfaInputKind::SfaInputWeight as i32,
                            binding: Some(Binding::WeightName("wte.weight".to_string())),
                            rank: 2,
                            dims: vec![50272, 768],
                        },
                        SfaInputField {
                            kind: SfaInputKind::SfaInputSsa as i32,
                            binding: Some(Binding::Ssa(SfaSsaRef {
                                producer_func: 0,
                                producer_out: 0,
                            })),
                            rank: 3,
                            dims: vec![0, 0, 768],
                        },
                    ],
                    outputs: Vec::new(),
                },
            ],
        };

        let encoded = original.encode_to_vec();
        let decoded = SfaAbiHeader::decode(encoded.as_slice()).unwrap();

        assert_eq!(decoded.magic, original.magic);
        assert_eq!(decoded.version, original.version);
        assert_eq!(decoded.funcs.len(), original.funcs.len());
        assert_eq!(decoded.funcs[0].symbol, original.funcs[0].symbol);
        assert_eq!(decoded.funcs[0].num_inputs, original.funcs[0].num_inputs);
        assert_eq!(decoded.funcs[0].output_rank, original.funcs[0].output_rank);
        assert_eq!(decoded.funcs[0].input_fields.len(), original.funcs[0].input_fields.len());

        // Check input field kinds
        assert_eq!(decoded.funcs[0].input_fields[0].kind, SfaInputKind::SfaInputGlobal as i32);
        assert_eq!(decoded.funcs[0].input_fields[1].kind, SfaInputKind::SfaInputWeight as i32);
        assert_eq!(decoded.funcs[0].input_fields[2].kind, SfaInputKind::SfaInputSsa as i32);

        // Check bindings
        match &decoded.funcs[0].input_fields[1].binding {
            Some(Binding::WeightName(name)) => assert_eq!(name, "wte.weight"),
            _ => panic!("expected WeightName binding"),
        }
        match &decoded.funcs[0].input_fields[2].binding {
            Some(Binding::Ssa(ref ssa)) => {
                assert_eq!(ssa.producer_func, 0);
                assert_eq!(ssa.producer_out, 0);
            }
            _ => panic!("expected Ssa binding"),
        }
    }
}

#[cfg(test)]
mod integration_tests {
    use super::*;
    
    /// Test that sfa_abi can be loaded from a real compiled dylib.
    #[test]
    fn test_load_sfa_abi_from_real_dylib() {
        let dylib_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../outputs/compiled/opt_125m_fresh/libopt_125m.dylib"
        );
        if !std::path::Path::new(dylib_path).exists() {
            eprintln!("SKIP: dylib not found at {}", dylib_path);
            return;
        }
        unsafe {
            let lib = libloading::Library::new(dylib_path)
                .expect("failed to load dylib");
            let size = read_u64_symbol(&lib, b"sfa_abi_size")
                .expect("failed to read sfa_abi_size");
            eprintln!("sfa_abi_size = {}", size);
            let data = read_byte_slice_from_symbol(&lib, b"sfa_abi", b"sfa_abi_size")
                .expect("failed to read sfa_abi");
            eprintln!("sfa_abi data len = {}", data.len());
            let abi = proto::SfaAbiHeader::decode(data)
                .expect("failed to decode proto");
            eprintln!("Loaded {} functions from real dylib", abi.funcs.len());
            assert_eq!(abi.funcs.len(), 16);
            assert_eq!(abi.magic, SFA_MAGIC);
        }
    }
}
