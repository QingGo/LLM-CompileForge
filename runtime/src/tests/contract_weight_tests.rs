//! Contract test: weight binding via sfa_abi.proto.
//!
//! Verifies that the runtime correctly consumes the SfaWeightEntry and
//! SfaConstant proto contracts for weight name mapping, embedded
//! constants, and compute graph weight binding classification.
//!
//! Independent of compiler — no compiled dylib or safetensors required.

use prost::Message;
use std::collections::HashMap;

use crate::model::abi::proto::{
    SfaAbiHeader, SfaFuncMeta, SfaInputField, SfaInputKind, SfaSsaRef, SfaWeightData,
    SfaWeightEntry,
};
use crate::model::abi::{build_compute_graph, Binding, SfaWeightProvider, SFA_MAGIC};
use crate::model::compute_graph::InputBinding;
use crate::model::tensor::Dtype;
use crate::model::weight_loader::ConstantTensor;

// ── Test 1: SfaWeightEntry proto roundtrip ──────────────────────────

#[test]
fn test_weight_entry_proto_roundtrip() {
    let original = SfaWeightEntry {
        compiled_name: "wte_weight".to_string(),
        hf_key: "model.decoder.embed_tokens.weight".to_string(),
    };

    let encoded = original.encode_to_vec();
    let decoded = SfaWeightEntry::decode(encoded.as_slice()).expect("decode SfaWeightEntry");

    assert_eq!(decoded.compiled_name, "wte_weight");
    assert_eq!(decoded.hf_key, "model.decoder.embed_tokens.weight");
}

#[test]
fn test_weight_entry_proto_roundtrip_multiple_entries() {
    let entries = vec![
        SfaWeightEntry {
            compiled_name: "wte_weight".to_string(),
            hf_key: "model.decoder.embed_tokens.weight".to_string(),
        },
        SfaWeightEntry {
            compiled_name: "lm_head_weight".to_string(),
            hf_key: "lm_head.weight".to_string(),
        },
        SfaWeightEntry {
            compiled_name: "layers_0_self_attn_q_proj_weight".to_string(),
            hf_key: "model.decoder.layers.0.self_attn.q_proj.weight".to_string(),
        },
    ];

    let wd = SfaWeightData {
        weight_entries: entries.clone(),
        constant_entries: Vec::new(),
    };

    let encoded = wd.encode_to_vec();
    let decoded = SfaWeightData::decode(encoded.as_slice()).expect("decode SfaWeightData");

    assert_eq!(decoded.weight_entries.len(), 3);
    for (i, entry) in decoded.weight_entries.iter().enumerate() {
        assert_eq!(
            entry.compiled_name, entries[i].compiled_name,
            "compiled_name mismatch at index {}",
            i
        );
        assert_eq!(
            entry.hf_key, entries[i].hf_key,
            "hf_key mismatch at index {}",
            i
        );
    }
}

// ── Test 2: SfaWeightProvider name_mapping lookup ───────────────────

#[test]
fn test_weight_provider_name_mapping_lookup() {
    let mut provider = SfaWeightProvider {
        name_mapping: HashMap::new(),
        constants: HashMap::new(),
    };
    provider
        .name_mapping
        .insert("compiled_name".to_string(), "hf_key".to_string());

    let hf_key = provider.name_mapping.get("compiled_name");
    assert_eq!(hf_key, Some(&"hf_key".to_string()));
}

#[test]
fn test_weight_provider_name_mapping_missing_key() {
    let mut provider = SfaWeightProvider {
        name_mapping: HashMap::new(),
        constants: HashMap::new(),
    };
    provider
        .name_mapping
        .insert("wte_weight".to_string(), "model.decoder.embed_tokens.weight".to_string());

    let found = provider.name_mapping.get("wte_weight");
    assert_eq!(found, Some(&"model.decoder.embed_tokens.weight".to_string()));

    let missing = provider.name_mapping.get("nonexistent_weight");
    assert_eq!(missing, None);
}

// ── Test 3: SfaWeightProvider constants lookup ──────────────────────

#[test]
fn test_weight_provider_constants_lookup() {
    let mut provider = SfaWeightProvider {
        name_mapping: HashMap::new(),
        constants: HashMap::new(),
    };
    let const_tensor = ConstantTensor {
        dtype: Dtype::F32,
        shape: vec![1, 4],
        data: vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40], // 1.0, 2.0
    };
    provider
        .constants
        .insert("_const_causal_mask".to_string(), const_tensor.clone());

    let found = provider.constants.get("_const_causal_mask");
    assert!(found.is_some(), "constant should be found");
    let ct = found.unwrap();
    assert_eq!(ct.dtype, Dtype::F32);
    assert_eq!(ct.shape, vec![1, 4]);
    assert_eq!(ct.data.len(), 8);
    assert_eq!(ct.data, vec![0x00, 0x00, 0x80, 0x3f, 0x00, 0x00, 0x00, 0x40]);
}

#[test]
fn test_weight_provider_constants_multiple_dtypes() {
    let mut provider = SfaWeightProvider {
        name_mapping: HashMap::new(),
        constants: HashMap::new(),
    };

    provider.constants.insert(
        "f32_const".to_string(),
        ConstantTensor {
            dtype: Dtype::F32,
            shape: vec![2],
            data: vec![0x00; 8], // 2 × f32
        },
    );
    provider.constants.insert(
        "f16_const".to_string(),
        ConstantTensor {
            dtype: Dtype::F16,
            shape: vec![4],
            data: vec![0x00; 8], // 4 × f16
        },
    );
    provider.constants.insert(
        "i64_const".to_string(),
        ConstantTensor {
            dtype: Dtype::I64,
            shape: vec![1],
            data: vec![0x00; 8], // 1 × i64
        },
    );

    let f32_ct = provider.constants.get("f32_const").unwrap();
    assert_eq!(f32_ct.dtype, Dtype::F32);
    assert_eq!(f32_ct.shape, vec![2]);

    let f16_ct = provider.constants.get("f16_const").unwrap();
    assert_eq!(f16_ct.dtype, Dtype::F16);
    assert_eq!(f16_ct.shape, vec![4]);

    let i64_ct = provider.constants.get("i64_const").unwrap();
    assert_eq!(i64_ct.dtype, Dtype::I64);
    assert_eq!(i64_ct.shape, vec![1]);

    let missing = provider.constants.get("nonexistent");
    assert!(missing.is_none());
}

// ── Test 4: build_compute_graph weight binding classification ───────

#[test]
fn test_build_compute_graph_weight_binding_classification() {
    let mut func = SfaFuncMeta {
        consumed_sub_output_flags: vec![],
        symbol: "layer0".to_string(),
        num_inputs: 2,
        output_rank: 2,
        input_fields: Vec::with_capacity(2),
        outputs: Vec::new(),
    };

    // Weight input with a name.
    func.input_fields.push(SfaInputField {
        kind: SfaInputKind::SfaInputWeight as i32,
        binding: Some(Binding::WeightName("q_proj_weight".to_string())),
        rank: 2,
        dims: vec![768, 768],
        dtype: String::new(),
    });

    // SSA input from previous function.
    func.input_fields.push(SfaInputField {
        kind: SfaInputKind::SfaInputSsa as i32,
        binding: Some(Binding::Ssa(SfaSsaRef {
            producer_func: 0,
            producer_out: 0,
        })),
        rank: 3,
        dims: vec![1, 64, 768],
        dtype: String::new(),
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

    let graph = build_compute_graph(&header, &wp).expect("build_compute_graph");

    assert_eq!(graph.functions.len(), 1);
    let func_def = &graph.functions[0];
    assert_eq!(func_def.inputs.len(), 2);

    // First input should be Weight with name "q_proj_weight".
    match &func_def.inputs[0].0 {
        InputBinding::Weight(name) => {
            assert_eq!(name, "q_proj_weight");
        }
        other => panic!("expected InputBinding::Weight, got {:?}", other),
    }

    // Second input should be SSA with producer_func=0, output_idx=0.
    match &func_def.inputs[1].0 {
        InputBinding::Ssa {
            producer_func,
            output_idx,
        } => {
            assert_eq!(*producer_func, 0);
            assert_eq!(*output_idx, 0);
        }
        other => panic!("expected InputBinding::Ssa, got {:?}", other),
    }
}

#[test]
fn test_build_compute_graph_global_input_classification() {
    // Verify that SfaInputKind::SfaInputGlobal maps to InputBinding::GlobalInput.
    let mut func = SfaFuncMeta {
        consumed_sub_output_flags: vec![],
        symbol: "main_0".to_string(),
        num_inputs: 1,
        output_rank: 2,
        input_fields: Vec::with_capacity(1),
        outputs: Vec::new(),
    };

    func.input_fields.push(SfaInputField {
        kind: SfaInputKind::SfaInputGlobal as i32,
        binding: None,
        rank: 2,
        dims: vec![0, 0],
        dtype: String::new(),
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

    let graph = build_compute_graph(&header, &wp).expect("build_compute_graph");

    assert_eq!(graph.functions.len(), 1);
    match &graph.functions[0].inputs[0].0 {
        InputBinding::GlobalInput => {} // expected
        other => panic!("expected InputBinding::GlobalInput, got {:?}", other),
    }
}
