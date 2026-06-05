//! Declarative model cache strategy — serialized from Python at compile
//! time so the Rust runtime reads exactly what the compiler intended.
//!
//! Mirrors `compiler/cache_policy.py::CachePolicy`.

use std::collections::HashMap;

/// Describes one contiguous cache storage region.
///
/// Mirrors `compiler/cache_policy.py::_SlabSpec`.
#[derive(Debug, Clone)]
pub struct SlabSpec {
    pub slab_id: String,
    pub storage: String,
    pub dims: HashMap<String, usize>,
    pub layout: String,
    pub dtype: String,
}

/// Describes when and how the executor intercepts an op for cache I/O.
///
/// Mirrors `compiler/cache_policy.py::_InterceptSpec`.
#[derive(Debug, Clone)]
pub struct InterceptSpec {
    pub slab_id: String,
    pub op_name: String,
    pub direction: String,
    pub source: String,
    pub layer: String,
    pub func_index: usize,
    pub output_index: usize,
}

/// Declarative model cache strategy.
#[derive(Debug, Clone)]
pub struct CachePolicy {
    pub slabs: Vec<SlabSpec>,
    pub intercepts: Vec<InterceptSpec>,
    pub block_size: usize,
    pub max_requests: usize,
}

impl CachePolicy {
    pub fn from_dict(val: &serde_json::Value) -> Result<Self, String> {
        if val.is_null() {
            return Ok(Self::none());
        }
        let obj = val
            .as_object()
            .ok_or_else(|| "expected JSON object for CachePolicy".to_string())?;

        let slabs = match obj.get("slabs") {
            Some(serde_json::Value::Array(arr)) => {
                let mut slabs = Vec::with_capacity(arr.len());
                for sv in arr {
                    let so = sv
                        .as_object()
                        .ok_or_else(|| "expected object in slabs array".to_string())?;
                    let dims_obj = so
                        .get("dims")
                        .and_then(|v| v.as_object())
                        .ok_or_else(|| "expected dims object in slab".to_string())?;
                    let mut dims = HashMap::new();
                    for (k, v) in dims_obj {
                        let n = v
                            .as_u64()
                            .ok_or_else(|| format!("dims.{}: expected integer", k))?
                            as usize;
                        dims.insert(k.clone(), n);
                    }
                    slabs.push(SlabSpec {
                        slab_id: so
                            .get("slab_id")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| "missing slab_id".to_string())?
                            .to_string(),
                        storage: so
                            .get("storage")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| "missing storage".to_string())?
                            .to_string(),
                        dims,
                        layout: so
                            .get("layout")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| "missing layout".to_string())?
                            .to_string(),
                        dtype: so
                            .get("dtype")
                            .and_then(|v| v.as_str())
                            .unwrap_or("float32")
                            .to_string(),
                    });
                }
                slabs
            }
            None => Vec::new(),
            Some(_) => return Err("slabs must be an array".to_string()),
        };

        let intercepts = match obj.get("intercepts") {
            Some(serde_json::Value::Array(arr)) => {
                let mut intercepts = Vec::with_capacity(arr.len());
                for iv in arr {
                    let io = iv
                        .as_object()
                        .ok_or_else(|| "expected object in intercepts array".to_string())?;
                    intercepts.push(InterceptSpec {
                        slab_id: io
                            .get("slab_id")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| "missing slab_id in intercept".to_string())?
                            .to_string(),
                        op_name: io
                            .get("op_name")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| "missing op_name in intercept".to_string())?
                            .to_string(),
                        direction: io
                            .get("direction")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| "missing direction in intercept".to_string())?
                            .to_string(),
                        source: io
                            .get("source")
                            .and_then(|v| v.as_str())
                            .ok_or_else(|| "missing source in intercept".to_string())?
                            .to_string(),
                        layer: io
                            .get("layer")
                            .and_then(|v| v.as_str())
                            .unwrap_or("sequential")
                            .to_string(),
                        func_index: io
                            .get("func_index")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize,
                        output_index: io
                            .get("output_index")
                            .and_then(|v| v.as_u64())
                            .unwrap_or(0) as usize,
                    });
                }
                intercepts
            }
            None => Vec::new(),
            Some(_) => return Err("intercepts must be an array".to_string()),
        };

        let block_size = obj
            .get("block_size")
            .and_then(|v| v.as_u64())
            .unwrap_or(16) as usize;
        let max_requests = obj
            .get("max_requests")
            .and_then(|v| v.as_u64())
            .unwrap_or(256) as usize;

        Ok(Self {
            slabs,
            intercepts,
            block_size,
            max_requests,
        })
    }

    pub fn none() -> Self {
        Self {
            slabs: vec![],
            intercepts: vec![],
            block_size: 16,
            max_requests: 256,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.slabs.is_empty()
    }

    pub fn from_proto(p: &crate::model::abi::proto::SfaCachePolicy) -> Result<Self, String> {
        let slabs: Vec<SlabSpec> = p
            .slabs
            .iter()
            .map(|s| {
                let mut dims = HashMap::with_capacity(4);
                dims.insert("blocks".to_string(), s.num_blocks as usize);
                dims.insert("layers".to_string(), s.num_layers as usize);
                dims.insert("heads".to_string(), s.num_heads as usize);
                dims.insert("dim".to_string(), s.head_dim as usize);
                SlabSpec {
                    slab_id: s.name.clone(),
                    storage: s.slab_type.clone(),
                    dims,
                    layout: s.layout.clone(),
                    dtype: if s.dtype.is_empty() {
                        "float32".to_string()
                    } else {
                        s.dtype.clone()
                    },
                }
            })
            .collect();

        let intercepts: Vec<InterceptSpec> = p
            .intercepts
            .iter()
            .map(|i| InterceptSpec {
                slab_id: i.slab_id.clone(),
                op_name: i.op_name_pattern.clone(),
                direction: i.intercept_type.clone(),
                source: i.source.clone(),
                layer: if i.layer.is_empty() {
                    "sequential".to_string()
                } else {
                    i.layer.clone()
                },
                func_index: i.param_indices.first().copied().unwrap_or(0) as usize,
                output_index: i.param_indices.get(1).copied().unwrap_or(0) as usize,
            })
            .collect();

        Ok(Self {
            slabs,
            intercepts,
            block_size: if p.block_size > 0 {
                p.block_size as usize
            } else {
                16
            },
            max_requests: if p.max_requests > 0 {
                p.max_requests as usize
            } else {
                256
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_policy_none_is_empty() {
        let policy = CachePolicy::none();
        assert!(policy.is_empty());
        assert_eq!(policy.slabs.len(), 0);
        assert_eq!(policy.intercepts.len(), 0);
        assert_eq!(policy.block_size, 16);
        assert_eq!(policy.max_requests, 256);
    }

    #[test]
    fn test_cache_policy_roundtrip_llama() {
        let json = serde_json::json!({
            "slabs": [
                {"slab_id": "k", "storage": "paged",
                 "dims": {"layers": 12, "heads": 8, "dim": 64},
                 "layout": "BNLD", "dtype": "float32"},
                {"slab_id": "v", "storage": "paged",
                 "dims": {"layers": 12, "heads": 8, "dim": 64},
                 "layout": "BNLD", "dtype": "float32"},
            ],
            "intercepts": [
                {"slab_id": "k", "op_name": "scaled_dot_product_attention",
                 "direction": "read_write", "source": "operand[1]", "layer": "sequential",
                 "func_index": 1, "output_index": 1},
                {"slab_id": "v", "op_name": "scaled_dot_product_attention",
                 "direction": "read_write", "source": "operand[2]", "layer": "sequential",
                 "func_index": 1, "output_index": 2},
            ],
            "block_size": 16,
            "max_requests": 256,
        });

        let policy = CachePolicy::from_dict(&json).expect("valid JSON");
        assert_eq!(policy.slabs.len(), 2);
        assert_eq!(policy.intercepts.len(), 2);
        assert!(!policy.is_empty());

        let k_slab = &policy.slabs[0];
        assert_eq!(k_slab.slab_id, "k");
        assert_eq!(k_slab.storage, "paged");
        assert_eq!(k_slab.dims.get("layers"), Some(&12));
        assert_eq!(k_slab.dims.get("heads"), Some(&8));
        assert_eq!(k_slab.dims.get("dim"), Some(&64));
        assert_eq!(k_slab.layout, "BNLD");
        assert_eq!(k_slab.dtype, "float32");

        let v_slab = &policy.slabs[1];
        assert_eq!(v_slab.slab_id, "v");
        assert_eq!(v_slab.dims.get("dim"), Some(&64));

        assert_eq!(policy.intercepts[0].slab_id, "k");
        assert_eq!(policy.intercepts[0].op_name, "scaled_dot_product_attention");
        assert_eq!(policy.intercepts[0].direction, "read_write");
        assert_eq!(policy.intercepts[0].source, "operand[1]");
        assert_eq!(policy.intercepts[0].layer, "sequential");
        assert_eq!(policy.intercepts[0].func_index, 1);
        assert_eq!(policy.intercepts[0].output_index, 1);

        assert_eq!(policy.intercepts[1].slab_id, "v");
        assert_eq!(policy.intercepts[1].source, "operand[2]");
        assert_eq!(policy.intercepts[1].func_index, 1);
        assert_eq!(policy.intercepts[1].output_index, 2);

        assert_eq!(policy.block_size, 16);
        assert_eq!(policy.max_requests, 256);
    }

    #[test]
    fn test_cache_policy_from_null() {
        let policy =
            CachePolicy::from_dict(&serde_json::Value::Null).expect("null should produce none()");
        assert!(policy.is_empty());
    }

    #[test]
    fn test_cache_policy_from_proto_llama() {
        let proto = crate::model::abi::proto::SfaCachePolicy {
            slabs: vec![
                crate::model::abi::proto::SfaSlabSpec {
                    name: "k".to_string(),
                    slab_type: "paged".to_string(),
                    layout: "BNLD".to_string(),
                    dtype: "float32".to_string(),
                    num_blocks: 0,
                    block_size: 16,
                    num_layers: 12,
                    num_heads: 8,
                    head_dim: 64,
                },
                crate::model::abi::proto::SfaSlabSpec {
                    name: "v".to_string(),
                    slab_type: "paged".to_string(),
                    layout: "BNLD".to_string(),
                    dtype: "float32".to_string(),
                    num_blocks: 0,
                    block_size: 16,
                    num_layers: 12,
                    num_heads: 8,
                    head_dim: 64,
                },
            ],
            intercepts: vec![
                crate::model::abi::proto::SfaInterceptSpec {
                    slab_id: "k".to_string(),
                    op_name_pattern: "scaled_dot_product_attention".to_string(),
                    intercept_type: "read_write".to_string(),
                    source: "operand[1]".to_string(),
                    layer: "sequential".to_string(),
                    param_indices: vec![1, 1],
                },
                crate::model::abi::proto::SfaInterceptSpec {
                    slab_id: "v".to_string(),
                    op_name_pattern: "scaled_dot_product_attention".to_string(),
                    intercept_type: "read_write".to_string(),
                    source: "operand[2]".to_string(),
                    layer: "sequential".to_string(),
                    param_indices: vec![1, 2],
                },
            ],
            block_size: 16,
            max_requests: 256,
        };

        let policy = CachePolicy::from_proto(&proto).expect("valid proto");
        assert_eq!(policy.slabs.len(), 2);
        assert_eq!(policy.intercepts.len(), 2);
        assert!(!policy.is_empty());

        let k_slab = &policy.slabs[0];
        assert_eq!(k_slab.slab_id, "k");
        assert_eq!(k_slab.storage, "paged");
        assert_eq!(k_slab.dims.get("layers"), Some(&12));
        assert_eq!(k_slab.dims.get("heads"), Some(&8));
        assert_eq!(k_slab.dims.get("dim"), Some(&64));
        assert_eq!(k_slab.layout, "BNLD");
        assert_eq!(k_slab.dtype, "float32");

        let v_slab = &policy.slabs[1];
        assert_eq!(v_slab.slab_id, "v");
        assert_eq!(v_slab.dims.get("dim"), Some(&64));

        assert_eq!(policy.intercepts[0].slab_id, "k");
        assert_eq!(
            policy.intercepts[0].op_name,
            "scaled_dot_product_attention"
        );
        assert_eq!(policy.intercepts[0].direction, "read_write");
        assert_eq!(policy.intercepts[0].source, "operand[1]");
        assert_eq!(policy.intercepts[0].layer, "sequential");
        assert_eq!(policy.intercepts[0].func_index, 1);
        assert_eq!(policy.intercepts[0].output_index, 1);

        assert_eq!(policy.intercepts[1].slab_id, "v");
        assert_eq!(policy.intercepts[1].source, "operand[2]");
        assert_eq!(policy.intercepts[1].func_index, 1);
        assert_eq!(policy.intercepts[1].output_index, 2);

        assert_eq!(policy.block_size, 16);
        assert_eq!(policy.max_requests, 256);
    }

    #[test]
    fn test_cache_policy_from_proto_empty() {
        let proto = crate::model::abi::proto::SfaCachePolicy {
            slabs: vec![],
            intercepts: vec![],
            block_size: 0,
            max_requests: 0,
        };
        let policy = CachePolicy::from_proto(&proto).expect("empty proto");
        assert!(policy.is_empty());
        assert_eq!(policy.block_size, 16);
        assert_eq!(policy.max_requests, 256);
    }

    #[test]
    fn test_cache_policy_from_proto_default_dtype() {
        let proto = crate::model::abi::proto::SfaCachePolicy {
            slabs: vec![crate::model::abi::proto::SfaSlabSpec {
                name: "k".to_string(),
                slab_type: "paged".to_string(),
                layout: "".to_string(),
                dtype: "".to_string(),
                num_blocks: 0,
                block_size: 16,
                num_layers: 1,
                num_heads: 1,
                head_dim: 1,
            }],
            intercepts: vec![],
            block_size: 32,
            max_requests: 128,
        };
        let policy = CachePolicy::from_proto(&proto).expect("valid");
        assert_eq!(policy.slabs[0].dtype, "float32");
        assert_eq!(policy.block_size, 32);
        assert_eq!(policy.max_requests, 128);
        assert!(!policy.is_empty());
    }

    #[test]
    fn test_cache_policy_from_proto_intercept_defaults() {
        let proto = crate::model::abi::proto::SfaCachePolicy {
            slabs: vec![],
            intercepts: vec![crate::model::abi::proto::SfaInterceptSpec {
                slab_id: "k".to_string(),
                op_name_pattern: "attention".to_string(),
                intercept_type: "write".to_string(),
                source: "output".to_string(),
                layer: "".to_string(),
                param_indices: vec![],
            }],
            block_size: 16,
            max_requests: 256,
        };
        let policy = CachePolicy::from_proto(&proto).expect("valid");
        assert_eq!(policy.intercepts.len(), 1);
        assert_eq!(policy.intercepts[0].layer, "sequential");
        assert_eq!(policy.intercepts[0].func_index, 0);
        assert_eq!(policy.intercepts[0].output_index, 0);
    }
}
