# External Memory Fusion Contract

This document defines the runtime contract for auditable n-gram external memory
fusion. It is the deployment-facing version of the qwen35-ple PLE Projector
design and should be mirrored by the Rust runtime and MLIR compiler.

## 1. Logit correction formula

Given base logits `p_b`, a sparse memory distribution `p_m`, and a learned
correction `(scale, bias)`, the runtime computes:

```
fused_logits[t] =
    base_logits[t]
    + scale * (log(p_m[t]) / temperature)
    + bias
```

Only tokens in the memory support are modified; all other logits are preserved.

## 2. Memory features

For each decoded position, the runtime should expose:

- `matched_order`
- `base_entropy`
- `memory_entropy`
- `density_ratio`
- `base_top1_prob`
- `memory_top1_prob`
- `memory_top1_agree_base`

These features are consumed by both the learned PLE Projector and the token
safety policy.

## 3. Safety gate

The runtime should support a density-ratio gate:

```
active = E_{p_m}[log(p_m / p_b)] >= threshold
```

and a learned token policy:

```
g = sigmoid(       normalized_features + b)
apply_memory = g >= tau
```

If the policy disables memory, the original base logits are used unchanged.

## 4. Python reference

`python_runtime/memory_fusion.py` is the authoritative NumPy reference. The
Rust runtime should match its numeric behavior for the supported logit
correction and gate modes.

## 5. Auditability

Every fused token should be able to report:

- whether memory was active
- matched n-gram order
- memory distribution support
- scale/bias used
- policy probability
