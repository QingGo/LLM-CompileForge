"""State-aware model export — monkey-patch layer forwards for tensor state I/O.

Instead of using transformers' complex Cache class hierarchy, we replace
state reads/writes with module-level tensor attributes so torch.export
captures explicit state tensor I/O.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def prepare_qwen_state_export(
    model: nn.Module,
) -> tuple[nn.Module, list[str], list[str]]:
    """Monkey-patch Qwen3.5 layers to expose state tensors.

    Each stateful layer gets a pair of attributes: _state_in (list of tensors
    to use as inputs) and _state_out (list of tensors to capture as outputs).

    Returns:
        (model, state_input_names, state_output_names)
        Names are like ["conv_state_0", "recurrent_state_0", "kc_3", "vc_3", ...]
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import (  # type: ignore[import-untyped]
        Qwen3_5Attention,
        Qwen3_5GatedDeltaNet,
    )

    state_in_names: list[str] = []
    state_out_names: list[str] = []

    for layer_idx, layer in enumerate(model.model.layers):
        if hasattr(layer, "linear_attn") and isinstance(layer.linear_attn, Qwen3_5GatedDeltaNet):
            _patch_linear_attn_for_export(layer, layer_idx, state_in_names, state_out_names)
        elif hasattr(layer, "self_attn") and isinstance(layer.self_attn, Qwen3_5Attention):
            _patch_full_attn_for_export(layer, layer_idx, state_in_names, state_out_names)

    return model, state_in_names, state_out_names


def _patch_linear_attn_for_export(
    layer: nn.Module, layer_idx: int,
    state_in_names: list[str], state_out_names: list[str],
) -> None:
    """Patch Qwen3_5GatedDeltaNet (linear attention) to use injected tensors."""
    module = layer.linear_attn

    def patched_forward(self, hidden_states, cache_params=None, attention_mask=None):
        from torch.nn.functional import silu, softplus  # noqa: N812
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_mask_to_padding_states

        hidden_states = apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape

        # Read injected state tensors
        conv_state = getattr(self, "_state_in_cs", None)
        recurrent_state = getattr(self, "_state_in_rs", None)
        use_precomputed = conv_state is not None and recurrent_state is not None

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(batch_size, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)

        new_cs = None
        new_rs = None

        if use_precomputed and seq_len == 1:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv, conv_state,
                self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation,
            )
        else:
            if use_precomputed:
                mixed_qkv = torch.cat([conv_state, mixed_qkv], dim=-1)
            if self.causal_conv1d_fn is not None:
                mixed_qkv = self.causal_conv1d_fn(
                    x=mixed_qkv, weight=self.conv1d.weight.squeeze(1),
                    bias=self.conv1d.bias, activation=self.activation, seq_idx=None,
                )
            else:
                mixed_qkv = silu(self.conv1d(mixed_qkv)[:, :, :mixed_qkv.shape[-1]])
            if use_precomputed:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]
            new_cs = self._make_conv_state(mixed_qkv, seq_len)

        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1,
        )
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * softplus(a.float() + self.dt_bias)
        if self.num_v_heads // self.num_k_heads > 1:
            ratio = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(ratio, dim=2)
            key = key.repeat_interleave(ratio, dim=2)

        if use_precomputed and seq_len == 1:
            core_attn_out, last_rec = self.recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state, output_final_state=True,
                use_qk_l2norm_in_kernel=True,
            )
        else:
            core_attn_out, last_rec = self.chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta,
                initial_state=recurrent_state if use_precomputed else None,
                output_final_state=True, use_qk_l2norm_in_kernel=True,
            )
        new_rs = last_rec

        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)

        # Store output state tensors
        object.__setattr__(self, "_state_out_cs", new_cs)
        object.__setattr__(self, "_state_out_rs", new_rs)

        return self.out_proj(core_attn_out)

    def _make_conv_state(self, mixed_qkv, seq_len):
        # Capture conv state: last conv_kernel_size-1 positions
        k = self.conv_kernel_size - 1
        return mixed_qkv[:, :, -min(k, seq_len):].detach() if hasattr(
            mixed_qkv, "detach"
        ) else mixed_qkv[:, :, -min(k, seq_len):]

    module._make_conv_state = _make_conv_state.__get__(module, type(module))
    module.forward = patched_forward.__get__(module, type(module))

    cs_name = f"conv_state_{layer_idx}"
    rs_name = f"recurrent_state_{layer_idx}"
    state_in_names.extend([cs_name, rs_name])
    state_out_names.extend([cs_name, rs_name])


def _patch_full_attn_for_export(
    layer: nn.Module, layer_idx: int,
    state_in_names: list[str], state_out_names: list[str],
) -> None:
    """Patch Qwen3_5Attention (full attention) to use injected K/V tensors."""
    module = layer.self_attn

    def patched_forward(self, hidden_states, position_embeddings,
                        attention_mask=None, position_ids=None,
                        past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        # Full QKV projection (mirrors original Qwen3_5Attention.forward)
        hd = self.head_dim

        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(*input_shape, -1, hd * 2), 2, dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # Read injected KV cache
        k_cache = getattr(self, "_state_in_k", None)
        v_cache = getattr(self, "_state_in_v", None)
        if k_cache is not None and v_cache is not None and k_cache.numel() > 1:
            key_states = torch.cat([k_cache, key_states], dim=2)
            value_states = torch.cat([v_cache, value_states], dim=2)

        # Store new KV for output
        q_len = input_shape[1]
        object.__setattr__(self, "_state_out_k",
                           key_states if q_len == 1 and k_cache is not None and k_cache.numel() > 1 else None)
        object.__setattr__(self, "_state_out_v",
                           value_states if q_len == 1 and v_cache is not None and v_cache.numel() > 1 else None)

        # SDPA
        is_causal = k_cache is None or k_cache.numel() <= 1
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states, key_states, value_states,
            attn_mask=None if is_causal else attention_mask,
            is_causal=is_causal,
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)

    module.forward = patched_forward.__get__(module, type(module))

    kc_name = f"kc_{layer_idx}"
    vc_name = f"vc_{layer_idx}"
    state_in_names.extend([kc_name, vc_name])
    state_out_names.extend([kc_name, vc_name])


def inject_state_tensors(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    """Inject state tensors as module attributes before forward."""
    for layer in model.model.layers:
        for _name, module in layer.named_modules():
            if hasattr(module, "_state_in_cs"):
                for _i, _li in enumerate(layer._state_indices if hasattr(layer, "_state_indices") else []):
                    pass
        # Brute-force: scan all modules with state attributes
        _inject(layer, state_dict)


def _inject(layer: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    for m in layer.modules():
        if m is layer:
            continue
        # LinearAttn
        if hasattr(m, "_state_in_cs"):
            for attr_name, state_key in [
                ("_state_in_cs", "conv_state"), ("_state_in_rs", "recurrent_state"),
            ]:
                full_key = f"{state_key}_{getattr(m, '_layer_idx', '?')}"
                if full_key in state_dict:
                    object.__setattr__(m, attr_name, state_dict[full_key])
                else:
                    object.__setattr__(m, attr_name, None)
        # FullAttn
        if hasattr(m, "_state_in_k"):
            for attr_name, state_key in [
                ("_state_in_k", "kc"), ("_state_in_v", "vc"),
            ]:
                full_key = f"{state_key}_{getattr(m, '_layer_idx', '?')}"
                if full_key in state_dict:
                    object.__setattr__(m, attr_name, state_dict[full_key])
                else:
                    object.__setattr__(m, attr_name, None)


def collect_state_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect state output tensors from all patched layers after forward."""
    result: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(model.model.layers):
        for m in layer.modules():
            if hasattr(m, "_state_out_cs"):
                for out_attr, state_key in [
                    ("_state_out_cs", "conv_state"),
                    ("_state_out_rs", "recurrent_state"),
                ]:
                    val = getattr(m, out_attr, None)
                    if val is not None:
                        result[f"{state_key}_{i}"] = val
            if hasattr(m, "_state_out_k"):
                for out_attr, state_key in [
                    ("_state_out_k", "kc"), ("_state_out_v", "vc"),
                ]:
                    val = getattr(m, out_attr, None)
                    if val is not None:
                        result[f"{state_key}_{i}"] = val
    return result


class StateExportWrapper(nn.Module):
    """Wraps a monkey-patched model for torch.export with state tensor I/O.

    Usage:
        model, in_names, out_names = prepare_qwen_state_export(model)
        wrapper = StateExportWrapper(model, in_names, out_names)
        # Prefill: pass None for all states
        empty_states = [torch.zeros(1) for _ in in_names]
        exported = torch.export.export(wrapper, (input_ids, *empty_states))
    """

    def __init__(self, model: nn.Module, state_in_names: list[str], state_out_names: list[str]):
        super().__init__()
        self._model = model
        self._state_in_names = state_in_names
        self._state_out_names = state_out_names

    def forward(self, input_ids: torch.Tensor, *state_tensors: torch.Tensor):
        # Build state dict from flat tensor list
        state_dict: dict[str, torch.Tensor] = {}
        for name, tensor in zip(self._state_in_names, state_tensors, strict=False):
            if tensor.numel() > 1:
                state_dict[name] = tensor

        _inject_all(self._model, state_dict)
        output = self._model(input_ids)
        logits = output.logits
        out_states = _collect_all(self._model)

        result = [logits]
        for name in self._state_out_names:
            result.append(out_states.get(name, torch.zeros(1)))
        return tuple(result)


def _inject_all(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    for i, layer in enumerate(model.model.layers):
        for m in layer.modules():
            if hasattr(m, "_state_in_cs"):
                cs = state_dict.get(f"conv_state_{i}")
                object.__setattr__(m, "_state_in_cs", cs)
                rs = state_dict.get(f"recurrent_state_{i}")
                object.__setattr__(m, "_state_in_rs", rs)
            if hasattr(m, "_state_in_k"):
                kc = state_dict.get(f"kc_{i}")
                object.__setattr__(m, "_state_in_k", kc)
                vc = state_dict.get(f"vc_{i}")
                object.__setattr__(m, "_state_in_v", vc)


def _collect_all(model: nn.Module) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    for i, layer in enumerate(model.model.layers):
        for m in layer.modules():
            if hasattr(m, "_state_out_cs"):
                cs = getattr(m, "_state_out_cs", None)
                if cs is not None:
                    result[f"conv_state_{i}"] = cs
                rs = getattr(m, "_state_out_rs", None)
                if rs is not None:
                    result[f"recurrent_state_{i}"] = rs
            if hasattr(m, "_state_out_k"):
                kc = getattr(m, "_state_out_k", None)
                if kc is not None:
                    result[f"kc_{i}"] = kc
                vc = getattr(m, "_state_out_v", None)
                if vc is not None:
                    result[f"vc_{i}"] = vc
    return result
