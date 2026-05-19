from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F  # noqa: N812

_log = logging.getLogger("hal.pytorch.attention")


class _AttentionOps:
    def _op_scaled_dot_product_attention(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        query, key, value = inputs[0], inputs[1], inputs[2]
        attn_mask = kwargs.get("attn_mask", None)
        if attn_mask is None and len(inputs) > 3:
            attn_mask = inputs[3]
        if attn_mask is not None and attn_mask.dim() > 4:
            attn_mask = attn_mask.squeeze(2)
        if attn_mask is not None and attn_mask.dim() == 4:
            bsize = query.shape[0]
            if attn_mask.shape[1] > 1 and attn_mask.shape[1] == bsize:
                attn_mask = attn_mask[:, :1, :, :]
        dropout_p = kwargs.get("dropout_p", 0.0)
        is_causal = kwargs.get("is_causal", False)
        scale = kwargs.get("scale", None)
        return F.scaled_dot_product_attention(
            query, key, value, attn_mask=attn_mask, dropout_p=dropout_p,
            is_causal=is_causal, scale=scale,
        )

    def _op_fused_attention_output(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        q_t, k_t, v_t = inputs[0], inputs[1], inputs[2]
        remaining = inputs[3:]
        mask: torch.Tensor | None = None
        o_weight: torch.Tensor | None = None
        o_bias: torch.Tensor | None = None
        roles: list[str] = kwargs.get("input_roles", [])
        if roles and len(roles) == len(remaining):
            for t, role in zip(remaining, roles, strict=False):
                if role == "mask":
                    mask = t
                elif role == "weight":
                    o_weight = t
                elif role == "bias":
                    o_bias = t
        else:
            for t in remaining:
                if t.dim() == 4:
                    mask = t
                elif t.dim() == 2:
                    o_weight = t
                elif t.dim() == 1:
                    o_bias = t
                elif o_weight is None and t.dim() <= 2:
                    o_weight = t
                else:
                    o_bias = t
        if o_weight is None:
            raise ValueError("fused_attention_output: missing output projection weight")
        sdpa_kwargs: dict[str, Any] = {}
        for key in ("scale", "is_causal", "dropout_p"):
            if key in kwargs:
                sdpa_kwargs[key] = kwargs[key]
        attn_out = F.scaled_dot_product_attention(q_t, k_t, v_t, attn_mask=mask, **sdpa_kwargs)
        t_attrs = kwargs.get("fuse_transpose_attrs", {})
        if isinstance(t_attrs, str):
            import ast
            try:
                t_attrs = ast.literal_eval(t_attrs)
            except (ValueError, SyntaxError):
                t_attrs = {}
        dim0 = t_attrs.get("dim0", 1)
        dim1 = t_attrs.get("dim1", 2)
        if isinstance(dim0, int) and isinstance(dim1, int):
            attn_out = attn_out.transpose(dim0, dim1)
        b, s = attn_out.shape[0], attn_out.shape[1]
        hidden = attn_out.shape[2] * attn_out.shape[3]
        attn_out = attn_out.reshape(b, s, hidden)
        return F.linear(attn_out, o_weight, o_bias)

    def _op_fused_attention_block(self, inputs: list[torch.Tensor], **kwargs: Any) -> torch.Tensor:
        roles: list[str] = kwargs.get("input_roles", [])
        rms_input: torch.Tensor | None = None
        rms_weight: torch.Tensor | None = None
        qkv_weight: torch.Tensor | None = None
        o_weight: torch.Tensor | None = None
        o_bias: torch.Tensor | None = None
        mask_4d: torch.Tensor | None = None

        if roles and len(roles) == len(inputs):
            for t, role in zip(inputs, roles, strict=False):
                if role == "rms_input":
                    rms_input = t
                elif role == "rms_weight":
                    rms_weight = t
                elif role == "qkv_weight":
                    qkv_weight = t
                elif role == "o_weight":
                    o_weight = t
                elif role == "o_bias":
                    o_bias = t
                elif role == "mask":
                    mask_4d = t
        else:
            tensors_1d: list[torch.Tensor] = []
            tensors_2d: list[torch.Tensor] = []
            tensors_3d: list[torch.Tensor] = []
            tensors_4d: list[torch.Tensor] = []
            for t in inputs:
                d = t.dim()
                if d == 1:
                    tensors_1d.append(t)
                elif d == 2:
                    tensors_2d.append(t)
                elif d == 4:
                    tensors_4d.append(t)
                else:
                    tensors_3d.append(t)
            rms_input = tensors_3d[0] if tensors_3d else inputs[0]
            rms_weight = tensors_1d[0] if tensors_1d else inputs[1]
            qkv_weight = tensors_2d[0] if len(tensors_2d) >= 1 else inputs[2]
            o_weight = tensors_2d[1] if len(tensors_2d) >= 2 else tensors_2d[0]
            o_bias = tensors_1d[1] if len(tensors_1d) >= 2 else None
            mask_4d = tensors_4d[0] if tensors_4d else None

        if rms_input is None or rms_weight is None or qkv_weight is None or o_weight is None:
            raise ValueError(
                "fused_attention_block: missing required inputs. "
                "Provide input_roles or ensure rms_input, rms_weight, qkv_weight, o_weight "
                "are present."
            )
        rms_normed = rms_input * torch.rsqrt(rms_input.pow(2).mean(-1, keepdim=True) + 1e-6)
        rms_normed = rms_normed * rms_weight
        orig_dtype = rms_normed.dtype
        qkv_out = F.linear(rms_normed.to(qkv_weight.dtype), qkv_weight)
        qkv_out = qkv_out.to(orig_dtype)
        hidden = qkv_out.shape[-1] // 3
        q = qkv_out[..., :hidden]
        k = qkv_out[..., hidden:2 * hidden]
        v = qkv_out[..., 2 * hidden:]

        n_heads: int = kwargs.get("n_heads", 0)
        if n_heads <= 0 and qkv_weight.dim() == 2:
            total_hidden = qkv_weight.shape[0]
            hidden_per_head = qkv_weight.shape[1]
            n_heads = max(1, total_hidden // (3 * hidden_per_head))
        if n_heads <= 0:
            n_heads = 1
        head_dim = hidden // n_heads

        bsz, seq = q.shape[0], q.shape[1]
        q = q.reshape(bsz, seq, n_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(bsz, seq, n_heads, head_dim).permute(0, 2, 1, 3)
        v = v.reshape(bsz, seq, n_heads, head_dim).permute(0, 2, 1, 3)
        sdpa_kwargs: dict[str, Any] = {}
        for key in ("scale", "is_causal", "dropout_p"):
            if key in kwargs:
                sdpa_kwargs[key] = kwargs[key]
        attn_out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=mask_4d, **sdpa_kwargs
        )
        attn_out = attn_out.permute(0, 2, 1, 3).reshape(bsz, seq, hidden * n_heads)
        return F.linear(attn_out, o_weight, o_bias)


from hal.pytorch_backend import _register_handler  # noqa: E402

_register_handler("scaled_dot_product_attention", "_op_scaled_dot_product_attention", 3, 4)
_register_handler("fused_attention_output", "_op_fused_attention_output", 3, None)
_register_handler("fused_attention_block", "_op_fused_attention_block", 3, None)
