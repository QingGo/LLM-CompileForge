"""Tests for SD-PA boundary splitting in the compiler.

When CachePolicy with scaled_dot_product_attention intercepts is provided,
each transformer layer should be split into TWO functions:
  main_Xa (QKV proj): outputs Q, K, V. K/V get consumed_internally=True.
  main_Xb (Attn+FFN): takes Q, K, V as inputs. Outputs hidden_state.

Without CachePolicy (or without SDPA intercepts), behavior is unchanged.
"""

from __future__ import annotations

import tempfile

import pytest
import torch

from compiler.cache_policy import CachePolicy


class _TinySDPALayer(torch.nn.Module):
    """A single transformer-style layer with scaled_dot_product_attention.

    QKV projection + SDPA + output projection.  No FFN to keep minimal.
    """

    def __init__(self, dim: int = 16, n_heads: int = 2) -> None:
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.q_proj = torch.nn.Linear(dim, dim)
        self.k_proj = torch.nn.Linear(dim, dim)
        self.v_proj = torch.nn.Linear(dim, dim)
        self.out_proj = torch.nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, d = x.shape
        q = self.q_proj(x).reshape(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).reshape(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).reshape(b, s, self.n_heads, self.head_dim).transpose(1, 2)
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(b, s, d)
        return self.out_proj(out)


class _TinyTransformer(torch.nn.Module):
    """Minimal transformer with one SDPA layer for compiler testing."""

    def __init__(self, dim: int = 16, n_heads: int = 2) -> None:
        super().__init__()
        self.layer = _TinySDPALayer(dim=dim, n_heads=n_heads)
        self.norm = torch.nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.layer(x))


# ── Test: consumed_internally is set correctly ────────────────


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_consumed_internally_with_cache_policy() -> None:
    """Compile a tiny SDPA model with CachePolicy — verify consumed_internally=True."""
    from compiler.pipeline import compile_mlir

    dim = 16
    model = _TinyTransformer(dim=dim).eval()

    batch, seq = 1, 4
    x = torch.randn(batch, seq, dim)

    policy = CachePolicy.for_llama(
        num_layers=1,
        num_kv_heads=2,
        head_dim=dim // 2,
    )

    with tempfile.TemporaryDirectory(prefix="kv_split_") as tmpdir:
        mlir_mod = compile_mlir(
            model,
            example_args=(x,),
            output_dir=tmpdir,
            apply_fusion=False,
            cache_policy=policy,
        )

        # At least one function output must have consumed_internally=True
        has_internal = any(
            consumed_internally
            for func in mlir_mod.functions
            for _, _, consumed_internally in func.outputs
        )
        assert has_internal, (
            "Expected at least one function output with "
            "consumed_internally=True when CachePolicy with "
            "SDPA intercepts is provided"
        )


# ── Test: function count increases with SDPA split ────────────


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_function_count_increases_with_sdpa_split() -> None:
    """With CachePolicy, SDPA layers are split → more functions than without."""
    from compiler.pipeline import compile_mlir

    dim = 16
    model = _TinyTransformer(dim=dim).eval()

    batch, seq = 1, 4
    x = torch.randn(batch, seq, dim)

    # Compile WITHOUT cache_policy first
    with tempfile.TemporaryDirectory(prefix="kv_no") as tmpdir1:
        mlir_no = compile_mlir(
            model,
            example_args=(x,),
            output_dir=tmpdir1,
            apply_fusion=False,
        )
        count_no_policy = len(mlir_no.functions)

    # Compile WITH cache_policy
    policy = CachePolicy.for_llama(
        num_layers=1,
        num_kv_heads=2,
        head_dim=dim // 2,
    )
    with tempfile.TemporaryDirectory(prefix="kv_yes") as tmpdir2:
        mlir_yes = compile_mlir(
            model,
            example_args=(x,),
            output_dir=tmpdir2,
            apply_fusion=False,
            cache_policy=policy,
        )
        count_with_policy = len(mlir_yes.functions)

    # With SDPA split, we should have MORE functions
    assert count_with_policy > count_no_policy, (
        f"Expected more functions with CachePolicy ({count_with_policy} > {count_no_policy}), "
        "but function count did not increase. SD-PA split may not have triggered."
    )

    # With SDPA split, at least 2 functions (QKV split from Attn+FFN)
    assert count_with_policy >= 2, (
        f"Expected at least 2 functions with CachePolicy, got {count_with_policy}"
    )


# ── Test: K/V outputs have consumed_internally=True ───────────


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_kv_outputs_marked_consumed_internally() -> None:
    """K and V outputs of the 'a' block must be consumed_internally=True."""
    from compiler.pipeline import compile_mlir

    dim = 16
    model = _TinyTransformer(dim=dim).eval()

    batch, seq = 1, 4
    x = torch.randn(batch, seq, dim)

    policy = CachePolicy.for_llama(
        num_layers=1,
        num_kv_heads=2,
        head_dim=dim // 2,
    )

    with tempfile.TemporaryDirectory(prefix="kv_kv_") as tmpdir:
        mlir_mod = compile_mlir(
            model,
            example_args=(x,),
            output_dir=tmpdir,
            apply_fusion=False,
            cache_policy=policy,
        )

        # Find "a" functions (QKV proj) — name ends with 'a'
        a_funcs = [f for f in mlir_mod.functions if f.name.endswith("a")]
        assert len(a_funcs) >= 1, (
            f"Expected at least one '*a' function, found {len(a_funcs)}"
        )

        # Each 'a' function should have at least one consumed_internally=True output
        for func in a_funcs:
            internal_outputs = [
                (name, tp) for name, tp, consumed in func.outputs if consumed
            ]
            assert len(internal_outputs) >= 1, (
                f"Function {func.name} has no consumed_internally=True outputs. "
                f"All outputs: {func.outputs}"
            )


# ── Test: without CachePolicy, no split ───────────────────────


@pytest.mark.unit
@pytest.mark.timeout(30)
def test_no_cache_policy_no_split() -> None:
    """Without CachePolicy (or without SDPA intercepts), behavior is unchanged."""
    from compiler.pipeline import compile_mlir

    dim = 16
    model = _TinyTransformer(dim=dim).eval()

    batch, seq = 1, 4
    x = torch.randn(batch, seq, dim)

    with tempfile.TemporaryDirectory(prefix="kv_no_") as tmpdir:
        mlir_mod = compile_mlir(
            model,
            example_args=(x,),
            output_dir=tmpdir,
            apply_fusion=False,
        )

        # No consumed_internally=True when no CachePolicy
        has_internal = any(
            consumed_internally
            for func in mlir_mod.functions
            for _, _, consumed_internally in func.outputs
        )
        assert not has_internal, (
            "Expected NO consumed_internally=True when CachePolicy is absent"
        )
