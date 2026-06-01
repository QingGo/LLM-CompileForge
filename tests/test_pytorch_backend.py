import pytest
import torch
import torch.nn.functional as F  # noqa: N812
from hal.registry import create, get, list_backends, register

from hal.interface import Buffer, Device, OpExecutor
from hal.pytorch_backend import PyTorchBackend, PyTorchBuffer, PyTorchDevice

# ═══════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════


def _assert_tensors_close(a: torch.Tensor, b: torch.Tensor, atol: float = 1e-5, rtol: float = 1e-4) -> None:
    assert torch.allclose(a, b, atol=atol, rtol=rtol), f"Tensors diverge: max diff {(a - b).abs().max():.6e}"


# ═══════════════════════════════════════════════════════════
# PyTorchDevice
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPyTorchDevice:
    def test_is_device(self):
        d = PyTorchDevice("cpu")
        assert isinstance(d, Device)

    def test_cpu_sync_is_noop(self):
        d = PyTorchDevice("cpu")
        d.synchronize()

    def test_device_type_property(self):
        d = PyTorchDevice("cpu")
        assert d.device_type == "cpu"

    def test_cuda_sync_does_not_crash_when_cuda_unavailable(self):
        d = PyTorchDevice("cuda")
        d.synchronize()


# ═══════════════════════════════════════════════════════════
# PyTorchBuffer
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPyTorchBuffer:
    def test_is_buffer(self):
        t = torch.empty(10)
        buf = PyTorchBuffer(t)
        assert isinstance(buf, Buffer)

    def test_data_ptr_valid(self):
        t = torch.empty(10)
        buf = PyTorchBuffer(t)
        assert buf.data_ptr == t.data_ptr()

    def test_copy_from(self):
        src = torch.tensor([1.0, 2.0, 3.0])
        dst = torch.empty(3)
        PyTorchBuffer(dst).copy_from(src)
        _assert_tensors_close(dst, src)

    def test_copy_to(self):
        src = torch.tensor([4.0, 5.0, 6.0])
        dst = torch.empty(3)
        PyTorchBuffer(src).copy_to(dst)
        _assert_tensors_close(dst, src)

    def test_create_tensor_view_shares_memory(self):
        storage = torch.arange(12, dtype=torch.float32).view(3, 4)
        buf = PyTorchBuffer(storage)
        view = buf.create_tensor((3, 4), torch.float32, "cpu")
        assert view.data_ptr() == storage.data_ptr()
        assert torch.equal(view, storage)

    def test_create_tensor_dtype_mismatch_raises(self):
        t = torch.empty(4, dtype=torch.float32)
        buf = PyTorchBuffer(t)
        with pytest.raises(ValueError, match="dtype mismatch"):
            buf.create_tensor((4,), torch.float64, "cpu")

    def test_create_tensor_device_mismatch_raises(self):
        t = torch.empty(4, device="cpu")
        buf = PyTorchBuffer(t)
        with pytest.raises(ValueError, match="device mismatch"):
            buf.create_tensor((4,), torch.float32, "meta")


# ═══════════════════════════════════════════════════════════
# PyTorchBackend — core
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPyTorchBackendCore:
    def test_is_executor(self):
        b = PyTorchBackend("cpu")
        assert isinstance(b, OpExecutor)

    def test_execute_unknown_op_raises(self):
        b = PyTorchBackend("cpu")
        with pytest.raises(ValueError, match="Unknown op"):
            b.execute("nonexistent", [torch.empty(1)])


# ═══════════════════════════════════════════════════════════
# PyTorchBackend — arithmetic ops
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPyTorchBackendArithmetic:
    def test_add(self):
        b = PyTorchBackend("cpu")
        a = torch.tensor([1.0, 2.0])
        c = torch.tensor([3.0, 4.0])
        out = b.execute("add", [a, c])
        _assert_tensors_close(out, a + c)

    def test_mul(self):
        b = PyTorchBackend("cpu")
        a = torch.tensor([2.0, 3.0])
        c = torch.tensor([4.0, 5.0])
        out = b.execute("mul", [a, c])
        _assert_tensors_close(out, a * c)

    def test_matmul(self):
        b = PyTorchBackend("cpu")
        a = torch.randn(3, 4)
        c = torch.randn(4, 2)
        out = b.execute("matmul", [a, c])
        _assert_tensors_close(out, torch.matmul(a, c))


# ═══════════════════════════════════════════════════════════
# PyTorchBackend — activation ops
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPyTorchBackendActivations:
    def test_gelu(self):
        b = PyTorchBackend("cpu")
        x = torch.randn(4, 8)
        out = b.execute("gelu", [x])
        _assert_tensors_close(out, F.gelu(x))

    def test_silu(self):
        b = PyTorchBackend("cpu")
        x = torch.randn(4, 8)
        out = b.execute("silu", [x])
        _assert_tensors_close(out, F.silu(x))

    def test_softmax_default_dim(self):
        b = PyTorchBackend("cpu")
        x = torch.randn(2, 5)
        out = b.execute("softmax", [x])
        _assert_tensors_close(out, F.softmax(x, dim=-1))

    def test_softmax_explicit_dim(self):
        b = PyTorchBackend("cpu")
        x = torch.randn(2, 5)
        out = b.execute("softmax", [x], dim=0)
        _assert_tensors_close(out, F.softmax(x, dim=0))


# ═══════════════════════════════════════════════════════════
# PyTorchBackend — normalization
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPyTorchBackendNormalization:
    def test_layer_norm_defaults(self):
        b = PyTorchBackend("cpu")
        x = torch.randn(2, 4, 8)
        out = b.execute("layer_norm", [x])
        ref = F.layer_norm(x, x.shape[-1:], eps=1e-5)
        _assert_tensors_close(out, ref)

    def test_layer_norm_with_weight_bias(self):
        b = PyTorchBackend("cpu")
        x = torch.randn(2, 4, 8)
        weight = torch.ones(8)
        bias = torch.zeros(8)
        out = b.execute("layer_norm", [x, weight, bias])
        ref = F.layer_norm(x, (8,), weight, bias, eps=1e-5)
        _assert_tensors_close(out, ref)

    def test_rms_norm(self):
        b = PyTorchBackend("cpu")
        x = torch.randn(2, 4, 8)
        eps = 1e-5
        out = b.execute("rms_norm", [x])
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        ref = x * torch.rsqrt(variance + eps)
        _assert_tensors_close(out, ref)

    def test_rms_norm_with_weight(self):
        b = PyTorchBackend("cpu")
        x = torch.randn(2, 4, 8)
        weight = torch.randn(8)
        eps = 1e-5
        out = b.execute("rms_norm", [x, weight])
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        ref = (x * torch.rsqrt(variance + eps)) * weight
        _assert_tensors_close(out, ref)


# ═══════════════════════════════════════════════════════════
# PyTorchBackend — shape ops
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPyTorchBackendShape:
    def test_permute(self):
        b = PyTorchBackend("cpu")
        x = torch.randn(2, 3, 4)
        out = b.execute("permute", [x], dims=(2, 0, 1))
        _assert_tensors_close(out, x.permute(2, 0, 1))

    def test_transpose(self):
        b = PyTorchBackend("cpu")
        x = torch.randn(2, 3)
        out = b.execute("transpose", [x], dim0=0, dim1=1)
        _assert_tensors_close(out, x.transpose(0, 1))

    def test_cat_default_dim(self):
        b = PyTorchBackend("cpu")
        a = torch.randn(2, 3)
        c = torch.randn(1, 3)
        out = b.execute("cat", [a, c])
        _assert_tensors_close(out, torch.cat([a, c], dim=0))

    def test_cat_explicit_dim(self):
        b = PyTorchBackend("cpu")
        a = torch.randn(2, 3)
        c = torch.randn(2, 4)
        out = b.execute("cat", [a, c], dim=1)
        _assert_tensors_close(out, torch.cat([a, c], dim=1))

    def test_slice_defaults(self):
        b = PyTorchBackend("cpu")
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = b.execute("slice", [x])
        _assert_tensors_close(out, x)

    def test_slice_dim1_range(self):
        b = PyTorchBackend("cpu")
        x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        out = b.execute("slice", [x], dim=1, start=1, end=3)
        expected = x[:, 1:3]
        _assert_tensors_close(out, expected)

    def test_view(self):
        b = PyTorchBackend("cpu")
        x = torch.arange(6, dtype=torch.float32)
        out = b.execute("view", [x], shape=(2, 3))
        _assert_tensors_close(out, x.view(2, 3))


# ═══════════════════════════════════════════════════════════
# PyTorchBackend — attention
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestPyTorchBackendAttention:
    def test_scaled_dot_product_attention(self):
        b = PyTorchBackend("cpu")
        q = torch.randn(1, 1, 2, 4)
        k = torch.randn(1, 1, 2, 4)
        v = torch.randn(1, 1, 2, 4)
        out = b.execute("scaled_dot_product_attention", [q, k, v])
        ref = F.scaled_dot_product_attention(q, k, v)
        _assert_tensors_close(out, ref)

    def test_sdpa_causal(self):
        b = PyTorchBackend("cpu")
        q = torch.randn(1, 1, 4, 8)
        out = b.execute("scaled_dot_product_attention", [q, q, q], is_causal=True)
        ref = F.scaled_dot_product_attention(q, q, q, is_causal=True)
        _assert_tensors_close(out, ref)

    def test_sdpa_scale_passthrough(self):
        """Regression: scale kwarg must be passed to F.sdpa, not ignored.
        When the IR sets scale=1.0 (Q is pre-scaled), SDPA should not apply
        its own default scale.  Bug: opt_125m_dynamic cos was 0.92 because
        scale was applied twice (Q pre-scale × SDPA default scale)."""
        b = PyTorchBackend("cpu")
        q = torch.randn(1, 1, 4, 8)
        k = torch.randn(1, 1, 4, 8)
        v = torch.randn(1, 1, 4, 8)
        # Pre-scale Q (as dynamic-shape export does)
        q_scaled = q * (1.0 / (8 ** 0.5))
        # SDPA with scale=1.0 on pre-scaled Q should match PyTorch reference
        out = b.execute("scaled_dot_product_attention", [q_scaled, k, v], scale=1.0)
        ref = F.scaled_dot_product_attention(q_scaled, k, v, scale=1.0)
        _assert_tensors_close(out, ref)
        # Pre-scaled Q + scale=1.0 must equal unscaled Q + default scale
        # (This equivalence is why the IR pre-scales Q and sets scale=1.0)
        out_default = b.execute("scaled_dot_product_attention", [q, k, v])
        _assert_tensors_close(out, out_default)
        # Verify kwarg passthrough: default scale (no kwarg) uses PyTorch default
        ref_default = F.scaled_dot_product_attention(q, k, v)
        _assert_tensors_close(out_default, ref_default)


# ═══════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════


@pytest.mark.unit
class TestRegistry:
    def test_list_backends_empty_initially(self):
        names = list_backends()
        assert isinstance(names, list)

    def test_register_and_list(self):
        register("test_pytorch", lambda **kw: PyTorchBackend(device="cpu"))
        assert "test_pytorch" in list_backends()

    def test_register_duplicate_raises(self):
        register("test_pytorch_dup", lambda **kw: PyTorchBackend(device="cpu"))
        with pytest.raises(ValueError, match="already registered"):
            register("test_pytorch_dup", lambda **kw: PyTorchBackend(device="cpu"))

    def test_get_unknown_raises(self):
        with pytest.raises(KeyError, match="not found"):
            get("nonexistent_backend")

    def test_create_instantiates_backend(self):
        register("test_create", lambda **kw: PyTorchBackend(device="cpu"))
        backend = create("test_create")
        assert isinstance(backend, OpExecutor)
        out = backend.execute("add", [torch.tensor([1.0]), torch.tensor([2.0])])
        _assert_tensors_close(out, torch.tensor([3.0]))

    def test_list_backends_is_sorted(self):
        register("z_backend", lambda **kw: PyTorchBackend(device="cpu"))
        register("a_backend", lambda **kw: PyTorchBackend(device="cpu"))
        names = list_backends()
        # At minimum: already-registered backends should appear in sorted order
        assert names.index("a_backend") < names.index("z_backend")
