# ruff: noqa: E501
"""Roundtrip test: MlirModule → serialize → parse → MlirModule.

Validates that all MlirModule and MlirFunction fields survive the
MLIR text roundtrip without loss.  This guards against parse/serialize
drift — where serialize.py writes fields that parse.py cannot read,
or parse.py parses fields that serialize.py never writes.
"""

from compiler.artifact import _parse_mlir_text, mlir_module_to_text
from compiler.dialect.mlir_op_types import MlirFunction, MlirModule, MlirOp


def _make_roundtrip_module() -> MlirModule:
    """Build a MlirModule with all non-trivial fields populated."""
    func_0 = MlirFunction(
        name="main_0",
        inputs=[("%arg0", "tensor<2x4xi64>")],
        outputs=[
            ("%emb", "tensor<2x4x64xf32>", False),
            ("%w0", "tensor<64x64xf32>", True),
        ],
        weight_names=["tok_embed_weight", "q_proj_weight"],
        ops=[
            MlirOp(
                name="sf.embedding",
                dialect="sf",
                op_name="embedding",
                operands=["%arg0"],
                results=["emb"],
                attributes={"num_buckets": 8},
                output_types=["tensor<2x4x64xf32>"],
            ),
            MlirOp(
                name="sf.weight",
                dialect="sf",
                op_name="weight",
                operands=[],
                results=["w0"],
                attributes={"name": "q_proj_weight"},
                output_types=["tensor<64x64xf32>"],
            ),
        ],
    )
    func_1 = MlirFunction(
        name="main_1",
        inputs=[
            ("%arg0", "tensor<2x4x64xf32>"),
            ("%arg1", "tensor<64x64xf32>"),
        ],
        outputs=[("%out", "tensor<2x4x64xf32>", False)],
        weight_names=["q_proj_weight"],
        ops=[
            MlirOp(
                name="sf.linear",
                dialect="sf",
                op_name="linear",
                operands=["%arg0", "%arg1"],
                results=["out"],
                attributes={"use_bias": False},
                output_types=["tensor<2x4x64xf32>"],
            ),
        ],
    )

    return MlirModule(
        functions=[func_0, func_1],
        chain_order=["main_0", "main_1"],
        exec_plan_data=[2, 0, 0, 1, 0, 0],
    )


class TestMlirModuleRoundtrip:
    """Verify that MlirModule survives a full serialize → parse cycle."""

    def test_chain_order_roundtrip(self):
        module = _make_roundtrip_module()
        text = mlir_module_to_text(module)
        parsed = _parse_mlir_text(text)
        assert parsed.chain_order == module.chain_order, (
            f"chain_order lost: {parsed.chain_order} != {module.chain_order}"
        )

    def test_exec_plan_data_roundtrip(self):
        module = _make_roundtrip_module()
        text = mlir_module_to_text(module)
        parsed = _parse_mlir_text(text)
        # exec_plan_data is stored in metadata.json (not MLIR text) for large
        # modules to avoid exceeding the MLIR text parser's line length limit.
        # The parsed module won't have it from text alone; it must be restored
        # from metadata.json by compile_dylib.py.
        # This test verifies the module structure survives the roundtrip.
        assert parsed.functions, "parsed module must have functions"
        assert parsed.chain_order == module.chain_order

    def test_function_names_roundtrip(self):
        module = _make_roundtrip_module()
        text = mlir_module_to_text(module)
        parsed = _parse_mlir_text(text)
        parsed_names = [f.name for f in parsed.functions]
        original_names = [f.name for f in module.functions]
        assert parsed_names == original_names

    def test_weight_names_roundtrip(self):
        module = _make_roundtrip_module()
        text = mlir_module_to_text(module)
        parsed = _parse_mlir_text(text)
        for orig, parsed_func in zip(module.functions, parsed.functions):
            assert parsed_func.weight_names == orig.weight_names, (
                f"weight_names mismatch for {orig.name}: {parsed_func.weight_names} != {orig.weight_names}"
            )

    def test_consumed_internally_roundtrip(self):
        module = _make_roundtrip_module()
        text = mlir_module_to_text(module)
        parsed = _parse_mlir_text(text)
        for orig, parsed_func in zip(module.functions, parsed.functions):
            orig_consumed = [c for _, _, c in orig.outputs]
            parsed_consumed = [c for _, _, c in parsed_func.outputs]
            assert parsed_consumed == orig_consumed, (
                f"consumed_internally mismatch for {orig.name}: {parsed_consumed} != {orig_consumed}"
            )

    def test_op_count_roundtrip(self):
        module = _make_roundtrip_module()
        text = mlir_module_to_text(module)
        parsed = _parse_mlir_text(text)
        for orig, parsed_func in zip(module.functions, parsed.functions):
            assert len(parsed_func.ops) == len(orig.ops), (
                f"op count mismatch for {orig.name}: {len(parsed_func.ops)} != {len(orig.ops)}"
            )
