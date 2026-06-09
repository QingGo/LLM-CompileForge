# ruff: noqa: E501
"""Unit tests for verify_weight_promotion_order.

Tests the weight name consistency verification used by compile_dylib.py
to ensure sf-promote-weights preserves the correct weight ordering.
"""

from compiler.dialect.mlir_op_types import MlirFunction, MlirModule, MlirOp
from scripts.checks.verify_weight_consistency import (
    verify_weight_promotion_order,
)


def _make_lowered_text(func_entries: list[tuple[str, list[str]]]) -> str:
    """Build minimal lowered MLIR text with sf.weight_names on func ops.

    Args:
        func_entries: list of (func_name, weight_names_list)
    """
    lines = ["module {"]
    for func_name, wnames in func_entries:
        names_str = ", ".join(f'"{n}"' for n in wnames)
        lines.append(
            f'  "func.func"() <{{sym_name = "{func_name}", function_type = () -> ()}}>'
            f' {{sf.weight_names = [{names_str}]}}'
            f' {{^bb0: return}})'
        )
    lines.append("}")
    return "\n".join(lines) + "\n"


class TestMain0WeightCount:
    def test_exact_match(self):
        """main_0 with 2 weight ops matching lowered IR — no errors."""
        module = MlirModule(
            functions=[
                MlirFunction(
                    name="main_0",
                    inputs=[],
                    outputs=[],
                    ops=[
                        MlirOp(
                            name="sf.weight", dialect="sf", op_name="weight",
                            operands=[], results=["w0"],
                            attributes={"name": "w_proj"},
                        ),
                        MlirOp(
                            name="sf.weight", dialect="sf", op_name="weight",
                            operands=[], results=["w1"],
                            attributes={"name": "w_bias"},
                        ),
                    ],
                ),
            ],
        )
        lowered = _make_lowered_text([("main_0", ["w_proj", "w_bias"])])
        errors = verify_weight_promotion_order(module, lowered)
        assert errors == [], f"expected no errors, got: {errors}"

    def test_count_mismatch(self):
        """main_0 has 3 weight ops but lowered IR has 2 names — should error."""
        module = MlirModule(
            functions=[
                MlirFunction(
                    name="main_0",
                    inputs=[],
                    outputs=[],
                    ops=[
                        MlirOp(
                            name="sf.weight", dialect="sf", op_name="weight",
                            operands=[], results=["w0"],
                            attributes={"name": "a"},
                        ),
                        MlirOp(
                            name="sf.weight", dialect="sf", op_name="weight",
                            operands=[], results=["w1"],
                            attributes={"name": "b"},
                        ),
                        MlirOp(
                            name="sf.weight", dialect="sf", op_name="weight",
                            operands=[], results=["w2"],
                            attributes={"name": "c"},
                        ),
                    ],
                ),
            ],
        )
        lowered = _make_lowered_text([("main_0", ["a", "b"])])
        errors = verify_weight_promotion_order(module, lowered)
        assert len(errors) > 0, f"expected errors for count mismatch, got none"


class TestNonMain0WeightNames:
    def test_exact_match(self):
        """main_1 with weight_names matching lowered IR — no errors."""
        module = MlirModule(
            functions=[
                MlirFunction(
                    name="main_1",
                    inputs=[],
                    outputs=[],
                    weight_names=["q_proj_weight", "k_proj_weight"],
                ),
            ],
        )
        lowered = _make_lowered_text(
            [("main_1", ["q_proj_weight", "k_proj_weight"])]
        )
        errors = verify_weight_promotion_order(module, lowered)
        assert errors == [], f"expected no errors, got: {errors}"

    def test_name_mismatch(self):
        """main_1 has different weight_names than lowered IR — should error."""
        module = MlirModule(
            functions=[
                MlirFunction(
                    name="main_1",
                    inputs=[],
                    outputs=[],
                    weight_names=["q_proj_weight"],
                ),
            ],
        )
        lowered = _make_lowered_text([("main_1", ["different_weight"])])
        errors = verify_weight_promotion_order(module, lowered)
        assert len(errors) > 0, f"expected errors for name mismatch, got none"


class TestMultiFunction:
    def test_multi_function_all_match(self):
        """Two functions, both matching lowered IR — no errors."""
        module = MlirModule(
            functions=[
                MlirFunction(
                    name="main_0",
                    inputs=[],
                    outputs=[],
                    ops=[
                        MlirOp(
                            name="sf.weight", dialect="sf", op_name="weight",
                            operands=[], results=["w0"],
                            attributes={"name": "embed"},
                        ),
                    ],
                ),
                MlirFunction(
                    name="main_1",
                    inputs=[],
                    outputs=[],
                    weight_names=["embed"],
                ),
            ],
        )
        lowered = _make_lowered_text([
            ("main_0", ["embed"]),
            ("main_1", ["embed"]),
        ])
        errors = verify_weight_promotion_order(module, lowered)
        assert errors == [], f"expected no errors, got: {errors}"

    def test_missing_function_in_lowered(self):
        """Function present in module but not in lowered IR — should error."""
        module = MlirModule(
            functions=[
                MlirFunction(
                    name="main_0",
                    inputs=[],
                    outputs=[],
                    ops=[
                        MlirOp(
                            name="sf.weight", dialect="sf", op_name="weight",
                            operands=[], results=["w0"],
                            attributes={"name": "embed"},
                        ),
                    ],
                ),
                MlirFunction(
                    name="main_1",
                    inputs=[],
                    outputs=[],
                    weight_names=["embed"],
                ),
            ],
        )
        lowered = _make_lowered_text([("main_0", ["embed"])])
        errors = verify_weight_promotion_order(module, lowered)
        assert len(errors) > 0, f"expected errors for missing main_1, got none"
