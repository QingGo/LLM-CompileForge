"""Contract tests for hal_ir.schema.json.

Validates that:
  1. The schema file is valid JSON Schema (self-consistent)
  2. A well-formed HAL IR document validates successfully
  3. Deliberately malformed documents are rejected
  4. Optional fields are handled gracefully (schema is permissive)
"""

from __future__ import annotations

import json
from pathlib import Path

import jsonschema
import pytest

_SCHEMA_PATH = Path(__file__).resolve().parent.parent.parent / "include" / "hal_ir.schema.json"
_HAL_IR_EXAMPLE = (
    Path(__file__).resolve().parent.parent.parent
    / "compiled" / "opt_125m_fresh" / "generated" / "hal_ir.json"
)


def _load_schema() -> dict:
    with open(_SCHEMA_PATH) as f:
        return json.load(f)


# ── Schema self-consistency ───────────────────────────────────────────


class TestSchemaSelfConsistency:
    """The schema file itself must be valid JSON Schema."""

    def test_schema_is_valid_json_schema(self) -> None:
        """Schema must pass jsonschema.Draft7Validator.check_schema()."""
        schema = _load_schema()
        # This raises jsonschema.SchemaError if invalid
        jsonschema.Draft7Validator.check_schema(schema)

    def test_schema_has_top_level_properties(self) -> None:
        """Top-level HalIR object must define model_name, num_functions, functions."""
        schema = _load_schema()
        props = schema.get("properties", {})
        assert "model_name" in props
        assert "num_functions" in props
        assert "functions" in props

    def test_schema_has_required_definitions(self) -> None:
        """Must define HalFunction, HalOp, HalTensorDef, HalWeightEntry."""
        schema = _load_schema()
        defs = schema.get("definitions", {})
        assert "HalFunction" in defs
        assert "HalOp" in defs
        assert "HalTensorDef" in defs
        assert "HalWeightEntry" in defs
        assert "ShapeElement" in defs


# ── Positive validation: well-formed HAL IR ───────────────────────────


# A minimal but complete HAL IR document matching the Rust types
_VALID_HAL_IR: dict = {
    "model_name": "test-model",
    "num_functions": 1,
    "functions": [
        {
            "name": "main_0",
            "layer": 0,
            "inputs": [
                {"name": "%arg0", "shape": [1, 768], "dtype": "f32"}
            ],
            "outputs": [
                {"name": "%42", "shape": [1, 768], "dtype": "f32"}
            ],
            "weights": [
                {
                    "name": "model.embed.weight",
                    "shape": ["50272", "768"],
                    "dtype": "f32",
                    "hal_name": "w0",
                    "ssa": "%0",
                }
            ],
            "constants": [],
            "weight_inputs": {"%arg0": "model.embed.weight"},
            "ops": [
                {
                    "op": "matmul",
                    "inputs": ["%arg0", "%0"],
                    "outputs": ["%42"],
                    "input_dtypes": ["f32", "f32"],
                    "output_dtypes": ["f32"],
                }
            ],
        }
    ],
}


_VALID_MINIMAL_HAL_IR: dict = {
    "model_name": "minimal",
    "num_functions": 0,
    "functions": [],
}


class TestPositiveValidation:
    """Well-formed HAL IR documents must validate against the schema."""

    def test_valid_hal_ir_passes(self) -> None:
        """A complete, well-formed HAL IR document passes validation."""
        schema = _load_schema()
        jsonschema.validate(_VALID_HAL_IR, schema)

    def test_minimal_hal_ir_passes(self) -> None:
        """A minimal HAL IR with no functions passes validation."""
        schema = _load_schema()
        jsonschema.validate(_VALID_MINIMAL_HAL_IR, schema)

    def test_hal_op_with_all_optional_fields(self) -> None:
        """HalOp with kind, weight, shape, value, dims, dim passes validation."""
        schema = _load_schema()
        op = {
            "op": "reshape",
            "kind": "reshape",
            "inputs": ["%10"],
            "outputs": ["%11"],
            "weight": "%w0",
            "shape": [1, "?", 64],
            "value": 0.125,
            "dims": [1, 2],
            "dim": 1,
            "input_dtypes": ["f32"],
            "output_dtypes": ["f32"],
        }
        # Just validate the HalOp definition is well-formed, not the whole IR
        validator = jsonschema.Draft7Validator(schema)
        _ = list(validator.iter_errors(op))
        # May fail on HalOp root since it expects full HalIR; validate just
        # nested HalOp shape by wrapping in a function
        func = {
            "name": "test",
            "layer": 0,
            "inputs": [],
            "outputs": [],
            "weights": [],
            "constants": [],
            "weight_inputs": {},
            "ops": [op],
        }
        hal_ir = {
            "model_name": "test",
            "num_functions": 1,
            "functions": [func],
        }
        jsonschema.validate(hal_ir, schema)

    def test_tensor_def_with_consumed_internally(self) -> None:
        """HalTensorDef with consumed_internally=true passes."""
        schema = _load_schema()
        hal_ir = {
            "model_name": "test",
            "num_functions": 1,
            "functions": [
                {
                    "name": "main_1a",
                    "layer": 1,
                    "inputs": [],
                    "outputs": [
                        {"name": "%out", "shape": [1, 768], "dtype": "f32",
                         "consumed_internally": True}
                    ],
                    "weights": [],
                    "constants": [],
                    "weight_inputs": {},
                    "ops": [],
                    "block_table": True,
                }
            ],
        }
        jsonschema.validate(hal_ir, schema)

    def test_layer_null_for_non_decoder(self) -> None:
        """Layer can be null for non-decoder functions."""
        schema = _load_schema()
        hal_ir = {
            "model_name": "test",
            "num_functions": 1,
            "functions": [
                {
                    "name": "preprocess",
                    "layer": None,
                    "inputs": [],
                    "outputs": [],
                    "weights": [],
                    "constants": [],
                    "weight_inputs": {},
                    "ops": [],
                }
            ],
        }
        jsonschema.validate(hal_ir, schema)

    def test_mixed_shape_elements(self) -> None:
        """Shape arrays can mix integers and strings."""
        schema = _load_schema()
        hal_ir = {
            "model_name": "test",
            "num_functions": 1,
            "functions": [
                {
                    "name": "main_0",
                    "layer": 0,
                    "inputs": [
                        {"name": "%arg0", "shape": [1, "?", 768, "?"], "dtype": "f32"}
                    ],
                    "outputs": [],
                    "weights": [],
                    "constants": [],
                    "weight_inputs": {},
                    "ops": [
                        {
                            "op": "reshape",
                            "inputs": ["%arg0"],
                            "outputs": ["%1"],
                            "shape": [1, "?"],
                        }
                    ],
                }
            ],
        }
        jsonschema.validate(hal_ir, schema)


# ── Negative validation: malformed JSON ───────────────────────────────


class TestNegativeValidation:
    """Deliberately malformed HAL IR documents must be rejected."""

    def test_model_name_wrong_type_fails(self) -> None:
        """model_name as integer instead of string fails."""
        schema = _load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "model_name": 42,
                    "num_functions": 0,
                    "functions": [],
                },
                schema,
            )

    def test_num_functions_wrong_type_fails(self) -> None:
        """num_functions as string instead of integer fails."""
        schema = _load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "model_name": "test",
                    "num_functions": "three",
                    "functions": [],
                },
                schema,
            )

    def test_functions_not_array_fails(self) -> None:
        """functions as object instead of array fails."""
        schema = _load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "model_name": "test",
                    "num_functions": 1,
                    "functions": {"main_0": {}},
                },
                schema,
            )

    def test_hal_op_inputs_wrong_type_fails(self) -> None:
        """HalOp with inputs as string instead of array fails."""
        schema = _load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "model_name": "test",
                    "num_functions": 1,
                    "functions": [
                        {
                            "name": "main_0",
                            "layer": 0,
                            "inputs": [],
                            "outputs": [],
                            "weights": [],
                            "constants": [],
                            "weight_inputs": {},
                            "ops": [
                                {
                                    "op": "matmul",
                                    "inputs": "not_an_array",
                                    "outputs": ["%42"],
                                }
                            ],
                        }
                    ],
                },
                schema,
            )

    def test_hal_op_outputs_wrong_type_fails(self) -> None:
        """HalOp with outputs as string instead of array fails."""
        schema = _load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "model_name": "test",
                    "num_functions": 1,
                    "functions": [
                        {
                            "name": "main_0",
                            "layer": 0,
                            "inputs": [],
                            "outputs": [],
                            "weights": [],
                            "constants": [],
                            "weight_inputs": {},
                            "ops": [
                                {
                                    "op": "matmul",
                                    "inputs": ["%0"],
                                    "outputs": "not_an_array",
                                }
                            ],
                        }
                    ],
                },
                schema,
            )

    def test_op_inputs_not_strings_fails(self) -> None:
        """HalOp inputs with non-string elements fails."""
        schema = _load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "model_name": "test",
                    "num_functions": 1,
                    "functions": [
                        {
                            "name": "main_0",
                            "layer": 0,
                            "inputs": [],
                            "outputs": [],
                            "weights": [],
                            "constants": [],
                            "weight_inputs": {},
                            "ops": [
                                {
                                    "op": "matmul",
                                    "inputs": [42],
                                    "outputs": ["%42"],
                                }
                            ],
                        }
                    ],
                },
                schema,
            )

    def test_tensor_def_shape_not_array_fails(self) -> None:
        """HalTensorDef with shape as string fails."""
        schema = _load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "model_name": "test",
                    "num_functions": 1,
                    "functions": [
                        {
                            "name": "main_0",
                            "layer": 0,
                            "inputs": [
                                {"name": "%arg0", "shape": "1x768", "dtype": "f32"}
                            ],
                            "outputs": [],
                            "weights": [],
                            "constants": [],
                            "weight_inputs": {},
                            "ops": [],
                        }
                    ],
                },
                schema,
            )

    def test_weight_shape_not_strings_fails(self) -> None:
        """HalWeightEntry shape with non-string elements fails (weights
        are always string-serialized shapes, integers are rejected)."""
        schema = _load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate(
                {
                    "model_name": "test",
                    "num_functions": 1,
                    "functions": [
                        {
                            "name": "main_0",
                            "layer": 0,
                            "inputs": [],
                            "outputs": [],
                            "weights": [
                                {
                                    "name": "w",
                                    "shape": [42],
                                    "dtype": "f32",
                                    "hal_name": "w0",
                                    "ssa": "%0",
                                }
                            ],
                            "constants": [],
                            "weight_inputs": {},
                            "ops": [],
                        }
                    ],
                },
                schema,
            )

    def test_completely_malformed_not_object(self) -> None:
        """Non-object document (e.g., a string) is rejected."""
        schema = _load_schema()
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate("not an object", schema)


# ── Real-world example validation (if available) ──────────────────────


@pytest.mark.integration
class TestRealWorldExample:
    """Validate against actual compiled hal_ir.json if available."""

    @pytest.fixture(scope="class")
    def hal_ir(self) -> dict:
        if not _HAL_IR_EXAMPLE.exists():
            pytest.skip(f"No compiled hal_ir.json at {_HAL_IR_EXAMPLE}")
        with open(_HAL_IR_EXAMPLE) as f:
            return json.load(f)

    def test_real_hal_ir_validates(self, hal_ir) -> None:
        """The compiled hal_ir.json must conform to the schema."""
        schema = _load_schema()
        jsonschema.validate(hal_ir, schema)

    def test_real_hal_ir_functions_have_ops(self, hal_ir) -> None:
        """Every function must have an ops array."""
        schema = _load_schema()
        for func in hal_ir["functions"]:
            assert isinstance(func["ops"], list)
            jsonschema.validate(
                func,
                schema["definitions"]["HalFunction"],
            )

    def test_real_hal_ir_ops_have_op_field(self, hal_ir) -> None:
        """Every op must have the 'op' string field."""
        schema = _load_schema()
        for func in hal_ir["functions"]:
            for op in func["ops"]:
                assert isinstance(op["op"], str)
                jsonschema.validate(
                    op,
                    schema["definitions"]["HalOp"],
                )
