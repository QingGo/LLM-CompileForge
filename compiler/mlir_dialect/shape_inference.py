"""Shape inference for sf dialect operations.

Each inference function takes the operation name, input types, and attributes
and returns the output tensor type(s).

Supports both RankedTensorType (from official MLIR) and Python tuples
(shape, element_type_string) for use in fx_to_mlir.py before MLIR objects exist.

This module is a convenience re-export hub.  The actual implementations live in:

- ``shape_inference_utils.py`` — type helpers & broadcasting
- ``shape_inference_activations.py`` — MLIR infer for element-wise, activation,
  matmul, shape manip, reduction, comparison
- ``shape_inference_matmul.py`` — MLIR infer for tensor creation, attention, fused ops, dispatch table
- ``shape_inference_pure.py`` — pure-Python _pure functions & infer_output_shape
"""

from compiler.mlir_dialect.shape_inference_activations import (  # noqa: F401
    _infer_broadcast,
    _infer_compare,
    _infer_elementwise,
    _infer_reduce,
    infer_add,
    infer_cat,
    infer_clamp_min,
    infer_cos,
    infer_cumsum,
    infer_div,
    infer_eq,
    infer_exp,
    infer_expand,
    infer_gelu,
    infer_gt,
    infer_layer_norm,
    infer_le,
    infer_linalg_norm,
    infer_linear,
    infer_logical_and,
    infer_lt,
    infer_matmul,
    infer_max,
    infer_mean,
    infer_mul,
    infer_ne,
    infer_neg,
    infer_pad,
    infer_permute,
    infer_pow,
    infer_relu,
    infer_rms_norm,
    infer_rsqrt,
    infer_select,
    infer_sigmoid,
    infer_silu,
    infer_sin,
    infer_slice,
    infer_softmax,
    infer_softplus,
    infer_sqrt,
    infer_squeeze,
    infer_sub,
    infer_sum,
    infer_tanh,
    infer_transpose,
    infer_tril,
    infer_triu,
    infer_unsqueeze,
    infer_var,
    infer_view,
)
from compiler.mlir_dialect.shape_inference_matmul import (  # noqa: F401
    _INFERENCE_TABLE,
    infer_arange,
    infer_chunk,
    infer_constant,
    infer_conv1d,
    infer_copy_,
    infer_diff,
    infer_einsum,
    infer_embedding,
    infer_expand_as,
    infer_eye,
    infer_full_like,
    infer_fused_attention_block,
    infer_fused_attention_output,
    infer_fused_qkv,
    infer_fused_rms_norm_matmul,
    infer_fused_silu_mul,
    infer_identity,
    infer_index,
    infer_masked_fill,
    infer_new_ones,
    infer_ones_like,
    infer_output_type,
    infer_scaled_dot_product_attention,
    infer_split,
    infer_stack,
    infer_sym_size,
    infer_type_as,
    infer_view_as,
    infer_weight,
    infer_zeros,
    infer_zeros_like,
)
from compiler.mlir_dialect.shape_inference_pure import (  # noqa: F401
    _PURE_TABLE,
    infer_output_shape,
)
from compiler.mlir_dialect.shape_inference_utils import (  # noqa: F401
    _broadcast_shapes,
    _broadcast_types,
    _elt_from_str,
    _elt_type_str,
    _infer_ir_via_pure,
    _make_ranked_type,
    _ranked_shape,
)
