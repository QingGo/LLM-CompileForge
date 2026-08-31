"""SFA weight data serialization — protobuf SfaWeightData format.

Produces the protobuf binary blob for the ``sfa_weights`` dylib export symbol
using schema-first serialization to guarantee format compatibility with the
Rust runtime.
"""

from __future__ import annotations

import torch

from gen.proto.python.sfa_abi_pb2 import SfaWeightData  # type: ignore[attr-defined]

# dtype code mapping (must match Rust runtime)
_DTYPE_TO_CODE: dict[torch.dtype, int] = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.int8: 5,
    torch.uint8: 6,
}


def build_weight_data(
    name_mapping: dict[str, str],
    constants: dict[str, torch.Tensor],
) -> bytes:
    """Serialize weight name mapping and constant tensors into protobuf SfaWeightData.

    Args:
        name_mapping: dict {compiled_name: hf_key} — weight name lookup table.
        constants: dict {name: tensor} — constant weight tensors (e.g. attention scale,
            causal mask).  Each tensor is a detached CPU torch.Tensor.

    Returns:
        Complete SfaWeightData protobuf binary as bytes.
    """
    msg = SfaWeightData()

    # Weight entries (sorted for deterministic output)
    for compiled_name, hf_key in sorted(name_mapping.items()):
        entry = msg.weight_entries.add()
        entry.compiled_name = compiled_name
        entry.hf_key = hf_key

    # Constant entries (sorted for deterministic output)
    for name, tensor in sorted(constants.items()):
        entry = msg.constant_entries.add()
        entry.name = name
        entry.dtype_code = _DTYPE_TO_CODE.get(tensor.dtype, 0)
        for dim in tensor.shape:
            entry.shape.append(dim)
        t = tensor.detach().cpu().contiguous()
        if t.dtype == torch.bfloat16:
            # bf16 is not a native numpy dtype; serialize raw uint16 storage.
            entry.data = t.view(torch.uint16).numpy().tobytes()
        else:
            entry.data = t.numpy().tobytes()

    return bytes(msg.SerializeToString())
