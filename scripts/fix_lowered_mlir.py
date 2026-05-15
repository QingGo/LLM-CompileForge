"""Fix lowered MLIR text to pass verifier for tensor.extract operations."""

import re
import sys


def fix_extract_issues(text: str) -> str:
    """Fix tensor.extract ops with wrong number of indices."""

    lines = text.split("\n")
    fixed = []
    # Track inserted zero constants to reuse
    zero_idx = 0

    for line in lines:
        # Pattern 1: tensor.extract(%X) from a multi-dim tensor (no indices)
        # %X = "tensor.extract"(%Y) : (tensor<DIMSf32>) -> f32
        m = re.search(
            r'(%\w+)\s*=\s*"tensor\.extract"\((%\w+)\)\s*:\s*\(tensor<(.+?)xf32>\)\s*->\s*f32',
            line,
        )
        if m:
            lhs, tensor_val, dims_str = m.group(1), m.group(2), m.group(3)
            dim_parts = [d for d in dims_str.split("x") if d]
            if dim_parts and dim_parts[0]:  # Has dimensions (not scalar)
                rank = len(dim_parts)
                # Need to add zero indices for all dims
                zero_vals = []
                for i in range(rank):
                    zero_idx += 1
                    z = f"%z{zero_idx}"
                    fixed.append(f"      {z} = \"arith.constant\"() <{{value = 0 : index}}> : () -> index")
                    zero_vals.append(z)
                indices_str = ", ".join(zero_vals)
                new_line = (
                    f'{lhs} = "tensor.extract"({tensor_val}, {indices_str})'
                    f" : (tensor<{dims_str}xf32>, {', '.join(['index'] * rank)}) -> f32"
                )
                fixed.append(new_line)
                continue

        # Pattern 2: tensor.extract(%X, %i, %j) from a scalar tensor (should have 0 indices)
        m2 = re.search(
            r'(%\w+)\s*=\s*"tensor\.extract"\((%\w+),\s*(%\w+),\s*(%\w+)\)\s*:\s*\(tensor<f32>,\s*index,\s*index\)\s*->\s*f32',
            line,
        )
        if m2:
            lhs, tensor_val, idx1, idx2 = m2.group(1), m2.group(2), m2.group(3), m2.group(4)
            new_line = (
                f'{lhs} = "tensor.extract"({tensor_val})'
                f" : (tensor<f32>) -> f32"
            )
            fixed.append(new_line)
            continue

        fixed.append(line)

    return "\n".join(fixed)


if __name__ == "__main__":
    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else input_path
    with open(input_path) as f:
        text = f.read()
    fixed = fix_extract_issues(text)
    with open(output_path, "w") as f:
        f.write(fixed)
    print(f"Fixed {input_path} -> {output_path}")
