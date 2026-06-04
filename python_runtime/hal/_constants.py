"""HAL module constants — extracted magic numbers for PyTorch backend.

Centralizes default EPS values, head count defaults, and other
scattered literal values previously hard-coded in pytorch_backend.py.
"""

DEFAULT_EPS = 1e-5
FUSED_ATTENTION_BLOCK_EPS = 1e-6
DEFAULT_N_HEADS = 4
MAX_SSA_CONTEXT = 20
