"""Engine module constants — extracted magic numbers.

Centralizes default values that were previously scattered as literal
integers/floats across the codebase, improving discoverability and
making bulk tuning easier.
"""

DEFAULT_BLOCK_SIZE = 16
DEFAULT_NUM_BLOCKS = 1000
DEFAULT_CHUNK_SIZE = 256
DEFAULT_MAX_BATCH_SIZE = 32
DEFAULT_MAX_TOKENS_PER_STEP = 512
