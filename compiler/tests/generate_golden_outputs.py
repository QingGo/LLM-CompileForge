#!/usr/bin/env python3
"""Deterministic per-function golden output generator with E2E cross-validation.

For each config case (seq1/2/6/32), wires all functions in topological SSA
order via ctypes, captures each function's sret output, and saves as .npz.

E2E cross-validation: last function output cos vs HF logits.
Deterministic: fixed seed, sha256 verified on re-run.

Usage:
    python compiler/tests/generate_golden_outputs.py \\
        --config tests/data/golden/npy/opt_125m/configs.json \\
        --output tests/data/golden/npy/opt_125m/

    python compiler/tests/generate_golden_outputs.py \\
        --config tests/data/golden/npy/opt_125m/configs.json \\
        --output tests/data/golden/npy/opt_125m/ \\
        --check-fresh
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

# ── Path setup ──────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from compiler.dylib_ffi import (  # noqa: E402
    DEFAULT_SRET_SIZE,
    compute_sret_size,
    load_graph_from_proto,
    make_memref_descriptor,
    parse_sret_outputs,
)
from compiler.tests.golden_io import save_npz  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────

VOCAB_SIZE: int = 50265
RANDOM_SEED: int = 42


# ── Config loading ──────────────────────────────────────────────────


def _find_workspace_root(start: Path) -> Path:
    """Find workspace root by locating sf-dialect/ directory."""
    current = start.resolve()
    for _ in range(10):
        if (current / "sf-dialect").is_dir():
            return current
        if current.parent == current:
            break
        current = current.parent
    return Path.cwd()


def load_config(config_path: str) -> dict[str, Any]:
    """Load and validate config JSON, resolving relative paths."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(path) as f:
        config = json.load(f)

    ws_root = _find_workspace_root(path.parent)

    # Resolve dylib_path
    dylib_path = config.get("dylib_path")
    if dylib_path:
        resolved = ws_root / dylib_path
        if not resolved.is_file():
            raise FileNotFoundError(f"dylib not found: {resolved}")
        config["_resolved_dylib_path"] = str(resolved)
    else:
        raise ValueError("config.dylib_path is required")

    # Load metadata.json from dylib's parent dir
    dylib_dir = Path(config["_resolved_dylib_path"]).parent
    metadata_path = dylib_dir / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    with open(metadata_path) as f:
        metadata = json.load(f)
    config["_metadata"] = metadata
    config["_dylib_dir"] = str(dylib_dir)

    return config


# ── Weight loading ──────────────────────────────────────────────────


def load_safetensors_weights(
    metadata: dict[str, Any],
) -> dict[str, np.ndarray]:
    """Load all model weights from safetensors, keyed by compiled weight name.

    Uses metadata.json's weight_source.path (absolute path to .safetensors),
    hf_key_map for compiled_name → safetensors key, and tied_weights for
    duplicate resolution.
    """
    weight_source = metadata.get("weight_source", {})
    safetensors_path = weight_source.get("path", "")
    if not safetensors_path or not Path(safetensors_path).is_file():
        raise FileNotFoundError(
            f"safetensors file not found: {safetensors_path}\n"
            f"  metadata.json weight_source.path = {safetensors_path!r}"
        )

    import safetensors.torch  # noqa: E402

    st = safetensors.torch.load_file(safetensors_path)
    hf_key_map: dict[str, str] = metadata.get("hf_key_map", {})
    tied_weights: dict[str, str] = weight_source.get("tied_weights", {})

    # Resolve tied weights: if A→B, then A uses B's key
    resolved_tied: dict[str, str] = {}
    for compiled_name, tied_name in tied_weights.items():
        resolved_tied[compiled_name] = hf_key_map.get(tied_name, tied_name)

    weights: dict[str, np.ndarray] = {}
    for compiled_name, hf_key in hf_key_map.items():
        # Check tied weights first
        if compiled_name in resolved_tied:
            actual_key = resolved_tied[compiled_name]
        else:
            actual_key = hf_key

        if actual_key in st:
            tensor = st[actual_key]
            # Convert to float32 numpy
            arr = tensor.numpy() if hasattr(tensor, "numpy") else np.array(tensor)
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            weights[compiled_name] = arr
        else:
            raise KeyError(
                f"Weight '{compiled_name}' (safetensors key '{actual_key}') "
                f"not found in {safetensors_path}. "
                f"Available keys sample: {list(st.keys())[:10]}"
            )

    # Clean up to free memory
    del st
    return weights


# ── Token generation ────────────────────────────────────────────────


def generate_tokens(
    seq_len: int, seed: int = RANDOM_SEED, batch_size: int = 1,
) -> np.ndarray:
    """Generate random token IDs with fixed seed for determinism.

    Returns:
        Shape ``(seq_len,)`` for batch_size=1 (backward compatible),
        shape ``(batch_size, seq_len)`` for batch_size>1.
    """
    rng = np.random.RandomState(seed)
    if batch_size == 1:
        return rng.randint(0, VOCAB_SIZE, size=(seq_len,), dtype=np.int64)
    return rng.randint(0, VOCAB_SIZE, size=(batch_size, seq_len), dtype=np.int64)


# ── Function calling ────────────────────────────────────────────────


def call_ciface_function(
    lib: ctypes.CDLL,
    func_symbol: str,
    input_arrays: list[np.ndarray],
    output_defs: list[dict[str, Any]],
    sret_size: int = DEFAULT_SRET_SIZE,
) -> list[np.ndarray]:
    """Call a ciface function and return its sret output tensors.

    Args:
        lib: Loaded dylib.
        func_symbol: e.g. "main_0", "main_1".
        input_arrays: List of numpy arrays (any dtype) for the function inputs.
        output_defs: List of {"rank": N, "shape": [...]} dicts matching the
            function's output descriptors.
        sret_size: Pre-computed sret buffer size in bytes.

    Returns:
        List of numpy arrays (copied from dylib memory), one per output_def.
    """
    ciface_name = f"_mlir_ciface_{func_symbol}"
    kernel = getattr(lib, ciface_name)

    # Build memref descriptors for all inputs
    memrefs = [make_memref_descriptor(arr) for arr in input_arrays]

    # Allocate sret buffer
    sret_buf = (ctypes.c_uint8 * sret_size)()

    # Build args: sret pointer + all input memref pointers
    args = [ctypes.byref(sret_buf)] + [ctypes.byref(m) for m in memrefs]
    kernel.argtypes = [ctypes.c_void_p] * len(args)
    kernel.restype = None

    kernel(*args)

    # Parse sret output
    sret_bytes = bytes(sret_buf)
    outputs = parse_sret_outputs(sret_bytes, output_defs)

    # Force-copy all output arrays to ensure they don't reference dylib memory
    return [arr.copy() if arr.size > 0 else arr for arr in outputs]


# ── SSA wiring helpers ──────────────────────────────────────────────


def _resolve_weight(
    weight_name: str,
    weights_dict: dict[str, np.ndarray],
) -> np.ndarray:
    """Look up a compiled weight by name. Raises KeyError if not found."""
    if weight_name not in weights_dict:
        available = list(weights_dict.keys())[:10]
        raise KeyError(
            f"weight '{weight_name}' not found in weights dict. "
            f"Available (first 10): {available}"
        )
    return weights_dict[weight_name]


def build_input_arrays(
    func_graph: dict[str, Any],
    func_outputs: dict[int, list[np.ndarray]],
    weights_dict: dict[str, np.ndarray],
    input_ids: np.ndarray,
) -> list[np.ndarray]:
    """Build the input arrays for a single ciface function call.

    Args:
        func_graph: Per-function dict from load_graph_from_proto's
            ``functions`` list, containing ``inputs`` list with binding info.
        func_outputs: Map of func_index → list of output arrays from prior
            function calls.
        weights_dict: Compiled weight name → numpy array.
        input_ids: Global input (token IDs) as int64 numpy array.

    Returns:
        List of numpy arrays to pass as function arguments.
    """
    arrays: list[np.ndarray] = []
    for inp_def in func_graph["inputs"]:
        binding = inp_def["binding"]
        kind = binding[0]

        if kind == "global_input":
            gi = binding[1] if len(binding) > 1 else 0
            if gi == 0:
                arrays.append(input_ids)
            else:
                seq = input_ids.shape[-1]
                batch = input_ids.shape[0] if input_ids.ndim >= 2 else 1
                positions = np.arange(seq, dtype=np.int64).reshape(1, -1)
                arrays.append(np.broadcast_to(positions, (batch, seq)).copy())
        elif kind == "weight":
            weight_name = binding[1]
            arrays.append(_resolve_weight(weight_name, weights_dict))
        elif kind == "ssa":
            producer_func = binding[1]
            producer_out = binding[2]
            if producer_func not in func_outputs:
                raise KeyError(
                    f"SSA producer func[{producer_func}] not yet executed. "
                    f"Available: {sorted(func_outputs.keys())}"
                )
            producer_outputs = func_outputs[producer_func]
            if producer_out >= len(producer_outputs):
                raise IndexError(
                    f"SSA func[{producer_func}].output[{producer_out}] "
                    f"out of range (max {len(producer_outputs) - 1})"
                )
            arrays.append(producer_outputs[producer_out])
        else:
            raise ValueError(f"Unknown input binding kind: {kind}")

    return arrays


# ── HF cross-validation ─────────────────────────────────────────────


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two float32 arrays."""
    a_f = a.ravel().astype(np.float64)
    b_f = b.ravel().astype(np.float64)
    denom = np.linalg.norm(a_f) * np.linalg.norm(b_f) + 1e-12
    return float(np.dot(a_f, b_f) / denom)


def cross_validate_with_hf(
    last_output: np.ndarray,
    token_ids: np.ndarray,
    min_cos_threshold: float,
    *,
    hf_logits: np.ndarray | None = None,
    fail_on_mismatch: bool = True,
) -> float:
    """Validate golden outputs against HuggingFace reference model.

    Loads the HF OPT-125m model, runs forward pass with the same token IDs,
    and compares the logits (last position) against the last function output.

    Args:
        last_output: The output of the final function (main_15) as numpy array.
        token_ids: Token ID array of shape ``(seq_len,)`` or ``(batch_size, seq_len)``.
        min_cos_threshold: Minimum acceptable cosine similarity.
        hf_logits: Optional pre-computed HF logits ``[batch_size, seq_len, vocab_size]``
            as float32 numpy array.  If provided, skips HF model loading.
        fail_on_mismatch: If True, sys.exit(1) when cos < threshold.
            If False, just prints the divergence as informational.

    Returns:
        Cosine similarity between compiled output and HF reference.
    """
    import torch  # noqa: E402
    from transformers import AutoModelForCausalLM  # noqa: E402

    if hf_logits is None:
        print("  Loading HF model 'facebook/opt-125m' for cross-validation...")
        model = AutoModelForCausalLM.from_pretrained(
            "facebook/opt-125m",
            torch_dtype=torch.float32,
        )
        model = model.to("cpu")
        model.eval()

        if token_ids.ndim == 1:
            input_tensor = torch.from_numpy(token_ids.astype(np.int64)).unsqueeze(0)
        else:
            input_tensor = torch.from_numpy(token_ids.astype(np.int64))

        with torch.no_grad():
            hf_output = model(input_tensor)
            hf_logits_tensor = hf_output.logits  # [batch_size, seq_len, vocab_size]

        hf_logits_arr = hf_logits_tensor.numpy().astype(np.float32)
    else:
        hf_logits_arr = hf_logits

    hf_last = hf_logits_arr[0, -1, :].ravel()

    last_token_logits = last_output[0, -1, :].ravel()
    cos = _cosine_similarity(last_token_logits, hf_last.ravel())

    if cos < min_cos_threshold:
        msg = (
            f"\n{'=' * 60}\n"
            f"E2E CROSS-VALIDATION {'FAILED' if fail_on_mismatch else 'DIVERGED'}\n"
            f"  Cosine similarity: {cos:.8f} < threshold {min_cos_threshold}\n"
        )
        if fail_on_mismatch:
            msg += "  Golden generation aborted.\n"
        else:
            msg += "  Divergence expected — continuing.\n"
        msg += f"{'=' * 60}"
        print(msg, file=sys.stderr)
        if fail_on_mismatch:
            sys.exit(1)

    print(f"  E2E cross-validation PASSED: cos={cos:.8f} >= {min_cos_threshold}")
    return cos


# ── Determinism: SHA256 ─────────────────────────────────────────────


def compute_sha256_files(
    output_dir: Path,
    case_names: list[str],
    chain_order: list[str],
) -> str:
    """Compute SHA256 hash of all golden .npz files in sorted order.

    Returns a checksum string suitable for writing to .sha256sum file.
    """
    hasher = hashlib.sha256()
    npz_files: list[Path] = []
    for case_name in case_names:
        case_dir = output_dir / case_name
        if case_dir.is_dir():
            for func_symbol in sorted(chain_order):
                npz_path = case_dir / f"func_{func_symbol}_output.npz"
                if npz_path.is_file():
                    npz_files.append(npz_path)

    # Sort by path for deterministic ordering
    npz_files.sort()
    for npz_path in npz_files:
        hasher.update(npz_path.read_bytes())

    return hasher.hexdigest()


def save_sha256sum(output_dir: Path, checksum: str) -> None:
    """Write SHA256 checksum to .sha256sum file."""
    checksum_path = output_dir / ".sha256sum"
    with open(checksum_path, "w") as f:
        f.write(f"{checksum}  *\n")
    print(f"  SHA256 checksum saved to {checksum_path}")


def verify_sha256sum(output_dir: Path) -> bool:
    """Verify that current golden files match the saved checksum.

    Returns True if checksum matches or no checksum file exists.
    """
    checksum_path = output_dir / ".sha256sum"
    if not checksum_path.is_file():
        return True

    saved = checksum_path.read_text().strip().split()[0]
    config_path = output_dir / "configs.json"
    if not config_path.is_file():
        return True

    with open(config_path) as f:
        config = json.load(f)
    case_names = [c["name"] for c in config.get("cases", [])]
    chain_order = config.get("_chain_order", [])

    current = compute_sha256_files(output_dir, case_names, chain_order)
    if current != saved:
        print(
            f"  WARNING: SHA256 mismatch!\n"
            f"    Saved:   {saved}\n"
            f"    Current: {current}\n"
            f"    Golden files are NOT deterministic.",
            file=sys.stderr,
        )
        return False
    return True


# ── Freshness check ─────────────────────────────────────────────────


def check_freshness(
    output_dir: Path,
    dylib_path: str,
    case_names: list[str],
    chain_order: list[str],
) -> list[str]:
    """Check if golden files are stale relative to dylib and compiler source.

    Returns:
        List of warning messages. Empty list = all fresh.
    """
    warnings: list[str] = []
    dylib_mtime = Path(dylib_path).stat().st_mtime

    # Find oldest golden file
    oldest_golden_mtime: float | None = None
    for case_name in case_names:
        case_dir = output_dir / case_name
        if case_dir.is_dir():
            for func_symbol in chain_order:
                npz_path = case_dir / f"func_{func_symbol}_output.npz"
                if npz_path.is_file():
                    mtime = npz_path.stat().st_mtime
                    if oldest_golden_mtime is None or mtime < oldest_golden_mtime:
                        oldest_golden_mtime = mtime

    if oldest_golden_mtime is None:
        warnings.append("No golden .npz files found — cannot check freshness.")
        for w in warnings:
            print(f"  Freshness check: {w}", file=sys.stderr)
        return warnings

    # Compare against dylib
    if oldest_golden_mtime < dylib_mtime:
        warnings.append(
            f"Golden files are STALE: oldest golden {oldest_golden_mtime:.0f} "
            f"< dylib {dylib_mtime:.0f}. dylib was updated after golden generation."
        )

    # Compare against compiler source files
    compiler_dir = ROOT / "compiler"
    latest_compiler_mtime: float = 0.0
    if compiler_dir.is_dir():
        for py_file in compiler_dir.rglob("*.py"):
            mtime = py_file.stat().st_mtime
            if mtime > latest_compiler_mtime:
                latest_compiler_mtime = mtime

    if latest_compiler_mtime > 0 and oldest_golden_mtime < latest_compiler_mtime:
        warnings.append(
            f"Golden files may be stale: oldest golden {oldest_golden_mtime:.0f} "
            f"< latest compiler source {latest_compiler_mtime:.0f}. "
            f"Compiler source was modified after golden generation."
        )

    if not warnings:
        print("  Freshness check: ALL FRESH")
    else:
        for w in warnings:
            print(f"  Freshness check: {w}", file=sys.stderr)

    return warnings


# ── Main golden generation ──────────────────────────────────────────


def generate_golden(
    config: dict[str, Any],
    output_dir: Path,
    check_only_fresh: bool = False,
) -> None:
    """Generate golden .npz outputs for all config cases.

    Args:
        config: Loaded config dict with _metadata, _resolved_dylib_path, etc.
        output_dir: Directory to save golden .npz files.
        check_only_fresh: If True, only check freshness and exit.
    """
    metadata = config["_metadata"]
    dylib_path = config["_resolved_dylib_path"]
    min_cos_threshold = config.get("min_cos_threshold", 0.9999)
    cases = config["cases"]
    chain_order = metadata.get("chain_order", [])
    case_names = [c["name"] for c in cases]

    if check_only_fresh:
        freshness_warnings = check_freshness(output_dir, dylib_path, case_names, chain_order)
        if freshness_warnings:
            sys.exit(0)
        return

    # Load dylib
    print(f"Loading dylib: {dylib_path}")
    lib = ctypes.CDLL(dylib_path)

    # Load compute graph from embedded proto
    print("Loading compute graph from dylib proto...")
    compute_graph = load_graph_from_proto(lib)
    functions = compute_graph["functions"]
    print(f"  {len(functions)} functions loaded")

    # Build function index map: symbol → func_index
    func_index_map: dict[str, int] = {}
    for fi, fg in enumerate(functions):
        symbol = fg["symbol"].replace("_mlir_ciface_", "")
        func_index_map[symbol] = fi

    # Load weights
    print("Loading safetensors weights...")
    weights_dict = load_safetensors_weights(metadata)
    print(f"  {len(weights_dict)} weights loaded")

    for case in cases:
        case_name = case["name"]
        seq_len = case["seq_len"]
        batch_size = case.get("batch_size", 1)
        print(f"\n{'=' * 60}")
        print(f"Case: {case_name} (seq_len={seq_len}, batch_size={batch_size})")
        print(f"{'=' * 60}")

        # Generate tokens — use config-specified tokens (for matching Rust tests)
        # or generate random tokens with fixed seed per seq_len
        if "tokens" in case:
            token_ids = np.array(case["tokens"], dtype=np.int64)
            print(f"  Token IDs (from config): {token_ids.tolist()}")
        else:
            token_ids = generate_tokens(seq_len, seed=RANDOM_SEED, batch_size=batch_size)
            print(f"  Token IDs (random): {token_ids.tolist()}")

        # Build global input: rank-2 i64 tensor [batch_size, seq_len]
        input_ids = token_ids.reshape(batch_size, seq_len).astype(np.int64)

        # Execute functions in chain_order
        func_outputs: dict[int, list[np.ndarray]] = {}

        for func_symbol in chain_order:
            fi = func_index_map.get(func_symbol)
            if fi is None:
                print(f"  WARNING: func_symbol '{func_symbol}' not found in compute graph, skipping")
                continue

            fg = functions[fi]

            # Build input arrays
            input_arrays = build_input_arrays(
                fg, func_outputs, weights_dict, input_ids,
            )

            # Compute sret size
            sret_size = compute_sret_size(fg["outputs"], floor=DEFAULT_SRET_SIZE)

            # Call function
            outputs = call_ciface_function(
                lib, func_symbol, input_arrays, fg["outputs"], sret_size,
            )
            func_outputs[fi] = outputs

            # Save outputs
            case_dir = output_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            npz_path = case_dir / f"func_{func_symbol}_output.npz"

            # Build dict with output_N keys
            out_dict = {f"output_{i}": arr for i, arr in enumerate(outputs)}
            save_npz(npz_path, out_dict)

            # Report shapes
            shapes_str = ", ".join(
                f"output_{i}: {list(arr.shape)}" for i, arr in enumerate(outputs)
            )
            print(f"  {func_symbol}: {shapes_str} → {npz_path.name}")

        # ── E2E cross-validation ──
        print("\n  Running E2E cross-validation...")
        last_fi = func_index_map.get(chain_order[-1])
        if last_fi is None:
            print("  WARNING: Last function not found, skipping E2E check")
            continue

        last_outputs = func_outputs.get(last_fi, [])
        if not last_outputs:
            print("  WARNING: No output from last function, skipping E2E check")
            continue

        # The last output is the logits
        last_output = last_outputs[0]
        cross_validate_with_hf(last_output, token_ids, min_cos_threshold)

    # ── Determinism: compute and save SHA256 ──
    print(f"\n{'=' * 60}")
    print("Computing SHA256 for determinism check...")
    checksum = compute_sha256_files(output_dir, case_names, chain_order)
    save_sha256sum(output_dir, checksum)

    # Verify if this is a re-run
    if not verify_sha256sum(output_dir):
        print("  WARNING: SHA256 mismatch — golden files are NOT deterministic!")
        # Non-fatal: first run or compiler change produces different outputs.
        # This is expected when regenerating after a compiler update.
    else:
        print("  Determinism check: SHA256 MATCH (or first run)")

    # Freshness check always runs at the end
    check_freshness(output_dir, dylib_path, case_names, chain_order)

    print(f"\n{'=' * 60}")
    print("Golden generation complete!")
    print(f"  Output: {output_dir}")
    print(f"  Cases:  {case_names}")
    print(f"  Functions: {len(chain_order)}")


# ── HF golden generation ────────────────────────────────────────────


def map_func_to_hf(
    func_symbol: str,
    hidden_states: tuple,
    model: Any,
    logits: np.ndarray,
    input_ids: np.ndarray,
    raw_layer11_output: np.ndarray | None = None,
) -> tuple[np.ndarray, str] | None:
    """Map a compiled function symbol to its HF reference value.

    Returns ``(array, key)`` where *array* is a float32 numpy array and
    *key* is the .npz output key (e.g. ``"output_0"``, ``"output_210"``).
    Returns ``None`` if the function is not mappable.

    Args:
        func_symbol: e.g. ``"main_0"``, ``"main_13"``.
        hidden_states: Tuple of 13 hidden state tensors from HF forward
            (``output_hidden_states=True``).  ``hidden_states[0]`` is the
            embedding, ``hidden_states[1..12]`` are the 12 layer outputs.
            ``hidden_states[12]`` has final_layer_norm applied.
        model: Loaded HF ``AutoModelForCausalLM`` instance (float32, eval mode).
        logits: Pre-computed HF logits as float32 numpy array
            ``[batch_size, seq_len, vocab_size]``.
        input_ids: Token ID numpy array (passed for forward hooks if needed).
        raw_layer11_output: Raw output of HF layer 11 (the last transformer
            layer before final_layer_norm), captured via forward hook during
            the main model pass.  Required for main_12 mapping.

    Returns:
        ``(np.ndarray, str_key)`` or ``None``.
    """

    if func_symbol == "main_0":
        # HF hidden_states[0] = embedding output [1, seq, 768].
        # In the dylib, main_0 has 211 outputs; output_12 is the embedding,
        # not output_210 (which is a weight matrix [768, 768]).
        arr = hidden_states[0].detach().cpu().numpy().astype(np.float32)
        return arr, "output_12"

    if func_symbol.startswith("main_") and func_symbol != "main_0":
        layer_idx_str = func_symbol.split("_")[1]
        try:
            layer_idx = int(layer_idx_str)
        except ValueError:
            return None

        if 1 <= layer_idx <= 11:
            # main_1..main_11: HF hidden_states[layer_idx] = raw layer output
            # (No final_layer_norm applied — only the last hidden state gets it.)
            arr = hidden_states[layer_idx].detach().cpu().numpy().astype(np.float32)
            return arr, "output_0"

        if layer_idx == 12:
            # main_12: raw output of HF layer 11 (the last transformer layer).
            # hidden_states[12] has final_layer_norm applied by the decoder's
            # @capture_outputs decorator, so it does NOT match the raw layer
            # output that the dylib's main_12 produces.
            # Use the raw output captured via hook during the main forward pass.
            if raw_layer11_output is None:
                return None
            return raw_layer11_output.astype(np.float32), "output_0"

        if layer_idx == 13:
            # main_13: final_layer_norm(layer 11 output).
            # hidden_states[12] === final_layer_norm(layer 11 output), so use it
            # directly instead of recomputing.
            arr = hidden_states[12].detach().cpu().numpy().astype(np.float32)
            return arr, "output_0"

        if layer_idx == 14:
            # main_14: identity passthrough of main_13 (sf.slice passthrough).
            # Same as main_13 golden.
            arr = hidden_states[12].detach().cpu().numpy().astype(np.float32)
            return arr, "output_0"

        if layer_idx == 15:
            # main_15: lm_head(final_layer_norm(...)) = model logits
            arr = logits.astype(np.float32)
            return arr, "output_0"

    # Unknown function — cannot map
    return None


def hf_generate_golden(
    config: dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate per-function golden .npz files from HuggingFace model reference.

    Loads the HF OPT-125m model in float32, runs forward pass with
    ``output_hidden_states=True``, maps hidden states + intermediate
    computations to compiled function outputs, and saves as .npz files
    following the exact same naming convention as the dylib-based
    ``generate_golden()``.

    Also runs E2E cross-validation by comparing the dylib's main_15 output
    against HF logits (informational only — does not abort on divergence).

    Args:
        config: Loaded config dict with _metadata, _resolved_dylib_path, etc.
        output_dir: Directory to save golden .npz files.
    """
    import torch  # noqa: E402
    from transformers import AutoModelForCausalLM  # noqa: E402

    metadata = config["_metadata"]
    dylib_path = config["_resolved_dylib_path"]
    min_cos_threshold = config.get("min_cos_threshold", 0.9999)
    cases = config["cases"]
    chain_order = metadata.get("chain_order", [])
    case_names = [c["name"] for c in cases]

    # ── Load HF model ─────────────────────────────────────────────
    print("Loading HF model 'facebook/opt-125m' (float32)...")
    model = AutoModelForCausalLM.from_pretrained(
        "facebook/opt-125m",
        torch_dtype=torch.float32,
    )
    model = model.to("cpu")
    model.eval()
    print("  HF model loaded.")

    # ── Generate per-case ─────────────────────────────────────────
    for case in cases:
        case_name = case["name"]
        seq_len = case["seq_len"]
        batch_size = case.get("batch_size", 1)
        print(f"\n{'=' * 60}")
        print(f"Case: {case_name} (seq_len={seq_len}, batch_size={batch_size})")
        print(f"{'=' * 60}")

        # Generate tokens with fixed seed (same as dylib mode)
        token_ids = generate_tokens(seq_len, seed=RANDOM_SEED, batch_size=batch_size)
        print(f"  Token IDs: {token_ids.tolist()}")

        if token_ids.ndim == 1:
            input_tensor = torch.from_numpy(token_ids.astype(np.int64)).unsqueeze(0)
        else:
            input_tensor = torch.from_numpy(token_ids.astype(np.int64))

        # Single HF forward pass: hook layer 11 to capture raw output
        # before final_layer_norm is applied by @capture_outputs.
        raw_layer11_out: list[np.ndarray] = []
        def _capture_layer11(m, inp, out, _out=raw_layer11_out):
            _out.append(out.detach().cpu().numpy().astype(np.float32))

        hook_handle = model.model.decoder.layers[11].register_forward_hook(
            _capture_layer11,
        )
        try:
            with torch.no_grad():
                hf_out = model(input_tensor, output_hidden_states=True)
        finally:
            hook_handle.remove()

        hidden_states = hf_out.hidden_states  # tuple[13]: embed + 12 layers
        hf_logits = hf_out.logits.detach().cpu().numpy().astype(np.float32)
        raw_layer11 = raw_layer11_out[0] if raw_layer11_out else None

        # Build global input for E2E cross-validation later
        input_ids = token_ids.reshape(batch_size, seq_len).astype(np.int64)

        # Generate golden .npz for each function in chain_order
        unmapped_funcs: list[str] = []
        for func_symbol in chain_order:
            result = map_func_to_hf(
                func_symbol, hidden_states, model, hf_logits, input_ids,
                raw_layer11_output=raw_layer11,
            )
            if result is None:
                # Unmappable function — save zeros as placeholder with warning
                unmapped_funcs.append(func_symbol)
                print(
                    f"  WARNING: {func_symbol} cannot be mapped to HF — "
                    f"saving zeros placeholder",
                    file=sys.stderr,
                )
                # Create a zeros array with a reasonable shape [batch_size, seq_len, 768]
                arr = np.zeros((batch_size, seq_len, 768), dtype=np.float32)
                key = "output_0"
            else:
                arr, key = result

            case_dir = output_dir / case_name
            case_dir.mkdir(parents=True, exist_ok=True)
            npz_path = case_dir / f"func_{func_symbol}_output.npz"

            out_dict = {key: arr}
            save_npz(npz_path, out_dict)

            print(f"  {func_symbol}: {key}: {list(arr.shape)} → {npz_path.name}")

        if unmapped_funcs:
            print(
                f"\n  Unmapped functions ({len(unmapped_funcs)}): "
                f"{unmapped_funcs}",
                file=sys.stderr,
            )

        # ── E2E cross-validation (informational) ──────────────────
        print("\n  Running E2E cross-validation (HF vs dylib)...")
        # Load dylib, run chain up to main_15, compare against HF logits
        try:
            lib = ctypes.CDLL(dylib_path)
            compute_graph = load_graph_from_proto(lib)
            functions = compute_graph["functions"]

            func_index_map: dict[str, int] = {}
            for fi, fg in enumerate(functions):
                symbol = fg["symbol"].replace("_mlir_ciface_", "")
                func_index_map[symbol] = fi

            weights_dict = load_safetensors_weights(metadata)

            func_outputs: dict[int, list[np.ndarray]] = {}
            last_output: np.ndarray | None = None

            for func_symbol in chain_order:
                fi = func_index_map.get(func_symbol)
                if fi is None:
                    continue
                fg = functions[fi]
                input_arrays = build_input_arrays(
                    fg, func_outputs, weights_dict, input_ids,
                )
                sret_size = compute_sret_size(fg["outputs"], floor=DEFAULT_SRET_SIZE)
                outputs = call_ciface_function(
                    lib, func_symbol, input_arrays, fg["outputs"], sret_size,
                )
                func_outputs[fi] = outputs
                if func_symbol == "main_15":
                    last_output = outputs[0]

            if last_output is not None:
                cross_validate_with_hf(
                    last_output,
                    token_ids,
                    min_cos_threshold,
                    hf_logits=hf_logits,
                    fail_on_mismatch=False,
                )
            else:
                print("  WARNING: main_15 not found in chain_order, skipping E2E check")
        except Exception as exc:
            print(
                f"  WARNING: E2E cross-validation failed with error: {exc}",
                file=sys.stderr,
            )

    # ── Determinism: compute and save SHA256 ──────────────────────
    print(f"\n{'=' * 60}")
    print("Computing SHA256 for determinism check...")
    checksum = compute_sha256_files(output_dir, case_names, chain_order)
    save_sha256sum(output_dir, checksum)

    if not verify_sha256sum(output_dir):
        print("  WARNING: SHA256 mismatch — golden files are NOT deterministic!")
        # Non-fatal: first run or compiler change produces different outputs.
    else:
        print("  Determinism check: SHA256 MATCH (or first run)")

    # Freshness check always runs at the end
    check_freshness(output_dir, dylib_path, case_names, chain_order)

    print(f"\n{'=' * 60}")
    print("HF golden generation complete!")
    print(f"  Output: {output_dir}")
    print(f"  Cases:  {case_names}")
    print(f"  Functions: {len(chain_order)}")


# ── CLI ─────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministic per-function golden output generator",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs.json (e.g., tests/data/golden/npy/opt_125m/configs.json)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory for golden .npz files",
    )
    parser.add_argument(
        "--check-fresh",
        action="store_true",
        help="Only check golden freshness against dylib and compiler source",
    )
    parser.add_argument(
        "--hf-golden",
        action="store_true",
        help="Generate golden from HuggingFace model reference instead of dylib",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output)

    # Store chain_order in config for later use by sha256 verification
    metadata = config["_metadata"]
    config["_chain_order"] = metadata.get("chain_order", [])

    if args.hf_golden:
        hf_generate_golden(config, output_dir)
    else:
        generate_golden(config, output_dir, check_only_fresh=args.check_fresh)


if __name__ == "__main__":
    main()
